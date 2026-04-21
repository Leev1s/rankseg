from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from ._rankseg import RankSEG


def _get_field(sample, name: str):
    value = getattr(sample, name, None)
    if value is not None:
        return value
    if isinstance(sample, Mapping):
        return sample.get(name)
    return None


def _to_tensor(field, field_name: str) -> torch.Tensor:
    if field is None:
        raise ValueError(f"`{field_name}` is missing.")

    if isinstance(field, torch.Tensor):
        return field

    data = getattr(field, "data", None)
    if isinstance(data, torch.Tensor):
        return data

    if isinstance(field, Mapping):
        data = field.get("data")
        if isinstance(data, torch.Tensor):
            return data

    raise TypeError(
        f"`{field_name}` must be a torch.Tensor, PixelData-like object with `.data`, or mapping with `data` tensor."
    )


def _as_4d(t: torch.Tensor, *, field_name: str, allow_class_axis: bool = True) -> torch.Tensor:
    if t.ndim == 4:
        return t
    if t.ndim == 3:
        if allow_class_axis:
            return t.unsqueeze(0)
        return t.unsqueeze(1)
    if t.ndim == 2:
        return t.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"`{field_name}` must have 2, 3, or 4 dimensions, but got shape {tuple(t.shape)}.")


def _labels_to_one_hot(labels: torch.Tensor, num_classes: int | None) -> torch.Tensor:
    labels = labels.long()
    labels_4d = _as_4d(labels, field_name="pred_sem_seg", allow_class_axis=False)

    if labels_4d.shape[1] != 1:
        raise ValueError(
            "`pred_sem_seg` fallback expects label maps with one channel, got shape "
            f"{tuple(labels_4d.shape)}."
        )

    labels_3d = labels_4d[:, 0]
    if num_classes is None:
        if labels_3d.numel() == 0:
            raise ValueError("`pred_sem_seg` is empty; unable to infer `num_classes`.")
        inferred = int(labels_3d.max().item()) + 1
        num_classes = max(inferred, 1)

    if num_classes <= 0:
        raise ValueError("`num_classes` must be a positive integer.")

    if labels_3d.numel() and int(labels_3d.max().item()) >= num_classes:
        raise ValueError(
            "`pred_sem_seg` contains class ids outside `num_classes`. "
            f"max label={int(labels_3d.max().item())}, num_classes={num_classes}."
        )

    one_hot = F.one_hot(labels_3d.clamp_min(0), num_classes=num_classes)
    return one_hot.permute(0, 3, 1, 2).float()


def _binary_probs_from_single_channel(scores: torch.Tensor, binary_mode: str) -> torch.Tensor:
    mode = binary_mode.strip().lower()
    if mode not in {"auto", "logits", "probs"}:
        raise ValueError("`binary_mode` must be one of {'auto', 'logits', 'probs'}.")

    scores = scores.float()
    if mode == "logits":
        return scores.sigmoid()
    if mode == "probs":
        return scores.clamp(0.0, 1.0)

    finite = bool(torch.isfinite(scores).all())
    in_unit_interval = bool(((scores >= 0) & (scores <= 1)).all())
    if finite and in_unit_interval:
        return scores
    return scores.sigmoid()


def _restore_single(data_sample, *, num_classes: int | None, binary_mode: str) -> torch.Tensor:
    seg_logits = _get_field(data_sample, "seg_logits")
    if seg_logits is not None:
        logits = _to_tensor(seg_logits, "seg_logits")
        logits = _as_4d(logits, field_name="seg_logits")

        if logits.shape[1] > 1:
            probs = logits.float().softmax(dim=1)
        else:
            probs = _binary_probs_from_single_channel(logits, binary_mode=binary_mode)

        if not bool(torch.isfinite(probs).all()):
            raise ValueError("Converted probabilities contain non-finite values.")
        if bool(torch.any((probs < 0) | (probs > 1))):
            raise ValueError("Converted probabilities must be in [0, 1].")
        return probs

    pred_sem_seg = _get_field(data_sample, "pred_sem_seg")
    if pred_sem_seg is not None:
        labels = _to_tensor(pred_sem_seg, "pred_sem_seg")
        probs = _labels_to_one_hot(labels, num_classes=num_classes)
        return probs

    raise ValueError("Neither `seg_logits` nor `pred_sem_seg` was found in input sample.")


def restore_semantic_probs_from_mmseg(data_sample, *, num_classes: int | None = None, binary_mode: str = "auto"):
    """Restore semantic probabilities from MMSeg-style prediction outputs.

    Parameters
    ----------
    data_sample : object | Mapping | Sequence
        A SegDataSample-like object (or mapping) with `seg_logits` and/or
        `pred_sem_seg`. A sequence of samples is also supported.

    num_classes : int, optional
        Required only when using `pred_sem_seg` fallback and class count cannot
        be inferred reliably from labels.

    binary_mode : {'auto', 'logits', 'probs'}, default='auto'
        How to interpret single-channel `seg_logits`:
        - 'logits': always apply sigmoid
        - 'probs': assume already probabilities
        - 'auto': treat as probabilities if all values are within [0, 1],
          otherwise apply sigmoid

    Returns
    -------
    torch.Tensor
        Probability tensor with shape (B, C, H, W).
    """
    if isinstance(data_sample, Sequence) and not isinstance(data_sample, (str, bytes, Mapping)):
        if len(data_sample) == 0:
            raise ValueError("`data_sample` sequence is empty.")
        probs_list = [_restore_single(sample, num_classes=num_classes, binary_mode=binary_mode) for sample in data_sample]
        return torch.cat(probs_list, dim=0)

    return _restore_single(data_sample, num_classes=num_classes, binary_mode=binary_mode)


def postprocess_mmseg(
    data_sample,
    *,
    rankseg_kwargs: dict | None = None,
    num_classes: int | None = None,
    binary_mode: str = "auto",
):
    """Apply RankSEG on MMSeg-style outputs and return optimized predictions."""
    if rankseg_kwargs is None:
        rankseg_kwargs = {}
    elif not isinstance(rankseg_kwargs, dict):
        raise ValueError("`rankseg_kwargs` must be a dictionary.")

    probs = restore_semantic_probs_from_mmseg(data_sample, num_classes=num_classes, binary_mode=binary_mode)

    if "metric" not in rankseg_kwargs:
        rankseg_kwargs = {**rankseg_kwargs, "metric": "dice"}
    if "solver" not in rankseg_kwargs:
        rankseg_kwargs = {**rankseg_kwargs, "solver": "RMA"}
    if "output_mode" not in rankseg_kwargs:
        rankseg_kwargs = {
            **rankseg_kwargs,
            "output_mode": "multilabel" if probs.shape[1] == 1 else "multiclass",
        }

    predictor = RankSEG(**rankseg_kwargs)
    return predictor.predict(probs)
