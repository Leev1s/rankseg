import torch

from rankseg.mmseg import postprocess_mmseg, restore_semantic_probs_from_mmseg


class _PixelData:
    def __init__(self, data):
        self.data = data


class _Sample:
    def __init__(self, *, seg_logits=None, pred_sem_seg=None):
        self.seg_logits = _PixelData(seg_logits) if seg_logits is not None else None
        self.pred_sem_seg = _PixelData(pred_sem_seg) if pred_sem_seg is not None else None


def test_restore_semantic_probs_from_mmseg_multiclass_softmax():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [1.0, -1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[-1.0, 0.5], [0.5, 0.5]],
        ]
    )
    sample = _Sample(seg_logits=logits)
    probs = restore_semantic_probs_from_mmseg(sample)

    assert probs.shape == (1, 3, 2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-6)


def test_restore_semantic_probs_from_mmseg_binary_auto_probs_passthrough():
    probs_in = torch.tensor([[[0.1, 0.8], [0.2, 0.4]]])
    sample = _Sample(seg_logits=probs_in)
    probs = restore_semantic_probs_from_mmseg(sample, binary_mode="auto")

    assert probs.shape == (1, 1, 2, 2)
    assert torch.allclose(probs.squeeze(0), probs_in)


def test_restore_semantic_probs_from_mmseg_binary_logits_sigmoid():
    logits = torch.tensor([[[-2.0, 0.0], [2.0, 4.0]]])
    sample = _Sample(seg_logits=logits)
    probs = restore_semantic_probs_from_mmseg(sample, binary_mode="logits")

    assert probs.shape == (1, 1, 2, 2)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_restore_semantic_probs_from_mmseg_pred_sem_seg_fallback_one_hot():
    labels = torch.tensor([[0, 1], [1, 2]])
    sample = _Sample(pred_sem_seg=labels)
    probs = restore_semantic_probs_from_mmseg(sample, num_classes=3)

    assert probs.shape == (1, 3, 2, 2)
    assert torch.equal(probs.argmax(dim=1).squeeze(0), labels)


def test_postprocess_mmseg_runs_rankseg_with_defaults():
    logits = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 0.5]],
            [[0.1, 0.0], [0.2, 0.4]],
            [[0.0, -1.0], [1.0, 0.8]],
        ]
    )
    sample = _Sample(seg_logits=logits)

    preds = postprocess_mmseg(sample)
    assert preds.shape == (1, 2, 2)
