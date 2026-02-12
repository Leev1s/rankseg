# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections.abc
from itertools import combinations

import numpy as np
import cv2
import paddle
import paddle.nn.functional as F
from skimage import measure, morphology
from paddleseg.utils import logger


def _generate_pred_from_logit(logit,
                               use_multilabel=False,
                               use_rankseg=False,
                               rankseg_metric='dice',
                               rankseg_solver='RMA',
                               rankseg_output_mode='multiclass',
                               **rankseg_kwargs):
    """
    Generate prediction from logit.

    Args:
        logit (Tensor): Model output logit, shape (batch, num_classes, H, W).
        use_multilabel (bool): Whether to use multilabel mode.
        use_rankseg (bool): Whether to use RankSEG optimization.
        rankseg_metric (str): RankSEG optimization metric ('dice', 'IoU', 'Acc').
        rankseg_solver (str): RankSEG solver ('RMA', 'BA', 'TRNA', 'BA+TRNA').
        rankseg_output_mode (str): RankSEG output mode ('multiclass', 'multilabel').
        **rankseg_kwargs: Additional parameters for RankSEG (e.g., smooth, pruning_prob).

    Returns:
        Tensor: Prediction result.
            - If use_rankseg=True and rankseg_output_mode='multiclass':
              Shape (batch, 1, H, W), values are class indices.
            - If use_rankseg=True and rankseg_output_mode='multilabel':
              Shape (batch, num_classes, H, W), values are 0/1.
            - If use_rankseg=False and use_multilabel=False:
              Shape (batch, 1, H, W), values are class indices (argmax).
            - If use_rankseg=False and use_multilabel=True:
              Shape (batch, num_classes, H, W), values are 0/1 (sigmoid > 0.5).
    """
    if not use_rankseg:
        # Original logic
        if not use_multilabel:
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
        else:
            pred = (F.sigmoid(logit) > 0.5).astype('int32')
    else:
        # RankSEG optimization
        import torch
        from rankseg import RankSEG

        # 1. Paddle Tensor -> NumPy -> PyTorch Tensor
        logit_np = logit.numpy()
        logit_torch = torch.from_numpy(logit_np)

        # 2. Compute softmax probabilities
        probs = torch.softmax(logit_torch, dim=1)

        # 3. RankSEG optimization
        rankseg = RankSEG(
            metric=rankseg_metric,
            solver=rankseg_solver,
            output_mode=rankseg_output_mode,
            **rankseg_kwargs
        )
        pred_torch = rankseg.predict(probs)

        # 4. PyTorch Tensor -> NumPy -> Paddle Tensor
        pred_np = pred_torch.cpu().numpy()

        # 5. Handle output shape
        if rankseg_output_mode == 'multiclass':
            # RankSEG multiclass output: (batch, H, W)
            # Convert to (batch, 1, H, W) for consistency
            if pred_np.ndim == 3:
                pred_np = pred_np[:, np.newaxis, :, :]
            pred = paddle.to_tensor(pred_np, dtype='int32')
        else:
            # RankSEG multilabel output: (batch, num_classes, H, W)
            pred = paddle.to_tensor(pred_np, dtype='int32')

    return pred


def reverse_transform(pred, trans_info, mode='nearest'):
    """recover pred to origin shape"""
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in trans_info[::-1]:
        if isinstance(item[0], list):
            trans_mode = item[0][0]
        else:
            trans_mode = item[0]
        if trans_mode == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu' and dtype in intTypeList:
                pred = paddle.cast(pred, 'float32')
                pred = F.interpolate(pred, [h, w], mode=mode)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(pred, [h, w], mode=mode)
        elif trans_mode == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred


def flip_combination(flip_horizontal=False, flip_vertical=False):
    """
    Get flip combination.

    Args:
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.

    Returns:
        list: List of tuple. The first element of tuple is whether to flip horizontally,
            and the second is whether to flip vertically.
    """

    flip_comb = [(False, False)]
    if flip_horizontal:
        flip_comb.append((True, False))
    if flip_vertical:
        flip_comb.append((False, True))
        if flip_horizontal:
            flip_comb.append((True, True))
    return flip_comb


def stfpm_post_transform(score_map):
    mask = score_map[0][0]
    mask[mask > 0.01] = 1
    mask[mask <= 0.01] = 0
    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    mask = paddle.to_tensor(mask[None, None, :, :])
    return paddle.concat([1 - mask, mask], axis=1)


def tensor_flip(x, flip):
    """Flip tensor according directions"""
    if flip[0]:
        x = x[:, :, :, ::-1]
    if flip[1]:
        x = x[:, :, ::-1, :]
    return x


def slide_inference(model, im, crop_size, stride):
    """
    Infer by sliding window.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        crop_size (tuple|list). The size of sliding window, (w, h).
        stride (tuple|list). The size of stride, (w, h).

    Return:
        Tensor: The logit of input image.
    """
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    # calculate the crop nums
    rows = int(np.ceil(1.0 * (h_im - h_crop) / h_stride)) + 1
    cols = int(np.ceil(1.0 * (w_im - w_crop) / w_stride)) + 1
    # prevent negative sliding rounds when imgs after scaling << crop_size
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols
    # TODO 'Tensor' object does not support item assignment. If support, use tensor to calculation.
    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logits = model(im_crop)
            if not isinstance(logits, collections.abc.Sequence):
                raise TypeError(
                    "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                    .format(type(logits)))
            logit = logits[0].numpy()
            if final_logit is None:
                final_logit = np.zeros([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RuntimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    final_logit = paddle.to_tensor(final_logit)
    return final_logit


def inference(model,
              im,
              trans_info=None,
              is_slide=False,
              stride=None,
              crop_size=None,
              use_multilabel=False,
              use_rankseg=False,
              rankseg_metric='dice',
              rankseg_solver='RMA',
              rankseg_output_mode='multiclass',
              **rankseg_kwargs):
    """
    Inference for image.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Image shape informating changed process. Default: None.
        is_slide (bool): Whether to infer by sliding window. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.
        use_rankseg (bool, optional): Whether to use RankSEG optimization. Default: False.
        rankseg_metric (str, optional): RankSEG optimization metric. Default: 'dice'.
        rankseg_solver (str, optional): RankSEG solver. Default: 'RMA'.
        rankseg_output_mode (str, optional): RankSEG output mode. Default: 'multiclass'.
        **rankseg_kwargs: Additional parameters for RankSEG.

    Returns:
        Tensor: If trans_info is not None, a prediction with shape (1, 1, h, w) or (1, num_classes, h, w) is returned.
        Tensor: logit with shape (1, num_classes, h, w) - used for metrics calculation in val.py.
            If trans_info is None, only logit with shape (1, num_classes, h, w) is returned.
    """
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        im = im.transpose((0, 2, 3, 1))
    if not is_slide:
        logits = model(im)
        if not isinstance(logits, collections.abc.Sequence):
            raise TypeError(
                "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                .format(type(logits)))
        logit = logits[0]
    else:
        logit = slide_inference(model, im, crop_size=crop_size, stride=stride)
    if hasattr(model, 'data_format') and model.data_format == 'NHWC':
        logit = logit.transpose((0, 3, 1, 2))
    if model.__class__.__name__ == 'STFPM':
        logit = stfpm_post_transform(logit)
    if trans_info is not None:
        logit = reverse_transform(logit, trans_info, mode='bilinear')

        # Generate prediction using helper function
        pred = _generate_pred_from_logit(
            logit,
            use_multilabel=use_multilabel,
            use_rankseg=use_rankseg,
            rankseg_metric=rankseg_metric,
            rankseg_solver=rankseg_solver,
            rankseg_output_mode=rankseg_output_mode,
            **rankseg_kwargs
        )

        return pred, logit
    else:
        return logit


def aug_inference(model,
                  im,
                  trans_info,
                  scales=1.0,
                  flip_horizontal=False,
                  flip_vertical=False,
                  is_slide=False,
                  stride=None,
                  crop_size=None,
                  use_multilabel=False,
                  use_rankseg=False,
                  rankseg_metric='dice',
                  rankseg_solver='RMA',
                  rankseg_output_mode='multiclass',
                  **rankseg_kwargs):
    """
    Infer with augmentation.

    Args:
        model (paddle.nn.Layer): model to get logits of image.
        im (Tensor): the input image.
        trans_info (list): Transforms for image.
        scales (float|tuple|list):  Scales for resize. Default: 1.
        flip_horizontal (bool): Whether to flip horizontally. Default: False.
        flip_vertical (bool): Whether to flip vertically. Default: False.
        is_slide (bool): Whether to infer by sliding wimdow. Default: False.
        crop_size (tuple|list). The size of sliding window, (w, h). It should be probided if is_slide is True.
        stride (tuple|list). The size of stride, (w, h). It should be probided if is_slide is True.
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.
        use_rankseg (bool, optional): Whether to use RankSEG optimization. Default: False.
        rankseg_metric (str, optional): RankSEG optimization metric. Default: 'dice'.
        rankseg_solver (str, optional): RankSEG solver. Default: 'RMA'.
        rankseg_output_mode (str, optional): RankSEG output mode. Default: 'multiclass'.
        **rankseg_kwargs: Additional parameters for RankSEG.

    Returns:
        Tensor: Prediction of image with shape (1, 1, h, w) or (1, num_classes, h, w).
        Tensor: logit with shape (1, num_classes, h, w).
    """
    if isinstance(scales, float):
        scales = [scales]
    elif not isinstance(scales, (tuple, list)):
        raise TypeError(
            '`scales` expects float/tuple/list type, but received {}'.format(
                type(scales)))
    final_logit = 0
    h_input, w_input = im.shape[-2], im.shape[-1]
    flip_comb = flip_combination(flip_horizontal, flip_vertical)
    num_augs = len(scales) * len(flip_comb)
    for scale in scales:
        h = int(h_input * scale + 0.5)
        w = int(w_input * scale + 0.5)
        im_scale = F.interpolate(im, [h, w], mode='bilinear')
        for flip in flip_comb:
            im_flip = tensor_flip(im_scale, flip)
            logit = inference(model,
                              im_flip,
                              is_slide=is_slide,
                              crop_size=crop_size,
                              stride=stride)
            logit = tensor_flip(logit, flip)
            logit = F.interpolate(logit, [h_input, w_input], mode='bilinear')
            # Accumulate final logits in place
            final_logit += logit
    # We average the accumulated logits to make the numeric values of `final_logit`
    # comparable to single-scale logits
    final_logit /= num_augs
    final_logit = reverse_transform(final_logit, trans_info, mode='bilinear')

    # Generate prediction using helper function
    pred = _generate_pred_from_logit(
        final_logit,
        use_multilabel=use_multilabel,
        use_rankseg=use_rankseg,
        rankseg_metric=rankseg_metric,
        rankseg_solver=rankseg_solver,
        rankseg_output_mode=rankseg_output_mode,
        **rankseg_kwargs
    )

    return pred, final_logit
