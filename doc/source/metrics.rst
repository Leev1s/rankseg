.. _metrics:

Metrics
=======

This page summarizes the definitions used in RankSEG for Dice, IoU, and Accuracy, and clarifies aggregation conventions.

Aggregation Convention
----------------------

- RankSEG optimizes metrics under a samplewise aggregation: the metric is computed per sample and then averaged across the dataset/batch.
- This matches TorchMetrics' ``aggregation_level='samplewise'`` for segmentation metrics.

Notation
--------

- Let y_c and p_c be the binary ground-truth mask and predicted binary mask at class c for a given sample.
- TP, FP, FN are computed per class and per sample.
- For multi-class, masks are mutually exclusive. For multi-label, masks can overlap.

Dice (F1 for segmentation)
--------------------------

Per-class Dice for a sample:

.. math::
   \mathrm{Dice}_c = \frac{2\,\mathrm{TP}_c}{2\,\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c + \epsilon}

- Range: [0, 1]. Higher is better.
- ``epsilon`` is a small constant to avoid division by zero (controlled via ``smooth`` in RankSEG).

IoU (Jaccard Index)
-------------------

Per-class IoU for a sample:

.. math::
   \mathrm{IoU}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c + \epsilon}

- Range: [0, 1]. Higher is better.
- ``epsilon`` avoids division by zero.

Averaging Across Classes
------------------------

Given per-class scores for a sample, common reductions are:

- Mean over classes (macro average)
- Weighted mean (e.g., by class frequency)

RankSEG itself optimizes the per-pixel decisions to improve the chosen metric under the requested task setting (multi-class vs multi-label). The final reported score typically averages per-class scores per-sample and then averages over samples (samplewise aggregation).

Accuracy (Acc)
--------------

Pixel accuracy is the proportion of correctly classified pixels:

.. math::
   \mathrm{Acc} = \frac{\text{# correctly classified pixels}}{\text{# pixels}}

Notes:
- Multi-class: usually computed after taking argmax over classes per pixel.
- Binary/multi-label: usually computed after thresholding probabilities (e.g., 0.5).

Multi-class vs Multi-label
--------------------------

- Multi-class: each pixel belongs to exactly one class. Predictions are typically one-hot per pixel.
- Multi-label: a pixel can belong to multiple classes. Predictions can have multiple positives per pixel.

Smoothing (epsilon)
-------------------

- RankSEG exposes ``smooth`` to add a small constant to numerator/denominator for Dice/IoU to stabilize divisions.
- Typical values: ``1e-6`` to ``1e-3`` depending on scale.

References
----------

- TorchMetrics DiceScore: https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html
- JMLR paper (RankSEG): https://www.jmlr.org/papers/v24/22-0712.html
- NeurIPS paper (extended methods): https://openreview.net/forum?id=4tRMm1JJhw
