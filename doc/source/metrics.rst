.. _metrics:

📊 Segmentation Metrics
========================

This page provides comprehensive definitions of segmentation metrics used in RankSEG, including Dice, IoU, and Accuracy. Understanding these metrics is essential for choosing the right optimization strategy for your segmentation task.

📝 Notation and Definitions
----------------------------

**Basic Notation:**

For a given sample :math:`i` and class :math:`c`:

- :math:`\mathbf{y}_{ic} \in \{0, 1\}^{H \times W}`: Ground-truth binary mask (1 = foreground, 0 = background)
- :math:`\mathbf{p}_{ic} \in [0, 1]^{H \times W}`: Predicted probability map
- :math:`\hat{\mathbf{y}}_{ic} \in \{0, 1\}^{H \times W}`: Predicted binary mask

**Confusion Matrix Elements:**

For each class :math:`c` and sample :math:`i`:

.. math::
   \begin{align}
   \text{TP}_{i,c} &= \sum_{w,h} \mathbb{1}[\hat{y}_{i,c,w,h} = 1, y_{i,c,w,h} = 1] \quad \text{(True Positives)} \\
   \text{FP}_{i,c} &= \sum_{w,h} \mathbb{1}[\hat{y}_{i,c,w,h} = 1, y_{i,c,w,h} = 0] \quad \text{(False Positives)} \\
   \text{FN}_{i,c} &= \sum_{w,h} \mathbb{1}[\hat{y}_{i,c,w,h} = 0, y_{i,c,w,h} = 1] \quad \text{(False Negatives)} \\
   \text{TN}_{i,c} &= \sum_{w,h} \mathbb{1}[\hat{y}_{i,c,w,h} = 0, y_{i,c,w,h} = 0] \quad \text{(True Negatives)}
   \end{align}

🎲 Dice Coefficient (F1 Score)
--------------------------------

The **Dice coefficient**, also known as the F1 score or Sørensen-Dice coefficient, measures the overlap between predicted and ground-truth segmentations. It is the harmonic mean of precision and recall.

**Mathematical Definition:**

For a single class :math:`c` and sample :math:`i`:

.. math::
   \text{Dice}_{ic} = \frac{2 \cdot \text{TP}_{ic} + \gamma}{2 \cdot \text{TP}_{ic} + \text{FP}_{ic} + \text{FN}_{ic} + \gamma}

Equivalently, in terms of set overlap:

.. math::
   \text{Dice}_{ic} = \frac{2 \cdot |\hat{\mathbf{y}}_{ic} \cap \mathbf{y}_{ic}| + \gamma}{|\hat{\mathbf{y}}_{ic}| + |\mathbf{y}_{ic}| + \gamma}

Where :math:`|\cdot|` denotes the cardinality (number of foreground pixels).

Then, the averaged Dice score for class :math:`c` across all samples :math:`i` is computed as:

.. math::
   \text{Dice}_c = \frac{1}{n} \sum_{i=1}^{n} \text{Dice}_{ic}.

**Properties:**

- **Range**: :math:`[0, 1]`, where 1 indicates perfect overlap and 0 indicates no overlap
- **Symmetric**: Treats false positives and false negatives equally
- **Smoothing**: :math:`\gamma` (controlled via ``smooth`` parameter) prevents division by zero
- **Use case**: Preferred when false positives and false negatives are equally important

🔲 IoU (Intersection over Union)
----------------------------------

The **IoU**, also known as the Jaccard Index, measures the ratio of intersection to union between predicted and ground-truth segmentations.

**Mathematical Definition:**

For a single class :math:`c` and sample :math:`i`:

.. math::
   \text{IoU}_{ic} = \frac{\text{TP}_{ic} + \gamma}{\text{TP}_{ic} + \text{FP}_{ic} + \text{FN}_{ic} + \gamma}

Equivalently:

.. math::
   \text{IoU}_{ic} = \frac{|\hat{\mathbf{y}}_{ic} \cap \mathbf{y}_{ic}| + \gamma}{|\hat{\mathbf{y}}_{ic} \cup \mathbf{y}_{ic}| + \gamma}

Then, the averaged IoU score for class :math:`c` across all samples :math:`i` is computed as:

.. math::
   \text{IoU}_c = \frac{1}{n} \sum_{i=1}^{n} \text{IoU}_{ic}.

**Properties:**

- **Range**: :math:`[0, 1]`, where 1 indicates perfect overlap
- **More strict than Dice**: IoU penalizes mismatches more heavily than Dice
- **Smoothing**: :math:`\gamma` (controlled via ``smooth`` parameter) prevents division by zero
- **Use case**: Common in object detection and instance segmentation (e.g., COCO dataset)


📈 Averaging Across Classes and Samples
---------------------------------------

RankSEG follows a **samplewise aggregation** strategy, which matches the convention used in TorchMetrics (``aggregation_level='samplewise'``).



📋 Quick Reference: Metric Comparison
---------------------------------------

.. list-table:: Segmentation Metrics Comparison
   :header-rows: 1
   :widths: 15 15 20 25 25

   * - Metric
     - Range
     - Best For
     - Advantages
     - Disadvantages
   * - **Dice**
     - [0, 1]
     - Balanced FP/FN importance
     - Symmetric, widely used in medical imaging
     - Less strict than IoU
   * - **IoU**
     - [0, 1]
     - Object detection, COCO
     - More strict overlap measure
     - Penalizes errors more heavily
   * - **Accuracy**
     - [0, 1]
     - Balanced classes
     - Simple, intuitive
     - Misleading with class imbalance

📚 References and Further Reading
-----------------------------------

**Academic Papers:**

- **RankSEG (JMLR 2023)**: Dai, B., et al. "RankSEG: A Consistent Ranking-based Framework for Segmentation." *Journal of Machine Learning Research*, 24(71):1-58, 2023. https://www.jmlr.org/papers/v24/22-0712.html

- **Extended Methods (NeurIPS 2024)**: Wang, Z., et al. "Improved Ranking-based Segmentation Methods." *NeurIPS*, 2024. https://openreview.net/forum?id=4tRMm1JJhw

**Documentation and Tools:**

- **TorchMetrics Dice Score**: https://lightning.ai/docs/torchmetrics/stable/segmentation/dice.html
- **TorchMetrics IoU**: https://lightning.ai/docs/torchmetrics/stable/segmentation/jaccard_index.html
- **COCO Evaluation**: https://cocodataset.org/#detection-eval

**Tutorials:**

- Dice vs IoU: When to use which? https://stats.stackexchange.com/questions/273537/
- Segmentation Metrics Explained: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
