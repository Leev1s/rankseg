.. _metrics:

📊 Segmentation Metrics
========================

This page provides comprehensive definitions of segmentation metrics used in RankSEG, including Dice, IoU, and Accuracy. Understanding these metrics is essential for choosing the right optimization strategy for your segmentation task.

🎯 Overview
------------

RankSEG optimizes segmentation predictions to maximize specific metrics during inference. Unlike traditional thresholding or argmax methods, RankSEG uses statistically-grounded algorithms to find optimal decision boundaries that directly maximize your target metric.

**Key Features:**

- **Metric-specific optimization**: Choose from Dice, IoU, and Accuracy
- **Samplewise aggregation**: Metrics are computed per sample, then averaged across the batch/dataset
- **Multi-class and multi-label support**: Flexible handling of different segmentation scenarios

📝 Notation and Definitions
----------------------------

**Basic Notation:**

For a given sample and class :math:`c`:

- :math:`\mathbf{y}_c \in \{0, 1\}^{H \times W}`: Ground-truth binary mask (1 = foreground, 0 = background)
- :math:`\mathbf{p}_c \in [0, 1]^{H \times W}`: Predicted probability map
- :math:`\hat{\mathbf{y}}_c \in \{0, 1\}^{H \times W}`: Predicted binary mask
- :math:`N`: Total number of pixels in the image (:math:`N = H \times W`)

**Confusion Matrix Elements:**

For each class :math:`c` and sample:

.. math::
   \begin{align}
   \text{TP}_c &= \sum_{i=1}^{N} \mathbb{1}[\hat{y}_{c,i} = 1, y_{c,i} = 1] \quad \text{(True Positives)} \\
   \text{FP}_c &= \sum_{i=1}^{N} \mathbb{1}[\hat{y}_{c,i} = 1, y_{c,i} = 0] \quad \text{(False Positives)} \\
   \text{FN}_c &= \sum_{i=1}^{N} \mathbb{1}[\hat{y}_{c,i} = 0, y_{c,i} = 1] \quad \text{(False Negatives)} \\
   \text{TN}_c &= \sum_{i=1}^{N} \mathbb{1}[\hat{y}_{c,i} = 0, y_{c,i} = 0] \quad \text{(True Negatives)}
   \end{align}

**Task Settings:**

- **Multi-class**: Each pixel belongs to exactly one class (mutually exclusive). Predictions are typically obtained via argmax.
- **Multi-label**: Pixels can belong to multiple classes simultaneously (overlapping). Each class is treated independently.

🎲 Dice Coefficient (F1 Score)
--------------------------------

The **Dice coefficient**, also known as the F1 score or Sørensen-Dice coefficient, measures the overlap between predicted and ground-truth segmentations. It is the harmonic mean of precision and recall.

**Mathematical Definition:**

For a single class :math:`c` and sample:

.. math::
   \text{Dice}_c = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c + \epsilon}

Equivalently, in terms of set overlap:

.. math::
   \text{Dice}_c = \frac{2 \cdot |\hat{\mathbf{y}}_c \cap \mathbf{y}_c|}{|\hat{\mathbf{y}}_c| + |\mathbf{y}_c| + \epsilon}

Where :math:`|\cdot|` denotes the cardinality (number of foreground pixels).

**Relation to Precision and Recall:**

.. math::
   \text{Dice}_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}

Where:

.. math::
   \text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}

**Properties:**

- **Range**: :math:`[0, 1]`, where 1 indicates perfect overlap and 0 indicates no overlap
- **Symmetric**: Treats false positives and false negatives equally
- **Smoothing**: :math:`\epsilon` (controlled via ``smooth`` parameter) prevents division by zero
- **Use case**: Preferred when false positives and false negatives are equally important

**RankSEG Optimization:**

RankSEG provides multiple solvers for Dice optimization:

- ``solver='BA'``: Blind Approximation (fast, works for binary segmentation)
- ``solver='TRNA'``: Truncated Refined Normal Approximation (more accurate)
- ``solver='BA+TRNA'``: Automatically selects between BA and TRNA
- ``solver='RMA'``: Reciprocal Moment Approximation (supports multi-class)

🔲 IoU (Intersection over Union)
----------------------------------

The **IoU**, also known as the Jaccard Index, measures the ratio of intersection to union between predicted and ground-truth segmentations.

**Mathematical Definition:**

For a single class :math:`c` and sample:

.. math::
   \text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c + \epsilon}

Equivalently:

.. math::
   \text{IoU}_c = \frac{|\hat{\mathbf{y}}_c \cap \mathbf{y}_c|}{|\hat{\mathbf{y}}_c \cup \mathbf{y}_c| + \epsilon}

**Relation to Dice:**

IoU and Dice are monotonically related:

.. math::
   \text{IoU}_c = \frac{\text{Dice}_c}{2 - \text{Dice}_c}, \quad \text{Dice}_c = \frac{2 \cdot \text{IoU}_c}{1 + \text{IoU}_c}

**Properties:**

- **Range**: :math:`[0, 1]`, where 1 indicates perfect overlap
- **More strict than Dice**: IoU penalizes mismatches more heavily than Dice
- **Smoothing**: :math:`\epsilon` prevents division by zero
- **Use case**: Common in object detection and instance segmentation (e.g., COCO dataset)

**RankSEG Optimization:**

- ``solver='RMA'``: Reciprocal Moment Approximation (currently the only supported solver for IoU)

✅ Pixel Accuracy
------------------

**Pixel Accuracy** measures the proportion of correctly classified pixels across all classes.

**Mathematical Definition:**

For a single sample:

.. math::
   \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]}{N}

For multi-class segmentation with :math:`C` classes:

.. math::
   \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\arg\max_c p_{c,i} = y_i]

Where :math:`y_i \in \{0, 1, \ldots, C-1\}` is the ground-truth class label for pixel :math:`i`.

**Properties:**

- **Range**: :math:`[0, 1]`, where 1 indicates perfect classification
- **Class imbalance sensitivity**: Can be misleading when classes are imbalanced (e.g., dominated by background)
- **Simple interpretation**: Easy to understand and communicate
- **Use case**: Suitable when all classes are equally important and balanced

**RankSEG Optimization:**

- ``solver='argmax'``: Standard argmax over class probabilities (multi-class)
- ``solver='TR'``: Truncation at 0.5 threshold (multi-label)

📈 Averaging Across Classes and Samples
-------------------------------------

RankSEG follows a **samplewise aggregation** strategy, which matches the convention used in TorchMetrics (``aggregation_level='samplewise'``).

**Aggregation Steps:**

1. **Per-class, per-sample**: Compute metric for each class :math:`c` in each sample :math:`s`
2. **Per-sample**: Average across classes (macro average):

   .. math::
      \text{Metric}_s = \frac{1}{C} \sum_{c=1}^{C} \text{Metric}_{s,c}

3. **Final score**: Average across samples:

   .. math::
      \text{Metric}_{\text{final}} = \frac{1}{S} \sum_{s=1}^{S} \text{Metric}_s

**Alternative Aggregation Methods:**

- **Micro-averaging**: Pool all predictions and compute metric globally (treats all pixels equally)
- **Weighted macro-averaging**: Weight classes by their frequency or importance
- **Class-specific**: Report metrics separately for each class

🏷️ Multi-class vs Multi-label Segmentation
--------------------------------------------

Understanding the difference between multi-class and multi-label segmentation is crucial for choosing the right RankSEG configuration.

**Multi-class Segmentation:**

- Each pixel belongs to **exactly one** class
- Predictions are mutually exclusive (one-hot encoded per pixel)
- Example: Semantic segmentation (person, car, road, sky, etc.)
- RankSEG setting: ``output_mode='multiclass'``

**Mathematical constraint:**

.. math::
   \sum_{c=1}^{C} \hat{y}_{c,i} = 1, \quad \forall i \in \{1, \ldots, N\}

**Multi-label Segmentation:**

- Each pixel can belong to **multiple** classes simultaneously
- Predictions can overlap (independent binary masks per class)
- Example: Medical imaging (tumor + edema), multi-organ segmentation
- RankSEG setting: ``output_mode='multilabel'``

**Mathematical constraint:**

.. math::
   \hat{y}_{c,i} \in \{0, 1\}, \quad \text{no constraint on } \sum_{c=1}^{C} \hat{y}_{c,i}

⚙️ Smoothing Parameter
-----------------------

The **smoothing parameter** :math:`\epsilon` (``smooth`` in RankSEG) is added to prevent division by zero and improve numerical stability.

**Effect on Metrics:**

For Dice:

.. math::
   \text{Dice}_c = \frac{2 \cdot \text{TP}_c + \epsilon}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c + \epsilon}

For IoU:

.. math::
   \text{IoU}_c = \frac{\text{TP}_c + \epsilon}{\text{TP}_c + \text{FP}_c + \text{FN}_c + \epsilon}

**Choosing :math:`\epsilon`:**

- **Small values** (``1e-6`` to ``1e-8``): Minimal impact, primarily for numerical stability
- **Larger values** (``1e-3`` to ``1``): Can help with class imbalance or small objects
- **Default**: ``smooth=0.0`` in RankSEG (no smoothing)

**Trade-offs:**

- Larger :math:`\epsilon` can bias metrics toward higher values
- May help gradient flow during training but less relevant for inference-only RankSEG

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

🎯 Choosing the Right Metric
------------------------------

**For Medical Imaging:**

- Use **Dice** when false positives and false negatives are equally important (e.g., organ segmentation)
- Use **IoU** when you need stricter overlap requirements
- Consider **class-specific metrics** for multi-organ scenarios

**For Natural Images:**

- Use **IoU** for object detection and instance segmentation (COCO standard)
- Use **Dice** for semantic segmentation when overlap quality matters
- Use **Accuracy** only if classes are well-balanced

**RankSEG Solver Selection:**

.. code-block:: python

   # For Dice optimization
   rankseg = RankSEG(metric='dice', solver='RMA')  # Multi-class support
   rankseg = RankSEG(metric='dice', solver='BA')   # Fast, binary only
   
   # For IoU optimization
   rankseg = RankSEG(metric='IoU', solver='RMA')   # Only option
   
   # For Accuracy optimization
   rankseg = RankSEG(metric='Acc', solver='argmax')  # Multi-class
   rankseg = RankSEG(metric='Acc', solver='TR')      # Multi-label

💡 Practical Examples
----------------------

**Example 1: Binary Segmentation (Tumor Detection)**

.. code-block:: python

   import torch
   from rankseg import RankSEG
   
   # Probability map from your model
   probs = torch.rand(4, 1, 256, 256)  # (batch, 1 class, H, W)
   
   # Optimize for Dice (common in medical imaging)
   rankseg = RankSEG(metric='dice', solver='BA', smooth=1e-6)
   preds = rankseg.predict(probs)

**Example 2: Multi-class Semantic Segmentation**

.. code-block:: python

   # Probability maps for 21 classes (e.g., PASCAL VOC)
   probs = torch.rand(4, 21, 512, 512)  # (batch, classes, H, W)
   
   # Optimize for IoU (COCO/VOC standard)
   rankseg = RankSEG(metric='IoU', solver='RMA', output_mode='multiclass')
   preds = rankseg.predict(probs)

**Example 3: Multi-label Segmentation**

.. code-block:: python

   # Multiple overlapping classes (e.g., medical multi-organ)
   probs = torch.rand(2, 5, 128, 128)  # (batch, organs, H, W)
   
   # Each organ treated independently
   rankseg = RankSEG(metric='dice', solver='RMA', output_mode='multilabel')
   preds = rankseg.predict(probs)

⚠️ Common Pitfalls and Tips
-----------------------------

**Pitfall 1: Using Accuracy with Imbalanced Classes**

❌ **Problem**: Background dominates, accuracy looks high but foreground is poorly segmented

✅ **Solution**: Use Dice or IoU instead, or compute class-weighted accuracy

**Pitfall 2: Confusing Multi-class and Multi-label**

❌ **Problem**: Using ``output_mode='multiclass'`` when classes can overlap

✅ **Solution**: Use ``output_mode='multilabel'`` for overlapping classes

**Pitfall 3: Ignoring Smoothing for Small Objects**

❌ **Problem**: Division by zero or unstable metrics for tiny objects

✅ **Solution**: Set ``smooth=1e-6`` or higher for numerical stability

**Pitfall 4: Wrong Aggregation Level**

❌ **Problem**: Comparing metrics computed with different aggregation strategies

✅ **Solution**: Always use samplewise aggregation for consistency with RankSEG

**Tip 1: Validate with Multiple Metrics**

Even if you optimize for one metric, evaluate with multiple metrics to get a complete picture:

.. code-block:: python

   from torchmetrics.segmentation import DiceScore
   
   # Optimize with RankSEG
   rankseg = RankSEG(metric='dice', solver='RMA')
   preds = rankseg.predict(probs)
   
   # Evaluate with multiple metrics
   dice_metric = DiceScore()
   dice_score = dice_metric(preds, targets)

**Tip 2: Experiment with Solvers**

Different solvers may work better for your specific data:

.. code-block:: python

   # Try different solvers and compare
   for solver in ['BA', 'TRNA', 'BA+TRNA', 'RMA']:
       rankseg = RankSEG(metric='dice', solver=solver)
       preds = rankseg.predict(probs)
       # Evaluate and compare

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
