Getting Started
===============

RankSEG provides plug-and-play modules to improve segmentation results during inference by using ranking-based methods that are statistically consistent with popular segmentation metrics.

Installation
------------

Install RankSEG using pip:

.. code-block:: bash

   pip install rankseg

.. raw:: html

  <style>
    /* Make tabs more compact */
    .method-selection .sd-tab-set > input + label {
      padding: 0.2rem 0.7rem;
      font-size: 0.9rem;
      margin: 0 0.35rem 0.35rem 0;  
      border: 1px solid #ddd;
      border-radius: 9999px; /* pill shape */
      line-height: 1.2;
    }

    /* Hover and active states for tabs */
    .method-selection .sd-tab-set > input:checked + label {
      border-color: #964dd1ff;
      color: #964dd1ff;
    }
    
    /* Make task tabs split space equally */
    .method-selection .tabs-task > input + label {
      flex: 1;
      text-align: center;
    }
    
    /* Make metric tabs split space equally */
    .method-selection .tabs-metric > input + label {
      flex: 1;
      text-align: center;
    }
    
    @media screen and (min-width: 960px) {

      .method-selection .sd-tab-set.tabs-task::before {
        content: "Task:";
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.2rem 3.6rem 0.3rem 0.6rem;
        display: block;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .method-selection .sd-tab-set.tabs-metric::before {
        content: "Metric:";
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.2rem 2.6rem 0.3rem 0.6rem;
        display: block;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
    }
    
    /* Nested tabs styling */
    .method-selection .sd-tab-set .sd-tab-set {
      margin-top: 0.5rem;
      margin-bottom: 0.5rem;
    }
    
  </style>

✨ Quick Start
--------------

RankSEG works as a post-processing step after your segmentation model produces probability maps. Here's a minimal example:

.. code-block:: python

   import torch
   from rankseg import RankSEG

   # Your model's probability output (batch_size, num_classes, height, width)
   probs = model(images)  # e.g., shape: (4, 21, 256, 256)
   
   # Create RankSEG predictor
   rankseg = RankSEG(metric='dice')
   
   # Get optimized predictions
   preds = rankseg.predict(probs)

**Key Benefits:**

- ✅ **No retraining required** - Works with any pre-trained segmentation model
- ✅ **Metric-aware** - Directly optimizes for your target metric (Dice, IoU, or AP)
- ✅ **Statistically consistent** - Theoretically guaranteed to improve performance
- ✅ **Easy integration** - Just 2 lines of code to add to your inference pipeline


Usage Examples by Task and Metric
----------------------------------

Choose your segmentation task and target metric below to see the corresponding code example.

.. container:: method-selection

  .. tab-set::
    :class: tabs-task

    .. tab-item:: Binary segmentation
      :class-label: task-binary

      .. tab-set::
        :class: tabs-metric

        .. tab-item:: AP
          :class-label: metric-ap
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes=1, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the AP metric
              rankseg = RankSEG(metric='AP')
              pred = rankseg.predict(probs)
              
        .. tab-item:: Dice
          :class-label: metric-dice
          
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes=1, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the Dice metric
              rankseg = RankSEG(metric='dice')
              pred = rankseg.predict(probs)
              
        .. tab-item:: IoU
          :class-label: metric-iou
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes=1, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the IoU metric
              rankseg = RankSEG(metric='IoU')
              pred = rankseg.predict(probs)

    .. tab-item:: Multiclass segmentation
      :class-label: task-multiclass

      .. tab-set::
        :class: tabs-metric

        .. tab-item:: AP
          :class-label: metric-ap
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the AP metric
              rankseg = RankSEG(metric='AP')
              pred = rankseg.predict(probs)

        .. tab-item:: Dice
          :class-label: metric-dice
          
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the Dice metric
              rankseg = RankSEG(metric='dice')
              pred = rankseg.predict(probs)

        .. tab-item:: IoU
          :class-label: metric-iou
                    
          .. code-block:: python
          
              import torch
              from rankseg import RankSEG
              ## `probs` (batch_size, num_classes, *image_shape) is the input probability tensor
              
              # Make segmentation prediction target the IoU metric
              rankseg = RankSEG(metric='IoU')
              pred = rankseg.predict(probs)

Advanced Configuration
----------------------

Solver Selection
~~~~~~~~~~~~~~~~

RankSEG provides multiple solver algorithms with different speed-accuracy trade-offs:

.. list-table::
   :widths: 15 20 65
   :header-rows: 1

   * - Solver
     - Speed
     - Description
   * - ``'RMA'``
     - Fastest
     - **Recommended for most cases.** Reciprocal Moment Approximation. Works for both binary and multiclass segmentation, and supports all metrics (Dice, IoU, AP). Good balance of speed and accuracy.
   * - ``'BA'``
     - Fast
     - Blind Approximation. Best for Dice metric when speed is critical. Requires ``eps`` parameter.
   * - ``'TRNA'``
     - Slow
     - Truncated Refined Normal Approximation. More accurate than BA for complex cases. Requires ``eps`` parameter.
   * - ``'BA+TRNA'``
     - Fast (adaptive)
     - Automatically selects between BA and TRNA based on data characteristics using Cohen's d.

Example with solver parameters:

.. code-block:: python

   from rankseg import RankSEG
   
   # RMA solver (default, works for all metrics)
   rankseg = RankSEG(metric='dice', solver='RMA')
   
   # BA solver with custom epsilon
   rankseg = RankSEG(metric='dice', solver='BA', eps=1e-4)
   
   # Automatic solver selection
   rankseg = RankSEG(metric='dice', solver='BA+TRNA', eps=1e-4)

Multi-label vs Multi-class Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RankSEG supports both multi-class (mutually exclusive) and multi-label (overlapping) segmentation:

.. code-block:: python

   # Multi-class: each pixel belongs to exactly one class
   rankseg = RankSEG(metric='dice', solver='RMA', return_binary_masks=False)
   preds = rankseg.predict(probs)  # Shape: (batch, height, width)
   
   # Multi-label: pixels can belong to multiple classes
   rankseg = RankSEG(metric='dice', solver='RMA', return_binary_masks=True)
   preds = rankseg.predict(probs)  # Shape: (batch, num_classes, height, width)

Pruning for Efficiency
~~~~~~~~~~~~~~~~~~~~~~

Use the ``pruning_prob`` parameter to skip classes with low probabilities:

.. code-block:: python

   # Skip classes where max probability < 0.5
   rankseg = RankSEG(metric='dice', solver='RMA', pruning_prob=0.5)
   preds = rankseg.predict(probs)

This can significantly speed up inference when many classes are unlikely.

Complete Example
----------------

Here's a complete example integrating RankSEG into a segmentation pipeline:

.. code-block:: python

   import torch
   import torch.nn as nn
   from torchvision import transforms
   from rankseg import RankSEG
   
   # Load your pre-trained segmentation model
   model = torch.load('unet_model.pth')
   model.eval()
   
   # Initialize RankSEG
   rankseg = RankSEG(
       metric='dice',           # Target metric: 'dice', 'IoU', or 'AP'
       solver='RMA',             # Solver algorithm
       pruning_prob=0.5,         # Skip low-probability classes
       return_binary_masks=False # Multi-class segmentation
   )
   
   # Inference function
   def segment_image(image):
       with torch.no_grad():
           # Get probability maps from model
           logits = model(image)
           probs = torch.softmax(logits, dim=1)  # or torch.sigmoid for binary
           
           # Apply RankSEG optimization
           preds = rankseg.predict(probs)
           
       return preds
   
   # Process a batch of images
   images = torch.randn(4, 3, 256, 256)  # Example batch
   predictions = segment_image(images)
   print(f"Predictions shape: {predictions.shape}")

GPU Acceleration
----------------

RankSEG automatically uses GPU if your input tensors are on GPU:

.. code-block:: python

   import torch
   from rankseg import RankSEG
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   # Move model and data to GPU
   model = model.to(device)
   images = images.to(device)
   
   # Get predictions (RankSEG will use GPU automatically)
   probs = model(images)
   rankseg = RankSEG(metric='dice', solver='RMA')
   preds = rankseg.predict(probs)  # Computed on GPU

Best Practices
--------------

1. **Choose the right metric**: Use the same metric that you'll evaluate your model with (Dice, IoU, or AP).

2. **Start with RMA solver**: It works well for all metrics and provides good speed-accuracy balance.

3. **Use pruning for speed**: Set ``pruning_prob=0.5`` to skip unlikely classes in multi-class segmentation.

4. **Validate improvements**: Always compare RankSEG predictions against baseline (argmax/threshold) on your validation set.

5. **GPU acceleration**: For large images or batches, ensure your tensors are on GPU for faster processing.

Common Issues
-------------

**Q: My predictions look the same as argmax/threshold?**

A: This can happen if your model's probabilities are already very confident. RankSEG provides the most benefit when probabilities are uncertain.

**Q: RankSEG is slow on my large images?**

A: Try using ``pruning_prob=0.5`` to skip unlikely classes, or use the ``'BA'`` solver for faster computation.

**Q: Should I use multi-class or multi-label mode?**

A: Use multi-class (``return_binary_masks=False``) when classes are mutually exclusive (e.g., semantic segmentation). Use multi-label (``return_binary_masks=True``) when objects can overlap (e.g., instance segmentation, medical imaging).

Next Steps
----------

- Check out the :doc:`API Reference </autoapi/rankseg/index>` for detailed parameter descriptions
- See :doc:`Citation </citation>` for how to cite RankSEG in your research
- Report issues or contribute on `GitHub <https://github.com/statmlben/rankseg>`_
              