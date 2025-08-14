Getting Started
===============

RankSEG provides plug-and-play modules to improve segmentation results during inference by using ranking-based methods that are statistically consistent with popular segmentation metrics.

.. raw:: html

  <style>
    /* Improved layout for tabs */
    .method-selection .sd-tab-set {
      margin-bottom: 1.5rem;
    }
    
    @media screen and (min-width: 960px) {
      .method-selection .sd-tab-set {
        --tab-caption-width: 30%;
      }

      .method-selection .sd-tab-set.tabs-task::before {
        content: "Task:";
        font-weight: bold;
        padding: 0.8rem 0.5rem 0.5rem 0.5rem;
        display: block;
        color: #555;
      }

      .method-selection .sd-tab-set.tabs-metric::before {
        content: "Metric:";
        font-weight: bold;
        padding: 0.8rem 0.5rem 0.5rem 0.5rem;
        display: block;
        color: #555;
      }
    }
    
  </style>

Suppose you have already trained a segmentation model ``model`` (for example, a U-Net trained with cross-entropy loss), and you want to use it to predict the segmentation results for a new images ``inputs``.

.. div:: method-selection

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
              from rankseg import rankseg
              ## `inputs` (batch_size, num_channels, height, width) is the input images
              ## `model` is the trained segmentation model producing the output logits
              
              # Load your model's predicted logits
              logits = model(inputs)  # shape: (batch_size, num_classes=1, height, width)
              probs = torch.sigmoid(logits)  # For binary segmentation: probs of segmentation
              
              # Make segmentation prediction target the AP metric
              rankseg = rankseg(metric='AP')
              pred = rankseg.predict(probs)
              
        .. tab-item:: Dice
          :class-label: metric-dice
          
          .. code-block:: python
          
              import torch
              from rankseg import rankseg
              ## `inputs` (batch_size, num_channels, height, width) is the input images
              ## `model` is the trained segmentation model producing the output logits
              
              # Load your model's predicted logits
              logits = model(inputs)  # shape: (batch_size, num_classes=1, height, width)
              probs = torch.sigmoid(logits)  # For binary segmentation: probs of segmentation
              
              # Make segmentation prediction target the Dice metric
              rankseg = rankseg(metric='Dice')
              pred = rankseg.predict(probs)
              
        .. tab-item:: IoU
          :class-label: metric-iou
                    
          .. code-block:: python
          
              import torch
              from rankseg import rankseg
              ## `inputs` (batch_size, num_channels, height, width) is the input images
              ## `model` is the trained segmentation model producing the output logits
              
              # Load your model's predicted logits
              logits = model(inputs)  # shape: (batch_size, num_classes=1, height, width)
              probs = torch.sigmoid(logits)  # For binary segmentation: probs of segmentation
              
              # Make segmentation prediction target the IoU metric
              rankseg = rankseg(metric='IoU')
              pred = rankseg.predict(probs)

    .. tab-item:: Multiclass segmentation
      :class-label: task-multiclass

      .. tab-set::
        :class: tabs-metric

        .. tab-item:: AP
          :class-label: metric-ap
                    
          .. code-block:: python
          
              import torch
              from rankseg import rankseg
              ## `inputs` (batch_size, num_channels, height, width) is the input images
              ## `model` is the trained segmentation model producing the output logits
              
              # Load your model's predicted logits
              logits = model(inputs)  # shape: (batch_size, num_classes, height, width)
              probs = torch.softmax(logits, dim=1)  # For multi-class segmentation
              
              # Make segmentation prediction target the AP metric
              rankseg = rankseg(metric='AP')
              pred = rankseg.predict(probs)

        .. tab-item:: Dice
          :class-label: metric-dice
          
          .. code-block:: python
          
              import torch
              from rankseg import rankseg
              ## `inputs` (batch_size, num_channels, height, width) is the input images
              ## `model` is the trained segmentation model producing the output logits
              
              # Load your model's predicted logits
              logits = model(inputs)  # shape: (batch_size, num_classes, height, width)
              probs = torch.softmax(logits, dim=1)  # For multi-class segmentation
              
              # Make segmentation prediction target the Dice metric
              rankseg = rankseg(metric='Dice')
              pred = rankseg.predict(probs)

        .. tab-item:: IoU
          :class-label: metric-iou
                    
          .. code-block:: python
          
              import torch
              from rankseg import rankseg
              ## `inputs` (batch_size, num_channels, height, width) is the input images
              ## `model` is the trained segmentation model producing the output logits
              
              # Load your model's predicted logits
              logits = model(inputs)  # shape: (batch_size, num_classes, height, width)
              probs = torch.softmax(logits, dim=1)  # For multi-class segmentation
              
              # Make segmentation prediction target the IoU metric
              rankseg = rankseg(metric='IoU')
              pred = rankseg.predict(probs)
              