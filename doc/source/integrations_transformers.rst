Transformers
============

This page documents the official RankSEG integration path for Hugging Face
``transformers`` semantic-segmentation outputs.

If you already run inference manually through a ``processor -> model ->
outputs`` workflow, this is the recommended entry point.

Recommended entry point
-----------------------

.. code-block:: python

   from rankseg.transformers import postprocess

The main helper is:

.. code-block:: python

   postprocess(
       outputs,
       *,
       model=None,
       output_type=None,
       target_sizes=None,
       original_sizes=None,
       reshaped_input_sizes=None,
       rankseg_kwargs=None,
       threshold=0.3,
       pad_size=None,
       apply_non_overlapping_constraints=False,
   )

Its role is intentionally narrow:

- restore probabilities from supported Hugging Face output families;
- resize them to the original image size when needed;
- apply ``RankSEG`` as the final post-processing step.

Minimal integration
-------------------

The standard Hugging Face inference structure stays the same. The only
integration change happens after ``outputs = model(**inputs)``.

.. code-block:: python

   from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
   from rankseg.transformers import postprocess
   from PIL import Image
   import requests

   processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
   model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

   image = Image.open(requests.get("https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80", stream=True).raw)
   inputs = processor(images=image, return_tensors="pt")
   outputs = model(**inputs)

   preds = postprocess(
       outputs,
       target_sizes=image.size[::-1],
       rankseg_kwargs={"metric": "dice"},
   )

For supported output families, ``postprocess(...)`` replaces the final
``argmax``-style decision step while preserving the surrounding Hugging Face
inference code.

Advanced helpers
----------------

The module also exposes:

.. code-block:: python

   from rankseg.transformers import restore_semantic_probs
   from rankseg.transformers import restore_sam_mask_probs

``restore_semantic_probs(...)`` is an advanced helper for users who want the
restored semantic probability map directly. It may be imported and used
directly, but ``postprocess(...)`` is the recommended inference entry point.

``restore_sam_mask_probs(...)`` is the SAM-family counterpart. It restores SAM
mask probabilities without requiring a Hugging Face processor object.

Supported output families
-------------------------

The current helper supports the main semantic-segmentation output families used
by ``transformers``:

- ``outputs.logits``
- ``outputs.class_queries_logits`` + ``outputs.masks_queries_logits``
- ``outputs.logits`` + ``outputs.pred_masks``
- ``outputs.semantic_seg``

When a branch requires model-specific handling, pass ``model=...`` so the
helper can follow the corresponding official post-processing behavior.

SAM output families
-------------------

``postprocess(...)`` also handles SAM-family outputs directly:

- SAM prompt masks from ``outputs.pred_masks`` + ``outputs.iou_scores``
- SAM3 semantic masks from ``outputs.semantic_seg``
- SAM3 instance masks from ``outputs.pred_logits`` + ``outputs.pred_boxes`` +
  ``outputs.pred_masks``

For SAM1 prompt masks, pass both ``original_sizes`` and
``reshaped_input_sizes`` from the processor inputs. For SAM2 prompt masks, pass
``original_sizes``. For SAM3 semantic and instance masks, pass ``target_sizes``
or ``original_sizes`` when resizing to the image size is desired.

The SAM path follows the official Transformers post-processing order up to the
final binary decision. RankSEG replaces that final thresholding step. For SAM3
instance outputs, ``threshold`` matches the official score-filtering argument.
``apply_non_overlapping_constraints`` applies only to SAM-family prompt-mask
outputs, matching the official mask post-processing API.

SAM3 image outputs can contain both instance and semantic fields. In that case,
``postprocess(...)`` follows the instance path by default, matching
``post_process_instance_segmentation(...)``. Pass ``output_type="semantic"`` to
follow ``post_process_semantic_segmentation(...)`` instead.

Current exclusions
------------------

The simplified API does not currently support:

- outputs with ``patch_offsets`` that require official patch-merge logic;
- tuple-style outputs such as ``return_dict=False`` returns;
- custom unstructured outputs from ``trust_remote_code=True`` models;
- SegGPT-style ``pred_masks`` semantic reconstruction.
- SAM video tracker state.

These cases should fail explicitly rather than silently using an incorrect
semantic restoration path.

Notebook and Colab demo
-----------------------

- Example script: `examples/transformers_rankseg.py <https://github.com/rankseg/rankseg/blob/main/examples/transformers_rankseg.py>`_
- Notebook: `notebooks/rankseg_with_transformers.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`_
- SAM family notebook: `notebooks/rankseg_with_sam_family.ipynb <https://github.com/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb>`_
- Colab: `Open the notebook in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_transformers.ipynb>`_
- SAM family Colab: `Open the SAM notebook in Colab <https://colab.research.google.com/github/rankseg/rankseg/blob/main/notebooks/rankseg_with_sam_family.ipynb>`_
