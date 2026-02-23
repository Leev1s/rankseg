# RankSeg on PaddleSeg 🚀

[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.3.0-blue?style=for-the-badge&logo=paddlepaddle)](https://www.paddlepaddle.org.cn/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?style=for-the-badge&logo=docker)](https://ghcr.io/leev1s/rankseg)

**RankSeg** integration into the **PaddleSeg** framework. 

## 🌟 Key Features

- **Metric-Aware Optimization**: Directly optimize for `dice`, `iou`, or `acc` during the inference/evaluation phase.
- **Flexible Solvers**: Choose from multiple optimization strategies: `RMA`, `BA`, `TRNA`, or `BA+TRNA`.
- **Seamless Drop-in**: Designed to work with standard PaddleSeg configuration files and models.
- **Reproducible Environment**: Fully containerized workflow using Docker and `uv`.

---

## 🛠️ Quick Start

The recommended way to run RankSeg is via our pre-built Docker container.

### 1. Pull the Image

```bash
docker pull ghcr.io/leev1s/rankseg:paddleseg
```

### 2. Download Dataset & Pretrained Models

Create a working directory on your host machine and download the assets:

```bash
mkdir -p ./data ./models ./output

# CamVid dataset
curl -L https://paddleseg.bj.bcebos.com/dataset/camvid.tar -o ./data/camvid.tar
(cd data && tar -xf camvid.tar)

# Pretrained model (PP-LiteSeg-T, STDC1)
mkdir -p models/pp_liteseg_camvid
curl -L https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/model.pdparams \
  -o models/pp_liteseg_camvid/model.pdparams
```

Your working directory should look like:

```
./
├── data/
│   └── camvid/          # extracted dataset
├── models/
│   └── pp_liteseg_camvid/
│       └── model.pdparams
└── output/              # results will be written here
```

### 3. Run Container

Mount the local directories into the container. Also mount `./output` to retrieve results on the host:

```bash
docker run --rm -it \
  -v ./data:/workspace/data \
  -v ./models:/workspace/pretrained_models \
  -v ./output:/workspace/output \
  ghcr.io/leev1s/rankseg:paddleseg
```

Container workspace layout:

```
/workspace/
├── configs/             # PaddleSeg model configs (bundled from upstream)
├── val.py               # Validation tool
├── predict.py           # Prediction tool
├── analyse.py           # Visualisation tool
├── data/                # ← mounted from host
├── pretrained_models/   # ← mounted from host
└── output/              # ← mounted from host (results written here)
```

### 4. Use Example

Once inside the container, the tools are ready at `/workspace`. The following example reproduces the CamVid benchmark result using PP-LiteSeg-T with RankSeg.

```bash
# Standard validation (baseline)
python val.py \
  --config configs/pp_liteseg/pp_liteseg_stdc1_camvid_960x720_10k.yml \
  --model_path pretrained_models/pp_liteseg_camvid/model.pdparams

# Validation with RankSeg (Dice metric)
python val.py \
  --config configs/pp_liteseg/pp_liteseg_stdc1_camvid_960x720_10k.yml \
  --model_path pretrained_models/pp_liteseg_camvid/model.pdparams \
  --use_rankseg --rankseg_metric="dice"
```

Expected results on CamVid test set:

| Model | mIoU | Dice |
|-|-|-|
| PP-LiteSeg-T | 75.92% | 81.38% |
| PP-LiteSeg-T + RankSeg | **76.13%** | **82.48%** |

#### Visualisation with `analyse.py`

`analyse.py` runs inference and saves colour-coded segmentation masks to `./output/result`.

```bash
python analyse.py \
  --config configs/pp_liteseg/pp_liteseg_stdc1_camvid_960x720_10k.yml \
  --model_path pretrained_models/pp_liteseg_camvid/model.pdparams \
  --use_rankseg --rankseg_metric="dice"
```

Results are written to `/workspace/output/result/`. Because `./output` is mounted from the host (see step 3), the visualisation files are immediately accessible on your machine after the run.

---

##  Some Supported Datasets & Models

| Dataset | Class (Config) | Task | Download |
| :--- | :--- | :--- | :---: |
| **CamVid** | `CamVid` | Road Scene | [🔗](https://paddleseg.bj.bcebos.com/dataset/camvid.tar) |
| **Cityscapes** | `Cityscapes` | Urban Scene | [🔗](https://www.cityscapes-dataset.com/) |
| **Pascal VOC** | `PascalVOC` | Object Seg | [🔗](https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar) |
| **ADE20K** | `ADE20K` | Scene Parsing | [🔗](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) |
| **EG1800** | `EG1800` | Portrait | [🔗](https://paddleseg.bj.bcebos.com/dataset/EG1800.zip) |

> *The `configs/` directory is bundled inside the container, copied from [PaddleSeg release/2.10](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs). It contains all upstream model configs ready to use.*

---

## 🚀 Advanced Usage: `val.py`

The `val.py` tool extends standard validation with RankSeg capabilities.

```bash
python rankseg/paddleseg/tools/val.py [arguments]
```

### RankSeg Options

| Argument | Type | Choices | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--use_rankseg` | `flag` | - | `False` | **Enable RankSeg optimization** |
| `--rankseg_metric` | `str` | `dice`, `iou`, `acc` | `dice` | Target metric to maximize |
| `--rankseg_solver` | `str` | `RMA`, `BA`, `TRNA` | `RMA` | Optimization algorithm |
| `--rankseg_output_mode` | `str` | `multiclass`, `multilabel` | `multiclass` | Output format |

### Standard Options

- `--config`: Path to `.yml` config file (**Required**)
- `--model_path`: Path to `.pdparams` weights (**Required**)
- `--save_dir`: Directory for results (Default: `./output/result`)

---

## 📂 Project Tree

```text
rankseg/
├── paddleseg/
│   ├── tools/
│   │   ├── val.py           # 🔍 Validation with RankSeg
│   │   ├── predict.py       # 🔍 Prediction with RankSeg
│   │   └── analyse.py       # 📊 Analysis with RankSeg
│   ├── configs/             # ⚙️ Model configs
│   ├── models/              # 🧠 Model architectures
│   ├── Dockerfile           # 🐳 Container spec
│   └── README.md            # 📄 This file
```

---

## Additional info about paddleseg repo

There are lot preconfigured pipline under `Paddleseg/config` folder. See [Paddleseg/configs](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.10/configs) for more details. Paddelseg use these configuration file to train, validate and export models. You can also modify these configuration file to fit your need.

```yml
# Paddleseg/configs/*.yml structure
_base_: '../_base_/cityscapes.yml' #import base dataset config

model:
  type: FCN
  backbone:
    type: UHRNet_W18_Small
    align_corners: False
    pretrained: https://bj.bcebos.com/paddleseg/dygraph/backbone/uhrnetw18_small_imagenet.tar.gz
  num_classes: 19
  pretrained: Null
  backbone_indices: [-1]

optimizer:
  weight_decay: 0.0005
```

---

*Powered by [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) | Maintained by [Leev1s](https://github.com/leev1s)*
