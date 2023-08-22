# TensorRT YOLOXP for Effcient Multitask Inference

## Purpose

This package detects arbitary objects using TensorRT for efficient and faster inference.
Specifically, this supports multi-precision and multi-devices inference for efficient inference on embedded platform.
Moreover, we support pre-process for YOLOX in GPU.

## Setup

Install basic libraries for inference on GPUs including CUDA, cuDNN, TensorRT and OpenCV.

Moreover, you need to install as bellow.
```bash
sudo apt-get install libgflags-dev
sudo apt-get install libboost-all-dev
```

## Download models

Download onnx "yoloXP-sPlus-T4-960x960-pseudo-finetune-semseg.onnx" from

https://drive.google.com/drive/folders/1OXpbS3k2rWvCawmBS1pBe7_eHD_RkQoR

## Build sources

```bash
git clone git@github.com:tier4/trt-yolox.git
cd trt-yolo
cd build/
cmake ..
make -j
```

## Start inference

-Build TensoRT engine

```bash
./trt-yolox --onnx yoloXP-sPlus-T4-960x960-pseudo-finetune-semseg.onnx  --precision int8 --calib Entropy --clip 6.0
```

-Infer from a Video

```bash
./trt-yolox --onnx yoloXP-sPlus-T4-960x960-pseudo-finetune-semseg.onnx  --precision int8 --calib Entropy --clip 6.0 --c 8 --rgb ../data/t4.colormap --names ../data/t4.names --cmap ../data/bdd100k_semseg.csv [--cuda] --v VIDEO_PATH
```

-Infer from images in a directory
```bash
./trt-yolox --onnx yoloXP-sPlus-T4-960x960-pseudo-finetune-semseg.onnx  --precision int8 --calib Entropy --clip 6.0 --c 8 --rgb ../data/t4.colormap --names ../data/t4.names --cmap ../data/bdd100k_semseg.csv [--cuda] --d DIRECTORY_PATH
```


### Cite

Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun, "YOLOX: Exceeding YOLO Series in 2021", arXiv preprint arXiv:2107.08430, 2021 [[ref](https://arxiv.org/abs/2107.08430)]


## Parameters

--precision={fp32,fp16,int8}

--c=Number of Classes

--calib={MinMax/Entropy/Legacy}

--calibration_images=path for calibration images

--rgb=colormap colormap for bbox (optional)

--cmap=colormap colormap for semantic segmentation

--cuda=cuda preprocess (optional)

--first=true partial quantization that first layer is held in fp16 (optional)

--last=true partial quantization that last layer is held in fp16 (optional)

--clip=MaxRange implicit quantization using 'setDynamicRange' (optional)

--batch_size N batched inference (Default is '1')

--multi-scale multi-scale inference with rois (optional)

--random_crop random crop inference (optional)

## Assumptions / Known limits

## Onnx model

| T4 Model | Resolutions | GFLOPS | Params[M] | Activation | Link |
| YOLOX-S | 960x960 | 59.7575 | 8.91725 | SWISH | https://drive.google.com/file/d/1qiUraIEgp45xC55ZhfvOS81e6FdGGOXN/view?usp=drive_link |
| YOLOX-M | 960x960 | 165.016 | 25.2415 | SWISH | https://drive.google.com/file/d/1zNf02RlBj6mmcUscE_8D4-VnZ1w4-gxt/view?usp=drive_link |
| YOLOX-X | 960x960 | 632.459 | 98.9011 | SWISH | https://drive.google.com/file/d/1T7Yy2xypSCtNonUmkAILnLtWWE8YzacQ/view?usp=drive_link |
| YOLOX-SPlus-Opt (V1) | 960x960 | 102.147 | 14.8016 | RELU (RELU6) | https://drive.google.com/file/d/1dp_luXzZhBr4kC4R65rl6OLNK95Dro_k/view?usp=drive_link |
| YOLOX-SPlus-Opt (V2) | 960x960 | 102.147 | 14.8016 | RELU (RELU6) | https://drive.google.com/file/d/1F5D0fVp7Wm6DUESxHei9GAX9eXlWl-FH/view?usp=drive_link |
| YOLOXP-SPlus-Opt-Semseg (V3) | 960x960 | 121.309 | 15.5052 | RELU (RELU6) | https://drive.google.com/file/d/1F5D0fVp7Wm6DUESxHei9GAX9eXlWl-FH/view?usp=drive_link |

V1: Optimized YOLOX-S for efficient inference with INT8 precision on Embedded GPUs and DLAs
V2: Better accuracy YOLOX-SPlus-Opt for cone detection using pseudo label based semi-supervised learning
V3: Multitask YOLOX-SPlus-Opt for detection and segmentation

## INT8 with Post Traninng Quantization (PTQ)

PTQ is simple approch to quantize models.
However, calibration methods is very sensitive for accuracy in quantized models.
For example, EntropyV2 that is recommand calibration in TensorRT causes crucial accuracy drop for YOLOX.
For maintaining accuracy, we recommand use MinMax calibration for YOLOX.
If accuracy in quantization is not enough, try partial quantization (--first, --last). 
(Legacy calibration is old API and has some bugs.)

## INT8 with Quanitized Aware Traning (QAT)

QAT model skips calibrations, and are quanized based on Q/DQ operators.
Current QAT  are slower than PTQ model for FP16/INT8 conversion and requires additional optimization for Q/DQ operators.
Please check bellow documents.
https://github.com/NVIDIA-AI-IOT/yolo_deepstream/blob/main/yolov7_qat/doc/Guidance_of_QAT_performance_optimization.md

## DLA

You can run your YOLOX on two DLAs on Xavier and Orin platforms.
However, DLA don't support some operation including slice in stem and  max pooling width large kernel in SPP.
This causes GPU fallback and drop inference performance.
To efficient inference on DLAs, you need to modify your YOLOX models.
Moreover, DLAs support only EntropyV2 quantization and you can get accuracy drop in INT8 on DLAs.
For best accuracy on DLAs, you run YOLOX with FP16 (half) precision.

### Todo

- [ ] Confirm deprecated TensorRT API
- [✓] Support Multi-batch execution

## Reference repositories

- <https://github.com/Megvii-BaseDetection/YOLOX>
- <https://github.com/wep21/yolox_onnx_modifier>
- <https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer>
- <https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat>

