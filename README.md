# TensorRT YOLOXP for Effcient Multitask Inference

## Purpose

This package detects and segments arbitary objects using TensorRT for efficient and faster inference.
Specifically, this supports multi-precision and multi-devices inference for efficient inference on embedded platform.
Moreover, we support pre-process for YOLOXP in GPU.


https://github.com/tier4/trt-yoloXP/assets/127905436/b28bd051-b853-4a28-b1e2-9f9a17e2d066



## Features

### 1. GPU-Accelerated Preprocessing

By offloading preprocessing tasks to the GPU, YOLOXP achieves higher overall throughput, significantly reducing latency and improving the efficiency of the detection pipeline.
![image](https://github.com/tier4/trt-yoloXP/assets/127905436/722861b2-622e-4189-9866-fd63a7ed27fd)


### 2. High Efficiency on Embedded GPUs

TRT-YOLOXP leverages profiling information from Xavier and Orin GPUs to utilize highly efficient operators, optimizing the model for performance. By exclusively using operators supported by the Deep Learning Accelerator (DLA), TRT-YOLOXP ensures seamless execution on DLA-enabled devices, maximizing efficiency.

![image](https://github.com/tier4/trt-yoloXP/assets/127905436/a6228a3d-e266-4a22-a3e5-6b31c1718d83)

### 3. Quantization-Aware Model

Unlike the original YOLOX, which experiences significant accuracy degradation with TensorRT Entropy calibration quantization, TRT-YOLOXP employs quantization-friendly activations using RELU6. This approach maintains high quantization accuracy on both GPUs and DLAs, preventing critical loss in precision during quantization.
![image](https://github.com/tier4/trt-yoloXP/assets/127905436/21d76493-a46d-4b11-864a-8e5e0dfc195b)

### 4. Multi-Task Model

TRT-YOLOXP supports multiple tasks within a single model, including object detection and semantic segmentation. By reusing the backbone for various tasks, it efficiently delivers multiple outputs, extending its applicability and optimizing resource use.
![image](https://github.com/tier4/trt-yoloXP/assets/127905436/8f3e14c5-680e-4d90-9ec0-fadab9c449e0)


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
For best accuracy on DLAs, we recommand to use YOLOXP-Opt with RELU6 which is quantization-friendly activations.

### Todo

- [ ] Confirm deprecated TensorRT API
- [✓] Support Multi-batch execution

## Reference repositories

- <https://github.com/Megvii-BaseDetection/YOLOX>
- <https://github.com/wep21/yolox_onnx_modifier>
- <https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer>
- <https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat>

