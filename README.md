# TensorRT YOLOX for Effcient Inference

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

## Building

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
./trt-yolox --onnx yolox-tiny.onnx --precision int8 --c 8 --calib MinMax --calibration_images ../calibration_images.txt
```

-Infer from a Video

```bash
./trt-yolox --onnx yolox-tiny.onnx --precision int8 --c 8 --calib MinMax --calibration_images ../calibration_images.txt [--cuda] [--rgb ../data/t4.colormap] --v VIDEO_PATH
```

-Infer from images in a directory

```bash
./trt-yolox --onnx yolox-tiny.onnx --precision int8 --c 8 --calib MinMax --calibration_images ../calibration_images.txt [--cuda] [--rgb ../data/t4.colormap] --d DIRECTORY_PATH
```

-Infer using Custom model from images in a directory

```bash
./trt-yolox --onnx ../onnx/yolox-dla-s-elan-BDD100K-640x640.onnx --c 10 --precision int8  --rgb ../data/bdd100k.colormap --calibration_images ../calibration_images.txt --calib MinMax --cuda --d ../test/ 
```

-Multi-scale Inference with batching using Custom model

```bash
./trt-yolox --onnx ../onnx/yolox-dla-s-elan-BDD100K-416x416-relu6-dynamic-batch.onnx --c 10 --rgb ../data/bdd100k.colormap  --precision int8 --calib Entropy --clip 6.0 --batch_size 6 --multi-scale --random_crop --cuda  --d ../test
```


### Cite

Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun, "YOLOX: Exceeding YOLO Series in 2021", arXiv preprint arXiv:2107.08430, 2021 [[ref](https://arxiv.org/abs/2107.08430)]


## Parameters

--precision={fp32,fp16,int8}

--c=Number of Classes

--calib={MinMax/Entropy/Legacy}

--calibration_images=path for calibration images

--rgb=colormap colormap for bbox (optional)

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
For best accuracy on DLAs, you run YOLOX with FP16 (half) precision.

### Todo

- [ ] Confirm deprecated TensorRT API
- [✓] Support Multi-batch execution

## Reference repositories

- <https://github.com/Megvii-BaseDetection/YOLOX>
- <https://github.com/wep21/yolox_onnx_modifier>
- <https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer>
- <https://github.com/NVIDIA-AI-IOT/yolo_deepstream/tree/main/yolov7_qat>

