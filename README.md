# e2e-detection
e2e-detection is a toolkit to help deep learning engineers test their PyTorch/TensorFlow models on NVIDIA Triton inference server using different inference engines.

Let's introduce the three-stage deployment pipeline. At first, deep learning scientists train their models through deep learning frameworks (TensorFlow/Pytorch). Second, trained models will be converted to inference-optimized formats (ONNX/TensorRT/OpenPPL/NCNN/MNN). Finally, the converted models will be deployed to Nvidia Triton server. We usually call Triton inference server and other inference engines because Triton is responsible for managing resources for x models using different engines. In Triton, inference engines also call backends.

![pipeline](https://github.com/efficient-edge/e2e-detection/blob/main/media/three_stage.png)
## Why do we need e2e-detection?
- Too many engineering efforts in three-stage deployment.
- It is easy to meet dependency errors during deployment.

## What can we do?
![pipeline](https://github.com/efficient-edge/e2e-detection/blob/main/media/e2e_detection.png)
- A Dockerfile to build all testing environments automatically.
- Testing your Pytorch/TensorFlow models in fewer lines of code.
- A use case of real-world deployment.
![pipeline](https://github.com/efficient-edge/e2e-detection/blob/main/media/case_execution.png)
<!-- have tested many pre-trained models from a popular object detection library ([SenseTime-MMLab mmdetection](https://github.com/open-mmlab/mmdetection)) on two inference engines ([SenseTime-MMLab OpenPPL](https://github.com/openppl-public/ppl.nn) and [Nvidia Triton](https://github.com/triton-inference-server/server)).  -->

> As a deep learning engineer, I highly recommend you use pre-trained models from [SenseTime-MMLab](https://github.com/open-mmlab) because the team is extremely active to develop advanced deep learning models for diverse tasks in video analytics (_i.e.,_ image classification ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmclassification.svg), object detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdetection.svg), semantic segmentation ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmsegmentation.svg), text detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmocr.svg), 3d object detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdetection3d.svg), pose estimation ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmpose.svg) and video understanding based on action ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmaction2.svg)).
<!-- ![applications](https://user-images.githubusercontent.com/40779233/188411410-a95bcf39-1d2a-4d41-865c-c725b3d715f0.png) -->

<!-- ## Pipelines
We provide a basic benchmark pipeline and an advanced video analytics pipeline for users. Details can refer to the figures above.
![pipeline](https://github.com/Jason-cs18/e2e-detection/blob/main/out/media/pipeline/pipeline.png)
![pipeline](https://github.com/Jason-cs18/e2e-detection/blob/main/out/media/deployment/deployment.png) -->
## Tutorials
- [Test Pytorch Models on Nvidia Triton server using TensorRT inference engine](https://github.com/efficient-edge/e2e-detection/blob/main/triton_pytorch.md): Faster RCNN, YOLOv3, DETR, Swin-Transformer.
- [Test TensorFlow Models on Nvidia Triton server using TensorRT inference engine](https://github.com/efficient-edge/e2e-detection/blob/main/triton_tensorflow.md): EfficientDet-Dx.

## Inference Engines


|Inference Engine|Support|Stable|Target Platform|
|:---:|:---:|:---:|:---:|
|[Nvidia TensorRT](https://github.com/NVIDIA/TensorRT) ![Github stars](https://img.shields.io/github/stars/NVIDIA/TensorRT.svg)|✔️|✔️|Nvidia GPU|
|[SenseTime-MMLab OpenPPL](https://github.com/openppl-public/ppl.nn) ![Github stars](https://img.shields.io/github/stars/openppl-public/ppl.nn.svg)| | | CPU/GPU/Mobile |
|[Tencent NCNN](https://github.com/Tencent/ncnn) ![Github stars](https://img.shields.io/github/stars/Tencent/ncnn.svg)| |✔️| Mobile CPU |
|[Alibaba MNN](https://github.com/alibaba/MNN) ![Github stars](https://img.shields.io/github/stars/alibaba/MNN.svg)|TBD|✔️| Mobile CPU/GPU/NPU |
## Applications
- [ ] Image Classification
- [x] Object Detection
- [ ] Semantic Segmentation
- [ ] Text Detection
- [ ] 3D Object Detection
- [ ] Pose Estimation
- [ ] Video understanding
<!-- ## e2e-detection vs MMDeploy
MMDeploy ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdeploy.svg) is an open-source deep learning model deployment toolset developed by SenseTime-MMLab. It helps developers to deploy Pytorch models on diverse inference engines fast (NCNN, ONNX, OpenVINO, TensorRT, LibTorch). Unlike it, e2e-detection targets to provide a benchmarking result only for users. 
- If you need to develop a customized inference engine, MMDeploy is a better choice. 
- If you only need to test your models, e2e-detection is more suitable.

|Tool|easy-to-use|conversion|optimization|benchmark|customized|
|:---:|:---:|:---:|:---:|:---:|:---:|
|MMDeploy||✔️|✔️|✔️|✔️|
|e2e-detection|✔️|✔️||✔️|| -->
## TODO
1. Nvidia Triton
   1. Client applications
   - [ ] send/receive http requests
   - [ ] parse the results
   - [ ] run the pipeline with a docker
2. Pytorch
    - [x] converse PyTorch models
    - [x] deploy it on the engine
    - [ ] benchmark the model on diverse devices
    - [ ] run the pipeline with a docker
3. TensorFlow
    - [ ] converse PyTorch models
    - [ ] deploy it on the engine
    - [ ] benchmark the model on diverse devices
    - [ ] run the pipeline with a docker
## References
1. [MMDetection: OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)
2. [OpenPPL: A primitive library for neural network](https://github.com/openppl-public/ppl.nn)
3. [Triton: An open-source inference serving software that streamlines AI inferencing](https://github.com/triton-inference-server/server)
4. [MMDeploy: An open-source deep learning model deployment toolset](https://github.com/open-mmlab/mmdeploy)