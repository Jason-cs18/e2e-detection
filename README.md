# e2e-detection
e2e-detection is a toolkit to help deep learning engineers test their PyTorch models on popular inference engines. 

## Why we need e2e-detection?
Usually, machine learning scientists train a model via deep learning frameworks (_i.e.,_ PyTorch and TensorFlow), and engineers deploy it to an inference engine. Unfortunately, most inference engines require users to converse and configure models manually. Thus, the deployment stage needs many engineering efforts. Especially for many advanced deep learning models, engineers must try them individually to ensure that engines support them.

To bridge the gap, we develop e2e-detection to help deep learning engineers benchmark their models on popular inference engines automatically. 
## What can we do?
With e2e-detection, you only need to select a pre-trained model and the target inference engine. During benchmarking, we will converse the model and test it on the engine automatically. Finally, you will get a report for the model. The report contains (1) benchmark setup (2) testing device (3) model info (4) testing data (5) output (6) resource usage.

Notice: Because e2e-detection supports to benchmark PyTorch models of diverse applications, we choose [SenseTime-MMLab](https://github.com/open-mmlab) as the model zoo/factory. If you want to test the customized PyTorch/TensorFlow models, you need to modify the converse script (xxx.py) manually. 

<!-- have tested many pretrained models from a popular object detection library ([SenseTime-MMLab mmdetection](https://github.com/open-mmlab/mmdetection)) on two inference engines ([SenseTime-MMLab OpenPPL](https://github.com/openppl-public/ppl.nn) and [Nvidia Triton](https://github.com/triton-inference-server/server)).  -->

> As a deep learning engineer, I highly recommend you use pre-trained models from [SenseTime-MMLab](https://github.com/open-mmlab) because the team is extremely active to develop advanced deep learning models for diverse tasks in video analytics (_i.e.,_ image classification ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmclassification.svg), object detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdetection.svg), semantic segmentation ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmsegmentation.svg), text detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmocr.svg), 3d object detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdetection3d.svg), pose estimation ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmpose.svg) and video understanding based on action ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmaction2.svg)).
<!-- ![applications](https://user-images.githubusercontent.com/40779233/188411410-a95bcf39-1d2a-4d41-865c-c725b3d715f0.png) -->

## Benchmark pipeline
![pipeline](https://github.com/Jason-cs18/e2e-detection/blob/main/out/media/pipeline/pipeline.png)
## Contents
 xxx
## Inference Engines
|Inference Engine|Model Conversion|Resource Management|
|:---:|:---:|:---:|
|[Nvidia Triton](https://github.com/triton-inference-server/server) ![Github stars](https://img.shields.io/github/stars/triton-inference-server/server.svg)|Hard & a small model zoo|Easy & mannual config|
|[SenseTime-MMLab OpenPPL](https://github.com/openppl-public/ppl.nn) ![Github stars](https://img.shields.io/github/stars/openppl-public/ppl.nn.svg)|Easy & a large model zoo|Hard & mannual coding|
## Applications
- [ ] Image Classification
- [x] Object Detection
- [ ] Semantic Segmentation
- [ ] Text Detection
- [ ] 3D Object Detection
- [ ] Pose Estimation
- [ ] Video understanding
## e2e-detection vs MMDeploy
MMDeploy ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdeploy.svg) is an open-source deep learning model deployment toolset developed by SenseTime-MMLab. It helps developers to deploy Pytorch models on diverse inference engines fast (ncnn, onnx, OpenVINO, TensorRT, LibTorch). Unlike it, e2e-detection targets to provide a benchmarking result only for users. 
- If you need to develop the customized inference engine, MMDeploy is a better choice. 
- If you only need to test your models, e2e-detection is more suitable.

|Tool|easy-to-use|conversion|optimization|benchmark|customized|
|:---:|:---:|:---:|:---:|:---:|:---:|
|MMDeploy||✔️|✔️|✔️|✔️|
|e2e-detection|✔️|✔️||✔️||
## TODO
1. Nvidia Triton
2. OpenPPL
    - [ ] converse PyTorch models
    - [ ] deploy it on the engine
    - [ ] benchmark the model
    - [ ] generate the report
    - [ ] run the pipeline with a docker
## References
1. [MMDetection: OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)
2. [OpenPPL: A primitive library for neural network](https://github.com/openppl-public/ppl.nn)
3. [Triton: An open source inference serving software that streamlines AI inferencing](https://github.com/triton-inference-server/server)
4. [MMDeploy: An open-source deep learning model deployment toolset](https://github.com/open-mmlab/mmdeploy)