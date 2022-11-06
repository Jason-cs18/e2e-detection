# Test Pytorch Models on Nvidia Triton server using TensorRT inference engine
In this page, we will show you a step-by-step tutorial to test PyTorch models on Nvidia Triton with Nvidia TensorRT engine (CPU/GPU).
## Pytorch model zoo
- [x] [mmdetection](https://github.com/open-mmlab/mmdetection)
- [ ] [mmyolo](https://github.com/open-mmlab/mmyolo)
## Contents
1. [Converse Pytorch Models to TensorRT](#converse-pytorch-models-to-tensorrt) 
2. [Inference on images with TensorRT models](#inference-on-images-with-tensorrt-models)
3. [Deploy TensorRT models on Nvidia Triton](#deploy-tensorrt-models-on-nvidia-triton)
4. [Test your video via a Triton client application](#test-your-video-via-a-triton-client-application)
### Converse Pytorch Models to TensorRT
1. Build mmdeploy Dockerfile w./wo. GPU (15 ~ 20 minutes)
```
# build a docker image for mmdeploy and mmdetection
docker build -t mmdeploy-gpu --network host mmdeploy/gpu/
# (optional) build a docker image for mmdeploy and mmdetection (in cpu)
docker build -t mmdeploy-cpu --network host mmdeploy/cpu
# (optional) run an interactive terminal in mmdeploy
docker run -it --network host --gpus all -v /home/jason/Toolbox/e2e-detection/temp:/root/workspace/temp mmdeploy-cpu
```
- You can check your image via the command
```
docker image ls
```
![tensorrt_gpu](https://github.com/efficient-edge/e2e-detection/blob/main/media/docker_gpu.png)

2. Converse PyTorch models to TensorRT (~ 3 minutes)
```
# faster_rcnn (2015), yolov3 (2018), detr (2020), swin_transformer (2021), efficientdet_dx (2020)
bash tensorrt_convert.sh yolov3
```
- When you complete this step, you will get a new end2end.engine in ./temp/tensorrt_models/yolov3/ and the speed profiling result with batch-size=1.

![tensorrt_gpu](https://github.com/efficient-edge/e2e-detection/blob/main/media/tensorrt_yolov3_model.png)
![tensorrt_gpu](https://github.com/efficient-edge/e2e-detection/blob/main/media/tensorrt_yolov3_speed.png)

> Notice: Because official efficientdet is implemented by tensorflow and non-official implementations have multiple errors in the deployment stage, we choose efficientdet-tensorflow as our source model. More details can be found in [tensorflow-triton-tutorial](https://github.com/efficient-edge/e2e-detection/blob/main/triton_tensorflow.md)
### Inference on images with TensorRT models
For Faster-RCNN, YOLOv3, DETR, Swin-Transformer, please type
```
docker run -it --gpus all --network host -v /home/jason/Toolbox/e2e-detection/temp:/root/workspace/temp mmdeploy-gpu python -W ignore temp/scripts/tensorrt_inference.py faster_rcnn/yolov3/detr/swin
```
After the execution, you will get a sample video with the detected bboxes in /temp/outputs/xxx-movie.mp4.
![tensorrt_gpu](https://github.com/efficient-edge/e2e-detection/blob/main/media/pytorch_video.png)
### Deploy TensorRT models on Nvidia Triton
### Test your video via a Triton client application