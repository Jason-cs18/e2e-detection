# Test Pytorch Models on Nvidia Triton server using TensorRT inference engine
In this page, we will show you a step-by-step tutorial to test PyTorch models on Nvidia Triton with Nvidia TensorRT engine (CPU/GPU). 
## Step-by-step Tutorial
1. [Converse Pytorch Models to TensorRT](#converse-pytorch-models-to-tensorrt) 
2. [Test TensorRT models](#test-tensorrt-models)
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
You can check your image via the command
```
docker image ls
```
![tensorrt_gpu](https://github.com/efficient-edge/e2e-detection/blob/main/media/docker_gpu.png)

2. Converse PyTorch models to TensorRT (~ 3 minutes)
```
# faster_rcnn, yolov3, detr, yolox, swin_transformer, efficientdet_dx
bash tensorrt_convert.sh yolov3
```
When you complete this step, you will get a new end2end.engine in ./temp/tensorrt_models/yolov3/ and the speed profiling result with batch-size=1.

![tensorrt_gpu](https://github.com/efficient-edge/e2e-detection/blob/main/media/tensorrt_yolov3_model.png)
![tensorrt_gpu](https://github.com/efficient-edge/e2e-detection/blob/main/media/tensorrt_yolov3_speed.png)
### Test TensorRT models
### Deploy TensorRT models on Nvidia Triton
### Test your video via a Triton client application