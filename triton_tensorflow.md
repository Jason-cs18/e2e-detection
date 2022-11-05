# Test TensorFlow models on Triton using TensorRT inference engine
As same as PyTorch models, we convert TensorFlow models to TensorRT formats and deploy them to Triton server. Because some popular models are trained with TensorFlow and their PyTorch versions are not ready to deployment, we create this tutorial to guide you test TensorFlow models on Triton step by step.
> Although we focus on efficientdet and their variants, you can modify the scripts to adapt on any TensorFlow models.
## TensorFlow model zoo
- [x] [Automl (efficientdet)](https://github.com/google/automl/tree/master/efficientdet)
## Contents
1. [Converse TensorFlow Models to TensorRT](#converse-tensorflow-models-to-tensorrt) 
2. [Inference on images with TensorRT models](#inference-on-images-with-tensorrt-models)
3. [Deploy TensorRT models on Nvidia Triton](#deploy-tensorrt-models-on-nvidia-triton)
4. [Test your video via a Triton client application](#test-your-video-via-a-triton-client-application)
### Converse TensorFlow Models to TensorRT
### Inference on images with TensorRT models
### Deploy TensorRT models on Nvidia Triton
### Test your video via a Triton client application