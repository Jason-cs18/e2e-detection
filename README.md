# e2e-detection
Deploy object detection models on inference engines (OpenPPL and Triton) in fewer lines of code.

Notice: To evaluate the generalization of this tool, I have tested many pretrained models from a popular object detection library ([SenseTime-MMLab mmdetection](https://github.com/open-mmlab/mmdetection)) on two inference engines ([SenseTime-MMLab OpenPPL](https://github.com/openppl-public/ppl.nn) and [Nvidia Triton](https://github.com/triton-inference-server/server)). 

> As a deep learning engineer, I highly recommend you use pre-trained models from [SenseTime-MMLab](https://github.com/open-mmlab) because the team is extremely active to develop advanced deep learning models for diverse tasks of video analytics (_i.e.,_ image classification ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmclassification.svg), object detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdetection.svg), semantic segmentation ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmsegmentation.svg), text detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmocr.svg), 3d object detection ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmdetection3d.svg), pose estimation ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmpose.svg) and video understanding based on action ![Github stars](https://img.shields.io/github/stars/open-mmlab/mmaction2.svg)).
<!-- ![applications](https://user-images.githubusercontent.com/40779233/188411410-a95bcf39-1d2a-4d41-865c-c725b3d715f0.png) -->

their difference and features
|Inference Engine|Model Conversion|Resource Management|
|:---:|:---:|:---:|
|Nvidia Triton|Hard & a small model zoo|Easy & mannual config|
|SenseTime-MMLab OpenPPL|Easy & a large model zoo|Hard & mannual coding|
|SenseTime-MMLab mmdeploy|Easy & a large model zoo|Hard & mannual coding|

GPU resource profiling

Extension to other applications (semantic segmentation, pose estimation, 3d object detection, etc)

## TODO



## Contents

### Open
## References
1. [mmdetection: OpenMMLab Detection Toolbox and Benchmark](https://github.com/open-mmlab/mmdetection)
2. [openppl: A primitive library for neural network](https://github.com/openppl-public/ppl.nn)
3. [triton: An open source inference serving software that streamlines AI inferencing](https://github.com/triton-inference-server/server)