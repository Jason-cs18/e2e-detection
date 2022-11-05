MODEL=$1
DOCKER_TEMP=/home/jason/Toolbox/e2e-detection/temp
YOLOV3_PATH=/home/jason/Toolbox/e2e-detection/temp/tensorrt_models/yolov3
FASTERCNN_PATH=/home/jason/Toolbox/e2e-detection/temp/tensorrt_models/faster_rcnn

if [[ $MODEL == 'fasterrcnn' ]]; then
    # Faster-RCNN (mmdetection, nips-2015)
    if [ -d "$FASTERCNN_PATH" ]; then
        echo "The fasterrcnn directory is existing!"
    else
        mkdir "$FASTERCNN_PATH"
    fi
    wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth
    mv faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth ./temp/checkpoints/faster_rcnn.pth
    # # convert pytorch to tensorrt
    docker run --gpus all -ti --network=host -v $DOCKER_TEMP:/root/workspace/temp mmdeploy-gpu \
    python -W ignore /root/workspace/mmdeploy/tools/deploy.py \
    /root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py \
    /root/workspace/temp/checkpoints/faster_rcnn.pth \
    /root/workspace/temp/testdata/0000008_01999_d_0000040.jpg \
    --work-dir /root/workspace/temp/tensorrt_models/faster_rcnn \
    --device cuda:0 && \
    # test the speed
    docker run --gpus all -ti --network=host -v $DOCKER_TEMP:/root/workspace/temp --privileged mmdeploy-gpu python -W ignore /root/workspace/mmdeploy/tools/profiler.py /root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py /mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py /root/workspace/temp/testdata/ \
    --model /root/workspace/temp/tensorrt_models/faster_rcnn/end2end.engine \
    --device cuda --shape 320x320 --num-iter 100
elif [[ $MODEL == 'yolov3' ]]; then
    # YOLOv3 (mmdetection, arxiv-2018)
    if [ -d "$YOLOV3_PATH" ]; then
        echo "The yolov3 directory is existing!"
    else
        mkdir "$YOLOV3_PATH"
    fi
    wget -c https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth
    mv yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth ./temp/checkpoints/yolov3.pth
    # # convert pytorch to tensorrt
    docker run --gpus all -ti --network=host -v $DOCKER_TEMP:/root/workspace/temp mmdeploy-gpu \
    python -W ignore /root/workspace/mmdeploy/tools/deploy.py \
    /root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
    /mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py \
    /root/workspace/temp/checkpoints/yolov3.pth \
    /root/workspace/temp/testdata/0000008_01999_d_0000040.jpg \
    --work-dir /root/workspace/temp/tensorrt_models/yolov3 \
    --device cuda:0 && \
    # test the speed
    docker run --gpus all -ti --network=host -v $DOCKER_TEMP:/root/workspace/temp --privileged mmdeploy-gpu python -W ignore /root/workspace/mmdeploy/tools/profiler.py /root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py /mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py /root/workspace/temp/testdata/ \
    --model /root/workspace/temp/tensorrt_models/yolov3/end2end.engine \
    --device cuda --shape 320x320 --num-iter 100
    # DETR (mmdetection, eccv-2021)
    # wget -c https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth
    # mv detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth ./openppl/temp/checkpoints/detr.pth
    # docker run --gpus all -ti --network=host -v $DOCKER_TEMP:/mmdetection/temp mmdetection python -W ignore ./tools/deployment/pytorch2onnx.py ./configs/detr/detr_r50_8x2_150e_coco.py \
    # ./temp/checkpoints/detr.pth --output-file ./temp/checkpoints/detr.onnx --simplify --dynamic-export
    # docker run -ti --network host -v $DOCKER_TEMP:/ppl.nn/temp -e PYTHONPATH=$PYTHONPATH:/ppl.nn/pplnn-build/install/lib openppl python3 -W ignore ./temp/scripts/inference.py -i ./temp/testdata/0000008_01999_d_0000040.jpg -o ./temp/detr_output.jpg -m ./temp/checkpoints/detr.onnx
elif [[ $MODEL == 'yolox' ]]; then
    # YOLOX (mmdetection, arxiv-2021)
    wget -c https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
    mv yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth ./openppl/temp/checkpoints/yolox.pth
    docker run --gpus all -ti --network=host -v $DOCKER_TEMP:/mmdetection/temp mmdetection python -W ignore ./tools/deployment/pytorch2onnx.py ./configs/yolox/yolox_x_8x8_300e_coco.py \
    ./temp/checkpoints/yolox.pth --output-file ./temp/checkpoints/yolox.onnx --simplify --dynamic-export
    docker run -ti --network host -v $DOCKER_TEMP:/ppl.nn/temp -e PYTHONPATH=$PYTHONPATH:/ppl.nn/pplnn-build/install/lib openppl python3 -W ignore ./temp/scripts/inference.py -i ./temp/testdata/0000008_01999_d_0000040.jpg -o ./temp/yolox_output.jpg -m ./temp/checkpoints/yolox.onnx
# elif [[ $MODEL == 'swin_transformer' ]]; then
    # Swin-Transformer (mmdetection, iccv-2021)
    # echo 'hello! swin-transformer'
fi

