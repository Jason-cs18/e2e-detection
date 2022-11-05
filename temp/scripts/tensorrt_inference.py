# inference script using tensorrt engine
from mmdeploy.apis import inference_model
import sys
import cv2
import mmcv
import os
import time

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def video_process(test_video, model_cfg, deploy_cfg, backend_files, device, name):
    # xmin, ymin, xmax, ymax, confidence
    # car (2), bus (5), truck (7)
    img_index = 0
    video = mmcv.VideoReader(test_video)
    for image in video:
        img_index += 1
        # if you need to draw segmentation points, please modify the code above
        if name == 'swin':
            result = inference_model(model_cfg, deploy_cfg, backend_files, image, device)[0][0]
        else:
            result = inference_model(model_cfg, deploy_cfg, backend_files, image, device)[0]
        for class_index in range(len(result)):
            # print(class_index)
            if class_index == 2:
                class_name = 'car'
                color = (255, 0, 0)    
            elif class_index == 5:
                class_name = 'bus'
                color = (0, 255, 0)
            elif class_index == 7:
                class_name = 'truck'
                color = (0, 0, 255)
            else: continue
            for bbox in result[class_index]:
                if bbox[-1] > 0.5:
                    # print(type(bbox[0]))
                    confidence = bbox[-1] * 100
                    text = f'{class_name}-{confidence:.1f}'
                    image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color)
                    image = cv2.putText(image, text, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color)
        cv2.imwrite(f'/root/workspace/temp/outputs/img{img_index:05d}.png', image)
        if img_index == 100:
            break
    os.system(f"ffmpeg -r 1 -i /root/workspace/temp/outputs/img%05d.png -vcodec mpeg4 -y /root/workspace/temp/outputs/{name}-movie.mp4")
    os.system("rm /root/workspace/temp/outputs/*.png")


def main(model_name):
    if model_name == 'faster_rcnn':
        print('inference with Faster-RCNN:')
        model_cfg = '/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.py'
        deploy_cfg = '/root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py'
        backend_files = ['/root/workspace/temp/tensorrt_models/faster_rcnn/end2end.engine']
        # test_img = '/root/workspace/temp/testdata/0000008_01999_d_0000040.jpg'
        test_video = '/root/workspace/temp/testdata/vdo.avi'
        device = 'cuda'
        # result = inference_model(model_cfg, deploy_cfg, backend_files, test_img, device)
        print('*'*20)
        video_process(test_video, model_cfg, deploy_cfg, backend_files, device, model_name)
    elif model_name == 'yolov3':
        print('inference with YOLOv3:')
        model_cfg = '/mmdetection/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
        deploy_cfg = '/root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py'
        backend_files = ['/root/workspace/temp/tensorrt_models/yolov3/end2end.engine']
        # test_img = '/root/workspace/temp/testdata/0000008_01999_d_0000040.jpg'
        test_video = '/root/workspace/temp/testdata/vdo.avi'
        device = 'cuda'
        # result = inference_model(model_cfg, deploy_cfg, backend_files, test_img, device)
        print('*'*20)
        video_process(test_video, model_cfg, deploy_cfg, backend_files, device, model_name)
    elif model_name == 'detr':
        print('inference with DETR:')
        model_cfg = '/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py'
        deploy_cfg = '/root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py'
        backend_files = ['/root/workspace/temp/tensorrt_models/detr/end2end.engine']
        # test_img = '/root/workspace/temp/testdata/0000008_01999_d_0000040.jpg'
        test_video = '/root/workspace/temp/testdata/vdo.avi'
        device = 'cuda'
        # result = inference_model(model_cfg, deploy_cfg, backend_files, test_img, device)
        print('*'*20)
        video_process(test_video, model_cfg, deploy_cfg, backend_files, device, model_name)
    elif model_name == 'swin':
        print('inference with Swin:')
        model_cfg = '/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
        deploy_cfg = '/root/workspace/mmdeploy/configs/mmdet/instance-seg/instance-seg_tensorrt_dynamic-320x320-1344x1344.py'
        backend_files = ['/root/workspace/temp/tensorrt_models/swin_transformer/end2end.engine']
        # test_img = '/root/workspace/temp/testdata/0000008_01999_d_0000040.jpg'
        test_video = '/root/workspace/temp/testdata/vdo.avi'
        device = 'cuda'
        # result = inference_model(model_cfg, deploy_cfg, backend_files, test_img, device)
        print('*'*20)
        # print(len(result[0][0]))
        # print(result[0][0])
        video_process(test_video, model_cfg, deploy_cfg, backend_files, device, model_name)

if __name__ == '__main__':
    model_name = sys.argv[-1]
    main(model_name)