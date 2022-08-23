# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()

def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    device = select_device('')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    imgsz = check_img_size(imgsz, s=model.stride)  # check image size
    model.warmup(imgsz=(1 if model.pt else 1, 3, *imgsz))  # warmup
    dataset = LoadImages(source, img_size=imgsz, stride=model.stride, auto=model.pt)
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        return (detect(im,im0s,dt,model))

def detect(im,im0s,dt,model):
    resutls=[]
    with dt[0]:
        im = torch.from_numpy(im).to(select_device())
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    # Inference
    with dt[1]:
        visualize = False
        pred = model(im, augment=False, visualize=visualize)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0 =  im0s.copy()

        resutls=[]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                resutls.append({"label":c,"conf":conf,"xyxy":xyxy})
    return resutls

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    source='data/images/bus.jpg'
    targets=(run(source=source))
    with open('data/test/bus.json', "w") as fil: # Opens the file using 'w' method. See below for list of methods.
        for index in range(len(targets)):
            target=targets[index]
            if target['label']==0:
                img = cv2.imread(source)
                crop_img = img[int(target['xyxy'][1]):int(target['xyxy'][3]),int(target['xyxy'][0]):int(target['xyxy'][2])]
                cv2.imwrite('data/test/cropped'+str(index)+'.jpg', crop_img)
                data=(run(source='data/test/cropped'+str(index)+'.jpg',weights='mtfl30.pt'))
                if len(data):
                    print((data[0]['conf']))
                    fil.write(str(data[0]['label']))
                    fil.write(' ')
                    fil.write(str(int(data[0]['xyxy'][0]+target['xyxy'][0])))
                    fil.write(' ')
                    fil.write(str(int(data[0]['xyxy'][1]+target['xyxy'][1])))
                    fil.write(' ')
                    fil.write(str(int(data[0]['xyxy'][2]+target['xyxy'][2])))
                    fil.write(' ')
                    fil.write(str(int(data[0]['xyxy'][3]+target['xyxy'][3])))
                    fil.write('\n')
            