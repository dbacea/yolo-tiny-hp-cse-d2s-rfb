# Official Boosting Lightweight Object Detection with Enhanced Feature Fusion and Optimized Receptive Field

Implementation of paper - ["Boosting Lightweight Object Detection with Enhanced Feature Fusion and Optimized Receptive Field"] (https://www.sciencedirect.com/science/article/pii/S0925231226001256).

## Performance 

MS COCO

| Model | Test Size | #Params (M) | BFLOPS | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | AP<sub>75</sub><sup>val</sup> | batch 32 fps on Jetson Nano |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| YOLOv7-tiny | 416 | 6.2 | 5.8 | **35.2%** | **52.8%** | **37.3%** | 14 *fps* |
| NanoDet-M-1.5x | 416 | 2.08 | 2.42 | **26.8%** | **-** | **-** | - *fps* |
| PP-PicoDet-M | 416 | 2.15 | 2.5 | **34.3%** | **49.8%** | **-** | - *fps* |
| YOLOv8-n | 416 | 3.15 | 3.7 | **32.0%** | **46.5%** | **34.1%** | 14 *fps* |
| YOLOv9-t | 416 | 2.40 | 4.1 | **32.2%** | **46.5%** | **34.2%** | 7 *fps* |
| YOLOv10-n | 416 | 2.29 | 2.8 | **32.6%** | **47.2%** | **35.0%** | 14 *fps* |
| YOLOv10-n | 416 | 2.29 | 2.8 | **32.6%** | **47.2%** | **35.0%** | 14 *fps* |
| YOLOv11-n | 416 | 2.15 | 2.74 | **32.5%** | **46.8%** | **34.7%** | 18 *fps* |
| YOLOv12-n | 416 | 2.54 | 2.5 | **33.3%** | **48.2%** | **35.4%** | 12 *fps* |
| ECF-YOLOv7-tiny | 416 | 5.97 | 9.3 | **37.8%** | **56.0%** | **40.0%** | 9 *fps* |
| **YOLOv7-tiny <br/> + HP-CSE + D2S-RFB** | 416 | 6.76 | 8.0 |  **38.8%** | **56.5%** | **41.7%** | 8 *fps* |
| **YOLOv12-n <br/>+ HP-CSE + D2S-RFB** | 416 | 3.01 | 3.7 | **35.0%** | **50.1%** | **37.4%** | 10 *fps* |

## Installation

Docker environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolo-tiny-hp-cse-d2s-rfb -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo-tiny-hp-cse-d2s-rfb --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd ./yolo-tiny-hp-cse-d2s-rfb
```

</details>

## Testing

``` shell
#test on MS COCO the model yolov7-tiny-hp-cse-d2s-rfb
python test.py --data data/coco.yaml --img 416 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights weights/yolov7-tiny-hp-cse-d2s-rfb.pt --name  yolov7-tiny-hp-cse-d2s-rfb_416_val

```

You will get the results:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.565
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.353
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.768
```

To measure accuracy, download [COCO-annotations for Pycocotools](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) to the `./coco/annotations/instances_val2017.json`


To test on the Pascal VOC dataset, please use the command below:
``` shell
#test on Pascal VOC the model yolov7-tiny-hp-cse-d2s-rfb which was trained on MS COCO
python test_voc.py --data data/VOC.yaml --task test --img 416 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights weights/yolov7-tiny-hp-cse-d2s-rfb.pt --name  yolov7-tiny-hp-cse-d2s-rfb_416_val
```

For downloading the Pascal VOC dataset, you could follow either of the two tutorials from https://docs.ultralytics.com/datasets/detect/voc/ or https://github.com/AlexeyAB/darknet/wiki/Train-and-Evaluate-Detector-on-Pascal-VOC-(VOCtrainval-2007-2012)-dataset 


To test any of the ablation studies, please replace the --weights with the desired model in the shell command from above. One of the models from the sub-directory weights/ablation could be selected. For the YOLOv8, YOLOv9, YOLOv10, YOLOv11 and YOLOv12 models, please follow the below instructions.
Classes that need be added into the codebases of YOLOv8, YOLOv9, YOLOv10, YOLOv11 and YOLOv12 for being able to use our proposed modules HP-CSE and D2S-RFB.

```
class DepthwiseSeparableConvBN(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, act=True): 
        super(DepthwiseSeparableConvBN, self).__init__() 
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin) 
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1) 
        self.bn = nn.BatchNorm2d(nout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
  
    def forward(self, x): 
        out = self.depthwise(x) 
        out = self.pointwise(out) 
        return self.act(self.bn(out))

    def fuseforward(self, x):
        out = self.depthwise(x) 
        out = self.pointwise(out) 
        return self.act(out)

class DepthwiseSeparableDilatedConvBN(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, kernel_size=3, p=None, d=1, act=True): 
        super(DepthwiseSeparableDilatedConvBN, self).__init__() 
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=autopad(kernel_size, p), dilation=d, groups=nin) 
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1) 
        self.bn = nn.BatchNorm2d(nout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
  
    def forward(self, x): 
        out = self.depthwise(x) 
        out = self.pointwise(out) 
        return self.act(self.bn(out))

    def fuseforward(self, x):
        out = self.depthwise(x) 
        out = self.pointwise(out) 
        return self.act(out)

#only needed for compatibility reasons with already trained models (see the weight files included in the repo), this block is the same as the HP-CSE.
class CSE_CAMv1(nn.Module):
    def __init__(self, c, r):
        super(CSE_CAMv1, self).__init__()
        c_o = c // r
        self.maxsqueeze = nn.AdaptiveMaxPool2d(1)
        self.avgsqueeze = nn.AdaptiveAvgPool2d(1)

        self.conv = Conv(c, c, 1, 1, None, 1, 1, nn.Mish())
        self.linear = nn.Sequential(
            nn.Conv1d(c, c_o, 1, 1, 0, 1, 1, False),
            nn.Mish(),
            nn.Conv1d(c_o, c, 1, 1, 0, 1, 1, False))

    def forward(self, x):
        b, c, _, _ = x.size()
        max = self.maxsqueeze(x).view(b,c)
        avg = self.avgsqueeze(x).view(b,c)
        max = torch.unsqueeze(max, 2)
        avg = torch.unsqueeze(avg, 2)
        linear_max = self.linear(max).view(b, c, 1, 1)
        linear_avg = self.linear(avg).view(b, c, 1, 1)
       	output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        output = self.conv(output)
        return output

class HP-CSE(nn.Module):
    def __init__(self, c, r):
        super(HP-CSE, self).__init__()
        c_o = c // r
        self.maxsqueeze = nn.AdaptiveMaxPool2d(1)
        self.avgsqueeze = nn.AdaptiveAvgPool2d(1)

        self.conv = Conv(c, c, 1, 1, None, 1, 1, nn.Mish())
        self.linear = nn.Sequential(
            nn.Conv1d(c, c_o, 1, 1, 0, 1, 1, False),
            nn.Mish(),
            nn.Conv1d(c_o, c, 1, 1, 0, 1, 1, False))

    def forward(self, x):
        b, c, _, _ = x.size()
        max = self.maxsqueeze(x).view(b,c)
        avg = self.avgsqueeze(x).view(b,c)
        max = torch.unsqueeze(max, 2)
        avg = torch.unsqueeze(avg, 2)
        linear_max = self.linear(max).view(b, c, 1, 1)
        linear_avg = self.linear(avg).view(b, c, 1, 1)
       	output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        output = self.conv(output)
        return output
		
class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        if len(x) == 2:
            return x[0]+x[1]
        elif len(x) == 3:
            return x[0]+x[1]+x[2]
        elif len(x) == 4:
            return x[0]+x[1]+x[2]+x[3]
        elif len(x) == 5:
            return x[0]+x[1]+x[2]+x[3]+x[4]

```

To test YOLOv9-t adapted with HP-CSE and D2S-RFB: 
1. git clone the YOLOv9 official code base from https://github.com/WongKinYiu/yolov9. 
2. After cloning the repo, you need to take the three modules from above (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN and DepthwiseSeparableDilatedConvBN) and add them into python file from path: yolov9-main/models/common.py
3. In the same file, extend the existing Shortcut class to the implementation attached above.
4. After copying the three modules, you can use the following command to test the module:
``` shell
#test yolov9-t-hp-cse-d2s-rfb
python val.py --data data/coco.yaml --img 416 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights 'weights/yolov9-t-hp-cse-d2s-rfb.pt' --save-json --name yolov9-t-hp-cse-d2s-rfb_416_val

```

To test YOLOv8-n, YOLOv10-n and YOLOv11-n adapted with HP-CSE and D2S-RFB: 
1. git clone the ultralytics official code base from https://github.com/ultralytics/ultralytics.
2. After cloning the repo, you need to take the four modules from above (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) and add them into python file from path: ultralytics/ultralytics/nn/modules/block.py
3. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the __all__ variable from line 13, into the same file.
4. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the __all__ variable from line 105, into the python file from path: ultralytics/ultralytics/nn/modules/__init__.py.
5. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the import statement from line 20, into the python file from path: ultralytics/ultralytics/nn/modules/__init__.py.
6. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the import statement from line 14, into the python file from path: ultralytics/ultralytics/nn/tasks.py
7. Add the following elif code at the line 1063, into the python file from path: ultralytics/ultralytics/nn/tasks.py
``` 
        elif m is Shortcut:
            c2 = ch[f[0]]
``` 
8. Navigate to the root folder of the clone repository and create/run the one of the following python scripts (first one is for yolov8-n-hp-cse-d2s-rfb, second one is for yolov10-n-hp-cse-d2s-rfb and the third one is for yolov11-n-hp-cse-d2s-rfb):

``` 
#python script content for testing yolov8-n-hp-cse-d2s-rfb
from ultralytics import YOLO

# Load a model
model = YOLO("yolo-tiny-hp-cse-d2s-rfb/weights/yolov8-n-hp-cse-d2s-rfb.pt") #please insert here the path on your hard-disk to the model file

# Validate the model
validation_results = model.val(data="coco.yaml", imgsz=416, batch=32, device="0")
``` 

``` 
#python script content for testing yolov10-n-hp-cse-d2s-rfb
from ultralytics import YOLO

# Load a model
model = YOLO("yolo-tiny-hp-cse-d2s-rfb/weights/yolov10-n-hp-cse-d2s-rfb.pt") #please insert here the path on your hard-disk to the model file

# Validate the model
validation_results = model.val(data="coco.yaml", imgsz=416, batch=32, device="0")
``` 

``` 
#python script content for testing yolov11-n-hp-cse-d2s-rfb
from ultralytics import YOLO

# Load a model
model = YOLO("yolo-tiny-hp-cse-d2s-rfb/weights/yolov11-n-hp-cse-d2s-rfb.pt") #please insert here the path on your hard-disk to the model file

# Validate the model
validation_results = model.val(data="coco.yaml", imgsz=416, batch=32, device="0")
```


To test YOLOv12-n adapted with HP-CSE and D2S-RFB:
1. git clone the YOLOv12 official code base from https://github.com/sunsmarterjie/yolov12.
2. After cloning the repo, you need to take the four modules from above (HP-CSE, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) and add them into python file from path: ultralytics/ultralytics/nn/modules/block.py
3. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the __all__ variable from line 13, into the same file.
4. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the __all__ variable from line 100, into the python file from path: ultralytics/ultralytics/nn/modules/__init__.py.
5. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the import statement from line 20, into the python file from path: ultralytics/ultralytics/nn/modules/__init__.py.
6. Add the modules names (HP-CSE, CSE_CAMv1, DepthwiseSeparableConvBN, DepthwiseSeparableDilatedConvBN and Shortcut) into the import statement from line 14, into the python file from path: ultralytics/ultralytics/nn/tasks.py
7. Add the following elif code at the line 1075, into the python file from path: ultralytics/ultralytics/nn/tasks.py
``` 
        elif m is Shortcut:
            c2 = ch[f[0]]
``` 
8. Navigate to the root folder of the clone repository and create/run the one of the following python script (for yolov12-n-hp-cse-d2s-rfb):

``` 
#python script content for testing yolov12-n-hp-cse-d2s-rfb
from ultralytics import YOLO

# Load a model
model = YOLO("yolo-tiny-hp-cse-d2s-rfb/weights/yolov12-n-hp-cse-d2s-rfb.pt") #please insert here the path on your hard-disk to the model file

# Validate the model
validation_results = model.val(data="coco.yaml", imgsz=416, batch=32, device="0")
```


All the weights of the models from the ablation studies can be found within the weights folder.


## Training

Data preparation for MS COCO

``` shell
bash scripts/get_coco.sh
```

* Download MS COCO dataset images ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). If you have previously used a different version of YOLO, we strongly recommend that you delete `train2017.cache` and `val2017.cache` files, and redownload [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip) 

Single GPU training

``` shell
# train yolov7-tiny-hp-cse-d2s-rfb. The average train duration is roughly 4-5 days, the gpu usage being 24 GB
python train.py --workers 8 --device 0 --batch-size 128 --epochs 500 --data data/coco.yaml --img 416 416 --cfg cfg/yolov7-tiny-hp-cse-d2s-rfb.yaml --weights '' --name yolov7-tiny-hp-cse-d2s-rfb --hyp data/hyp.scratch.tiny.yaml

```

Multiple GPU training

``` shell
# train yolov7-tiny-hp-cse-d2s-rfb
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 train.py --workers 16 --device 0,1 --sync-bn --batch-size 256 --epochs 500 --data data/coco.yaml --img 416 416 --cfg cfg/yolov7-tiny-hp-cse-d2s-rfb.yaml --weights '' --name yolov7-tiny-hp-cse-d2s-rfb --hyp data/hyp.scratch.tiny.yaml

```

Inside the sub-directory cfg you can find the yaml configurations with the tiny/nano network architectures for YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11 and YOLOv12.

## Inference

On video:
``` shell
python detect.py --weights weights/yolov7-tiny-hp-cse-d2s-rfb.pt --conf 0.25 --img-size 416 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights weights/yolov7-tiny-hp-cse-d2s-rfb.pt --conf 0.25 --img-size 416 --source inference/images/horses.jpg
```

## Check BFLOPs
``` shell
pip install thop
python detect_flops.py --weights weights/yolov7-tiny-hp-cse-d2s-rfb.pt --img-size 416 --conf 0.25 --source inference/images/horses.jpg
```

## Check speed
``` shell
python test.py --task speed --data data/coco.yaml --img 416 --batch 32 --conf 0.001 --iou 0.7 --device 0 --weights weights/yolov7-tiny-hp-cse-d2s-rfb.pt --name yolov7-tiny-hp-cse-d2s-rfb_416_b32_val_500ep
```

</details>

Tested with: Python 3.7.13, Pytorch 1.11.0+cu116


## Citation

```
@article{bacea2026boosting,
  title={Boosting lightweight object detection with enhanced feature fusion and optimized receptive field},
  author={Bacea, Dan-Sebastian and Oniga, Florin},
  journal={Neurocomputing},
  pages={132728},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)
* [https://github.com/dbacea/ecf-yolov7-tiny]

</details>
