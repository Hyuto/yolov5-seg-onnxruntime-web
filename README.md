# YOLOv5 Segmentation with onnxruntime-web

<p align="center">
  <img src="./sample.png" />
</p>

![love](https://img.shields.io/badge/Made%20with-🖤-white)
![react](https://img.shields.io/badge/React-blue?logo=react)
![onnxruntime-web](https://img.shields.io/badge/onnxruntime--web-white?logo=onnx&logoColor=black)
![opencv.js](https://img.shields.io/badge/opencv.js-green?logo=opencv)

---

Object Segmentation application right in your browser.
Serving YOLOv5 segmentation in browser using onnxruntime-web with `wasm` backend.

## Setup

```bash
git clone https://github.com/Hyuto/yolov5-seg-onnxruntime-web.git
cd yolov5-seg-onnxruntime-web
yarn install # Install dependencies
```

## Scripts

```bash
yarn start # Start dev server
yarn build # Build for productions
```

## Models

**Main Model**

YOLOv5n-seg model converted to onnx.

```
used model : yolov5n-seg.onnx
size       : 8 Mb
```

**NMS**

ONNX model to perform NMS operator [CUSTOM].

```
nms-yolov5.onnx
```

**Mask**

ONNX model to produce mask for every object detected [CUSTOM].

```
mask-yolov5-seg.onnx
```

## Use another model

> :warning: **Size Overload** : used YOLOv5 segmentation model in this repo is the smallest with size of 8 MB, so other models is definitely bigger than this which can cause memory problems on browser.

Use another YOLOv5 model.

1. Clone [yolov5](https://github.com/ultralytics/yolov5) repository

   ```bash
   git clone https://github.com/ultralytics/yolov5.git && cd yolov5
   ```

   Install `requirements.txt` first

   ```bash
   pip install -r requirements.txt
   ```

2. Export model to onnx format
   ```bash
   python export.py --weights yolov5*-seg.pt --include onnx --img 640
   ```
3. Copy `yolov5*-seg.onnx` to `./public/model`
4. Update `modelName` in `App.jsx` to new model name
   ```jsx
   ...
   // configs
   const modelName = "yolov5*-seg.onnx"; // change to new model name
   const modelInputShape = [1, 3, 640, 640];
   const topk = 100;
   const iouThreshold = 0.4;
   const confThreshold = 0.2;
   const classThreshold = 0.2;
   ...
   ```
5. Done! 😊

## Reference

- https://github.com/ultralytics/yolov5
- https://github.com/UNeedCryDear/yolov5-seg-opencv-dnn-cpp
