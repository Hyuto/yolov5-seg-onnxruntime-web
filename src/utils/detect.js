import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    if (width % stride >= stride / 2) width = (Math.floor(width / stride) + 1) * stride;
    else width = Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    if (height % stride >= stride / 2) height = (Math.floor(height / stride) + 1) * stride;
    else height = Math.floor(height / stride) * stride;
  }
  return [width, height];
};

const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT, [0, 0, 0, 255]); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio, maxSize];
};

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv5 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} confThreshold Float representing the threshold for deciding when to remove boxes based on confidence score
 * @param {Number} classThreshold class threshold
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  confThreshold,
  classThreshold,
  inputShape
) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);

  const [input, xRatio, yRatio, maxSize] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor("float32", new Float32Array([topk, iouThreshold, confThreshold])); // nms config tensor
  const { output0, output1 } = await session.net.run({ images: tensor }); // run session and get output layer
  const { selected_idx } = await session.nms.run({ detection: output0, config: config }); // get selected idx from nms

  const boxes = [];

  // looping through output
  for (let idx = 0; idx < output0.dims[1]; idx++) {
    if (!selected_idx.data.includes(idx)) continue; // skip if index isn't selected

    const data = output0.data.slice(idx * output0.dims[2], (idx + 1) * output0.dims[2]); // get rows
    let [x, y, w, h] = data.slice(0, 4);
    const confidence = data[4]; // detection confidence
    const scores = data.slice(5, 85); // classes probability scores
    let score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    score *= confidence; // multiply score by conf

    // filtering by score thresholds
    if (score >= classThreshold) {
      x = Math.floor((x - 0.5 * w) * xRatio); // left
      y = Math.floor((y - 0.5 * h) * yRatio); //top
      w = Math.floor(w * xRatio); // width
      h = Math.floor(h * yRatio); // height

      boxes.push({
        label: label,
        probability: score,
        bounding: [x, y, w, h],
      });

      console.log([x, y, w, h, maxSize]);

      const mask = new Tensor("float32", new Float32Array([x, y, w, h, ...data.slice(85)]));
      const maskConfig = new Tensor("float32", new Float32Array([640, 255, 255, 255, 150]));
      const masked = await session.mask.run({
        detection: mask,
        mask: output1,
        config: maskConfig,
      });
      console.log(masked);
    }
  }

  renderBoxes(canvas, boxes); // Draw boxes

  input.delete();
};
