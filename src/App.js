import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState("Loading OpenCV.js...");
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // Configs
  const modelName = "yolov5n-seg.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.4;
  const confThreshold = 0.2;
  const classThreshold = 0.2;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    // create session
    setLoading("Loading YOLOv5 model...");
    const [yolov5, nms, mask] = await Promise.all([
      InferenceSession.create(`${process.env.PUBLIC_URL}/model/${modelName}`),
      InferenceSession.create(`${process.env.PUBLIC_URL}/model/nms-yolov5.onnx`),
      InferenceSession.create(`${process.env.PUBLIC_URL}/model/mask-yolov5-seg.onnx`),
    ]);

    // warmup main model
    setLoading("Warming up model...");
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    await yolov5.run({ images: tensor });

    setSession({ net: yolov5, nms: nms, mask: mask });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && <Loader>{loading}</Loader>}
      <div className="header">
        <h1>YOLOv5 Object Segmentation App</h1>
        <p>
          YOLOv5 object segmentation application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              confThreshold,
              classThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
