import random
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the COCO class labels
try:
    with open(r"C:\Users\deban\Downloads\Object detection\coco.txt", "r") as f:
        class_list = f.read().splitlines()
except FileNotFoundError:
    st.error("COCO labels file not found. Please ensure the file path is correct.")
    st.stop()

# Generate random colors for each class
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(len(class_list))
]

# Load a pretrained YOLOv8 model
try:
    model = YOLO(r"C:\Users\deban\Downloads\Object detection\yolov8n.pt")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()


# Video Transformer for Streamlit WebRTC
class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert image from BGR to RGB
        img = frame.to_ndarray(format="bgr24")

        # Predict on the current frame
        try:
            results = model.predict(source=[img], conf=0.45, save=False, verbose=False)
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
            return img

        # Retrieve detections from results
        detections = results[0]
        if detections.boxes is not None:
            for box in detections.boxes:
                clsID = int(box.cls.numpy()[0])  # Class ID
                conf = box.conf.numpy()[0]  # Confidence score
                bb = box.xyxy.numpy()[0]  # Bounding box coordinates

                # Draw the bounding box
                cv2.rectangle(
                    img,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[clsID],
                    2,
                )

                # Display the class name and confidence
                label = f"{class_list[clsID]}: {conf:.2f}"
                cv2.putText(
                    img,
                    label,
                    (int(bb[0]), int(bb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

        return img


# Streamlit App
st.title("Real-time Object Detection with YOLOv8")

st.write("""
This application uses YOLOv8 for real-time object detection. 
Click 'Start' to begin the webcam feed and detect objects.
""")

# Start the WebRTC Stream
try:
    webrtc_streamer(key="object-detection", video_transformer_factory=YOLOVideoTransformer)
except Exception as e:
    st.error(f"WebRTC Stream Error: {e}")
