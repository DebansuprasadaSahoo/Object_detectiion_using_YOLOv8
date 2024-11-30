import random
import cv2
import numpy as np
from ultralytics import YOLO

# Load the COCO class labels
with open(r"C:\Users\deban\Downloads\Object detection\coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Generate random colors for each class
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(len(class_list))
]

# Load a pretrained YOLOv8 model
model = YOLO(r"C:\Users\deban\Downloads\Object detection\yolov8n.pt")

# Set webcam as video source (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is not read correctly, break the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Predict on the current frame
    results = model.predict(source=[frame], conf=0.45, save=False, verbose=False)

    # Retrieve detections from results
    detections = results[0]
    if detections.boxes is not None:
        for box in detections.boxes:
            clsID = int(box.cls.numpy()[0])  # Class ID
            conf = box.conf.numpy()[0]  # Confidence score
            bb = box.xyxy.numpy()[0]  # Bounding box coordinates

            # Draw the bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                2,
            )

            # Display the class name and confidence
            label = f"{class_list[clsID]}: {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (int(bb[0]), int(bb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
