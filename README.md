# Object_detectiion_using_YOLOv8
Real-Time Object Detection with YOLOv8 and OpenCV 🚀🎥

⚠️ DISCLAMER :-
webcam not working, cause-
When deploying to Streamlit Cloud, webcam or device access can be problematic because it's typically not supported on cloud platforms for security and privacy reasons.
frontend code - https://lnkd.in/gCk8Bpji (Try this frontend)

In this project, I built a real-time object detection system using the powerful YOLOv8 model and OpenCV. The system processes live webcam footage, identifies objects in the frame, and overlays bounding boxes with labels and confidence scores. Here's how it works:

🧩 Key Features:

🧭 Pre-Trained YOLOv8 Model:
The YOLOv8 model, pre-trained on the COCO dataset, can detect 80 common objects like people, cars, chairs, and more.
It's optimized for speed and accuracy, making it ideal for real-time applications.

🌐 Live Webcam Feed:
Captures frames from the webcam in real-time and passes them through the detection pipeline.

✴️ Object Detection Pipeline:
Bounding Boxes: Draws rectangles around detected objects.
Labels & Confidence Scores: Displays the object name and its detection confidence percentage.

〽️ Randomized Colors for Classes:
Each detected class is assigned a unique color to distinguish objects visually.
Streamlined Performance:
Processes frames continuously in a loop for smooth, real-time detection.

🛠️ How It Works:
🌟 COCO Labels: The model uses a list of object classes from the COCO dataset (e.g., "person," "bicycle," "cat").
🌟YOLO Prediction: Each frame is processed by YOLO, which returns:
Class ID (object type),
Bounding box coordinates,
Detection confidence.
🌟Visualization: OpenCV is used to draw rectangles and display labels on the live video feed.

🔍 Code Workflow:
🌟Load YOLOv8 Model:
Pre-trained weights are loaded to enable out-of-the-box object detection.
🌟Live Video Capture:
OpenCV captures video frames from the webcam.
🌟Frame Processing:
Each frame is sent to YOLO for predictions.
🌟Detection Overlay:
Bounding boxes and labels are drawn on the frame for detected objects.
🌟Real-Time Display:
The processed frame is displayed in a live OpenCV window.

💻 Tech Stack:
🌟YOLOv8: State-of-the-art object detection model.
🌟OpenCV: Real-time image processing library.
🌟Python: The programming language that ties everything together.

🚀 How to Run:
🌟Install the required libraries: ultralytics, opencv-python, numpy.
Ensure your device's webcam is functional.
Run the script and observe real-time object detection in action!
Press Q to exit the detection loop.

🎯 Use Cases:
🌟 Smart Surveillance: Detect and track objects in security feeds.
🌟Retail Analytics: Monitor customer activity in real time.
🌟Autonomous Systems: Enhance perception in robots and drones.
