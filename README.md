😷 Face Mask Detection with Email Alerts

This project detects whether a person is wearing a face mask or not using computer vision and deep learning techniques.
It uses a pre-trained face detection model (Haar Cascade) and a custom-trained CNN model to classify faces into:

✅ With Mask
❌ Without Mask (with email alert notification)

🛠️ Features
Real-time face mask detection using a webcam.
Sends email alerts when a person without a mask is detected.
Supports image and video input.
Uses OpenCV for face detection.
Uses TensorFlow/Keras for deep learning model inference.
High accuracy for mask classification.

📂 Project Structure
graphql
Copy
Edit
FACE MASK DETECTION/
│
├── MyTrainingModel.keras         # Trained CNN model file
├── test.py                       # Main detection + email alert script
├── ProcessingAndTraining.ipynb   # Model training notebook
├── dataset/                      # Dataset (images with/without mask)
├── haarcascade_frontalface_default.xml # Face detection model
├── requirements.txt              # Required Python libraries
├── outputs/                      # Example output images

📊 Technologies Used
Python 3
OpenCV
TensorFlow / Keras
NumPy
smtplib (for sending emails)

📈 Accuracy
The trained model achieves high accuracy on test images and works in real-time.
When a "Without Mask" face is detected, the system triggers an email alert to a specified address.

📸 Outputs
1️⃣ With Mask Detection
2️⃣ Without Mask Detection (Email Alert Triggered)
