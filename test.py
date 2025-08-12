import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import smtplib
from email.mime.text import MIMEText

# ====== Alert timing globals ======
alert_interval = 30       # seconds between alerts
last_alert_time = 0       # will hold timestamp of last alert

# ====== Email sending (Gmail SMTP) ======
def send_email_alert(subject, body, to_email):
    sender_email = "meghanabajjuri@gmail.com"
    sender_password = "gjzn wzod bgfl urol"  # <- paste the 16-char app password (no spaces)

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to_email

    try:
        # Use SMTP_SSL for Gmail
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Error sending email: {e}")

# ====== Load model & params ======
model = load_model("MyTrainingModel.keras")
imgDimension = (32, 32)
labels = {0: "With Mask", 1: "Without Mask"}
font = cv2.FONT_HERSHEY_SIMPLEX

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = img.astype(np.float32)
    img = img.reshape(imgDimension[1], imgDimension[0], 1)
    return img

def detect_mask_on_webcam():
    global last_alert_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, imgDimension)
            preprocessed_face = preprocessing(resized_face)
            preprocessed_face = np.expand_dims(preprocessed_face, axis=0)

            predictions = model.predict(preprocessed_face, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class] * 100

            label = labels[predicted_class]
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{label}: {confidence:.2f}%"
            cv2.putText(frame, text, (x, y-10), font, 0.5, color, 2)

            if label == "Without Mask":
                current_time = time.time()
                if current_time - last_alert_time > alert_interval:
                    # send alert to yourself
                    send_email_alert(
                        subject="🚨 No Mask Detected!",
                        body=f"A person without a mask was detected (confidence: {confidence:.2f}%).",
                        to_email="meghanabajjuri@gmail.com"  # or any recipient
                    )
                    last_alert_time = current_time

        cv2.imshow('Live Mask Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_mask_on_webcam()
