# Driver-Drowsiness-Detection
## 🧠 Overview

This project monitors the driver’s eye movements using a webcam. If the driver appears drowsy (based on eye aspect ratio or a trained CNN model), it triggers an alarm and optionally sends an SMS alert.

## 📌 Features

- Real-time facial landmark detection using `dlib`
- Eye Aspect Ratio (EAR) based drowsiness detection
- Alarm alert using `pygame`
- Web interface using Flask to stream live video
- Optional:
  - SMS alerts using Twilio API
  - Deep learning-based drowsiness detection using a custom CNN
- Portable and ready for integration in vehicle systems

## 🎯 Tech Stack

- Python 3.x
- OpenCV
- dlib
- Flask
- Pygame (for alarm)
- Twilio (for mobile alerts)
- TensorFlow/Keras (for CNN model)
Due to GitHub’s file size limit, you need to manually download the facial landmark predictor model:

Download here (official source):
shape_predictor_68_face_landmarks.dat.bz2

Extract the file using any archive tool (like WinRAR or 7-Zip).
You will get a file named:
shape_predictor_68_face_landmarks.dat

Place this file in your project root folder (same directory as main.py).
