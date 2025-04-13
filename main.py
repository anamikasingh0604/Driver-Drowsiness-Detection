import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import threading
from flask import Flask, Response
import pygame


app = Flask(__name__)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear



def sound_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("mixkit-classic-alarm-995.wav")
    pygame.mixer.music.play()

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))


cap = cv2.VideoCapture(0)

THRESHOLD_EAR = 0.25
DROWSY_FRAMES = 20
frame_count = 0



def generate_frames():
    global frame_count
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Draw eyes
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            if ear < THRESHOLD_EAR:
                frame_count += 1
                if frame_count >= DROWSY_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    threading.Thread(target=sound_alarm).start()
            else:
                frame_count = 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
