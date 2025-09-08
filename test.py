import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

def eye_center(landmarks, idxs, w, h):
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    return np.mean(xs), np.mean(ys)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                # центры глаз
                lx, ly = eye_center(face_landmarks.landmark, LEFT_EYE, w, h)
                rx, ry = eye_center(face_landmarks.landmark, RIGHT_EYE, w, h)

                # вектор между глазами
                dx, dy = rx - lx, ry - ly
                eye_dist = math.hypot(dx, dy)
                angle = math.degrees(math.atan2(dy, dx))

                # центр обоих глаз
                cx, cy = (lx + rx) / 2, (ly + ry) / 2

                # размеры прямоугольника (подбираются экспериментально)
                box_w = eye_dist * 2.0   # ширина = 2× расстояние между глазами
                box_h = eye_dist * 0.6   # высота = 1.2× расстояние

                # создаём rotated rect
                rect = ((cx, cy), (box_w, box_h), angle)
                box = cv2.boxPoints(rect)
                box = box.astype(int)

                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)


        cv2.imshow('Rotated Eyes Box - MediaPipe', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Esc
            break

cap.release()
cv2.destroyAllWindows()
