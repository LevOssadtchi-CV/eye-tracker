import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import os
import random

# --- Настройки ---
OUTPUT_DIR = "eyes_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

POINT_COLOR = (0, 200, 0)
POINT_RADIUS = 15
POINT_SPEED = 8  # пикселей за кадр

# --- MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

# --- Инициализация CSV ---
csv_file = open(CSV_FILE, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "x_min", "y_min", "x_max", "y_max", "point_x", "point_y"])

# --- Функции ---
def eye_center(landmarks, idxs, w, h):
    xs = [landmarks[i].x * w for i in idxs]
    ys = [landmarks[i].y * h for i in idxs]
    return np.mean(xs), np.mean(ys)

def crop_eyes(frame, rect):
    (cx, cy), (w, h), angle = rect
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    x_min = int(cx - w/2)
    x_max = int(cx + w/2)
    y_min = int(cy - h/2)
    y_max = int(cy + h/2)
    crop = rotated[y_min:y_max, x_min:x_max]
    return crop, x_min, y_min, x_max, y_max

# --- Камера ---
cap = cv2.VideoCapture(0)
screen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Начальная позиция и цель ---
point_x = screen_w // 2
point_y = screen_h // 2
target_x = random.randint(POINT_RADIUS, screen_w - POINT_RADIUS)
target_y = random.randint(POINT_RADIUS, screen_h - POINT_RADIUS)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # --- движение точки к цели ---
        dx = target_x - point_x
        dy = target_y - point_y
        dist = math.hypot(dx, dy)

        if dist < POINT_SPEED:  # цель достигнута → выбираем новую
            target_x = random.randint(POINT_RADIUS, screen_w - POINT_RADIUS)
            target_y = random.randint(POINT_RADIUS, screen_h - POINT_RADIUS)
        else:
            point_x += int(POINT_SPEED * dx / dist)
            point_y += int(POINT_SPEED * dy / dist)

        cv2.circle(frame, (point_x, point_y), POINT_RADIUS, POINT_COLOR, -1)

        # --- MediaPipe ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            face_landmarks = results.multi_face_landmarks[0]

            lx, ly = eye_center(face_landmarks.landmark, LEFT_EYE, w, h)
            rx, ry = eye_center(face_landmarks.landmark, RIGHT_EYE, w, h)

            dx, dy = rx - lx, ry - ly
            eye_dist = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))

            cx, cy = (lx + rx)/2, (ly + ry)/2

            lh = abs(max([face_landmarks.landmark[i].y*h for i in LEFT_EYE]) -
                     min([face_landmarks.landmark[i].y*h for i in LEFT_EYE]))
            rh = abs(max([face_landmarks.landmark[i].y*h for i in RIGHT_EYE]) -
                     min([face_landmarks.landmark[i].y*h for i in RIGHT_EYE]))
            avg_h = (lh + rh)/2

            box_w = eye_dist * 2.0
            box_h = avg_h * 1.5

            rect = ((cx, cy), (box_w, box_h), angle)
            crop, x_min, y_min, x_max, y_max = crop_eyes(frame, rect)

            if crop.size > 0:
                filename = f"frame_{frame_count:05d}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, crop)

                rel_x_min = x_min / screen_w
                rel_y_min = y_min / screen_h
                rel_x_max = x_max / screen_w
                rel_y_max = y_max / screen_h
                rel_point_x = point_x / screen_w
                rel_point_y = point_y / screen_h
                csv_writer.writerow([filename, rel_x_min, rel_y_min, rel_x_max, rel_y_max, rel_point_x, rel_point_y])

                frame_count += 1

        cv2.namedWindow("Gaze Collection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Gaze Collection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gaze Collection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
