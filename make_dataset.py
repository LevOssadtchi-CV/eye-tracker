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

# Цвет точки (приятный зеленый)
POINT_COLOR = (0, 200, 0)
POINT_RADIUS = 15
POINT_SPEED = 5  # пикселей за кадр

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

def eye_height(landmarks, idxs, h):
    ys = [landmarks[i].y * h for i in idxs]
    return max(ys) - min(ys)

def crop_eyes(frame, rect):
    # Поворачиваем изображение так, чтобы глаза были горизонтально
    (cx, cy), (w, h), angle = rect
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    # axis-aligned bbox
    x_min = int(cx - w/2)
    x_max = int(cx + w/2)
    y_min = int(cy - h/2)
    y_max = int(cy + h/2)
    crop = rotated[y_min:y_max, x_min:x_max]
    return crop, x_min, y_min, x_max, y_max

# --- Инициализация камеры ---
cap = cv2.VideoCapture(0)
screen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Начальное положение точки (центр экрана) ---
point_x = screen_w // 2
point_y = screen_h // 2
dir_x = random.choice([-1,1])
dir_y = random.choice([-1,1])

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # отразим по горизонтали
        frame = cv2.flip(frame, 1)

        # --- обновление позиции точки ---
        point_x += POINT_SPEED * dir_x
        point_y += POINT_SPEED * dir_y
        if point_x < POINT_RADIUS or point_x > screen_w - POINT_RADIUS:
            dir_x *= -1
        if point_y < POINT_RADIUS or point_y > screen_h - POINT_RADIUS:
            dir_y *= -1

        # --- рисуем точку ---
        cv2.circle(frame, (point_x, point_y), POINT_RADIUS, POINT_COLOR, -1)

        # --- обработка MediaPipe ---
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

                # центр глаз
                cx, cy = (lx + rx)/2, (ly + ry)/2

                # высота глаз
                lh = abs(max([face_landmarks.landmark[i].y*h for i in LEFT_EYE]) -
                         min([face_landmarks.landmark[i].y*h for i in LEFT_EYE]))
                rh = abs(max([face_landmarks.landmark[i].y*h for i in RIGHT_EYE]) -
                         min([face_landmarks.landmark[i].y*h for i in RIGHT_EYE]))
                avg_h = (lh + rh)/2

                # размеры прямоугольника
                box_w = eye_dist * 2.0
                box_h = avg_h * 1.5

                rect = ((cx, cy), (box_w, box_h), angle)
                crop, x_min, y_min, x_max, y_max = crop_eyes(frame, rect)

                # сохраняем кадр без ресайза
                filename = f"frame_{frame_count:05d}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(filepath, crop)

                # сохраняем относительные координаты и имя файла
                rel_x_min = x_min / screen_w
                rel_y_min = y_min / screen_h
                rel_x_max = x_max / screen_w
                rel_y_max = y_max / screen_h
                rel_point_x = point_x / screen_w
                rel_point_y = point_y / screen_h
                csv_writer.writerow([filename, rel_x_min, rel_y_min, rel_x_max, rel_y_max, rel_point_x, rel_point_y])

                frame_count += 1

        # --- показ видео на весь экран ---
        cv2.namedWindow("Gaze Collection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Gaze Collection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Gaze Collection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Esc
            break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
