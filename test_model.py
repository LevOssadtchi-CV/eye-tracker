import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import os

# --- Настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/gaze_best.pth"
POINT_COLOR = (0, 200, 0)  # Зеленая точка
POINT_RADIUS = 10
CALIBRATION_INSTRUCTION = "Look at the center of the screen and press ENTER to calibrate"
CALIBRATION_DONE = "Calibration complete! The direction of view is corrected."

# --- Определение модели (такая же, как в обучении) ---
class ComplexGazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]

# --- Функции из сбора датасета ---
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

# --- Загрузка модели ---
model = ComplexGazeNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Инициализация калибровки ---
calibrated = False
error_rel_x = 0.0
error_rel_y = 0.0

# --- Инициализация камеры ---
cap = cv2.VideoCapture(0)
screen_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x_rel = 0.5  # Центр экрана в относительных координатах
center_y_rel = 0.5

# --- MediaPipe Face Mesh ---
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Отразим по горизонтали для естественного отображения
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        pred_x_rel = 0.0
        pred_y_rel = 0.0
        point_x = 0
        point_y = 0

        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            for face_landmarks in results.multi_face_landmarks:
                # Центры глаз
                lx, ly = eye_center(face_landmarks.landmark, LEFT_EYE, w, h)
                rx, ry = eye_center(face_landmarks.landmark, RIGHT_EYE, w, h)

                # Вектор между глазами
                dx, dy = rx - lx, ry - ly
                eye_dist = np.hypot(dx, dy)
                angle = np.degrees(np.arctan2(dy, dx))

                # Центр глаз
                cx, cy = (lx + rx)/2, (ly + ry)/2

                # Высота глаз
                lh = abs(max([face_landmarks.landmark[i].y*h for i in LEFT_EYE]) -
                         min([face_landmarks.landmark[i].y*h for i in LEFT_EYE]))
                rh = abs(max([face_landmarks.landmark[i].y*h for i in RIGHT_EYE]) -
                         min([face_landmarks.landmark[i].y*h for i in RIGHT_EYE]))
                avg_h = (lh + rh)/2

                # Размеры прямоугольника
                box_w = eye_dist * 2.0
                box_h = avg_h * 1.5
                rect = ((cx, cy), (box_w, box_h), angle)

                # Обрезаем область глаз
                crop, x_min, y_min, x_max, y_max = crop_eyes(frame, rect)

                # Подготовка изображения для модели
                img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))  # Фиксированный размер, как при обучении
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                img_tensor = torch.tensor(img, dtype=torch.float).unsqueeze(0).to(DEVICE)

                # Предсказание
                with torch.no_grad():
                    pred = model(img_tensor)
                    pred_x_rel, pred_y_rel = pred[0].cpu().numpy()  # Относительные координаты

                # Применение калибровки, если выполнена
                if calibrated:
                    pred_x_rel = pred_x_rel + error_rel_x
                    pred_y_rel = pred_y_rel + error_rel_y

                # Преобразование в экранные координаты
                point_x = int(pred_x_rel * screen_w)
                point_y = int(pred_y_rel * screen_h)

                # Рисуем прямоугольник и точку
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Синий прямоугольник
                cv2.circle(frame, (point_x, point_y), POINT_RADIUS, POINT_COLOR, -1)  # Зеленая точка

                # Рисуем центр экрана для калибровки (красный крест)
                cv2.line(frame, (screen_w//2 - 10, screen_h//2), (screen_w//2 + 10, screen_h//2), (0, 0, 255), 2)
                cv2.line(frame, (screen_w//2, screen_h//2 - 10), (screen_w//2, screen_h//2 + 10), (0, 0, 255), 2)

        # Отображение инструкции
        if not calibrated:
            cv2.putText(frame, CALIBRATION_INSTRUCTION, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, CALIBRATION_DONE, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Показываем кадр
        cv2.imshow("Gaze Prediction with Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc для выхода
            break
        elif key == 13 and not calibrated and results.multi_face_landmarks:  # Enter для калибровки
            # Рассчитываем вектор ошибки (относительный): центр - предсказание
            error_rel_x = center_x_rel - pred_x_rel
            error_rel_y = center_y_rel - pred_y_rel
            calibrated = True
            print(f"Калибровка выполнена. Вектор ошибки: ({error_rel_x:.4f}, {error_rel_y:.4f})")

cap.release()
cv2.destroyAllWindows()
