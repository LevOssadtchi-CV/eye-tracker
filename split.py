import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Настройки ---
INPUT_DIR = "eyes_dataset"         # твоя папка с 6000 картинок
CSV_FILE = os.path.join(INPUT_DIR, "metadata.csv")
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Создаем папки ---
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# --- Загружаем CSV ---
df = pd.read_csv(CSV_FILE)

# --- Перемешиваем ---
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- Делим на train и temp (val+test) ---
train_df, temp_df = train_test_split(df, test_size=(1-TRAIN_RATIO), random_state=42)

# --- Делим temp на val и test ---
val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
val_df, test_df = train_test_split(temp_df, test_size=(1-val_size), random_state=42)

# --- Функция для копирования файлов ---
def copy_files(df_split, split_name):
    for _, row in df_split.iterrows():
        src = os.path.join(INPUT_DIR, row['filename'])
        dst = os.path.join(OUTPUT_DIR, split_name, row['filename'])
        shutil.copy2(src, dst)
    # Сохраняем CSV для этой части
    df_split.to_csv(os.path.join(OUTPUT_DIR, f"{split_name}_metadata.csv"), index=False)

# --- Копируем файлы ---
copy_files(train_df, "train")
copy_files(val_df, "val")
copy_files(test_df, "test")

print("Датасет успешно разделён на train/val/test!")

