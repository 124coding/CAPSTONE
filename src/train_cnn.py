import os
import numpy as np
import tensorflow as tf
from tf.keras.models import Model
from tf.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten,
    Dense, Input
)
from tf.keras.optimizers import Nadam
from tf.keras.callbacks import (
    ReduceLROnPlateau, ModelCheckpoint,
    CSVLogger, EarlyStopping
)
from tf.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

# --- 설정 ---
DATA_DIR        = "ppo_data"
MODEL_DIR       = "models"
MODEL_NAME      = "cnn_angle_predictor.h5"
BATCH_SIZE      = 32
EPOCHS          = 100
ROI_SIZE        = (59, 255)      # (height, width)
VALID_SPLIT     = 0.2
SEED            = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# --- 데이터 로딩 & 전처리 함수 ---
def load_data_paths(data_dir):
    """이미지 파일 경로와 라벨(.txt) 경로 리스트로 반환"""
    img_paths, labels = [], []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(".png"):
            img_paths.append(os.path.join(data_dir, fname))
            labels.append(
                float(fname.replace(".png", ".txt"))
            )
    return img_paths, np.array(labels, dtype=np.float32)

def preprocess_image(path, label):
    """tf.data 맵 함수: 읽어서 전처리"""
    # 1) 파일 읽고 BGR→RGB
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.cast(img, tf.uint8)
    img = tf.image.resize(img, ROI_SIZE)
    img = tf.reverse(img, axis=[-1])  # BGR->RGB
    # 2) 정규화
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# --- 경로 & 라벨 ---
img_paths, labels = load_data_paths(DATA_DIR)
train_paths, val_paths, y_train, y_val = train_test_split(
    img_paths, labels,
    test_size=VALID_SPLIT,
    random_state=SEED
)

# --- tf.data.Dataset 생성 ---
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train))
val_ds   = tf.data.Dataset.from_tensor_slices((val_paths,   y_val))

train_ds = (train_ds
    .shuffle(len(train_paths), seed=SEED)
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (val_ds
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# --- 모델 정의 ---
inp = Input(shape=(*ROI_SIZE, 3), name="image_input")
x = Conv2D(16, 3, activation='relu', padding='same')(inp)
x = MaxPooling2D()(x)
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(10, activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(1, name='angle')(x)

model = Model(inp, out)
model.compile(
    optimizer=Nadam(learning_rate=1e-4),
    loss='mse',
    metrics=['mae']
)

# --- 콜백 설정 ---
checkpoint_path = os.path.join(
    MODEL_DIR, "cnn_best.h5"
)
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        checkpoint_path, monitor='val_loss',
        save_best_only=True, verbose=1
    ),
    CSVLogger("cnn_training_log.csv"),
    EarlyStopping(
        monitor='val_loss', patience=15,
        verbose=1, restore_best_weights=True
    )
]

# --- 학습 ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# --- 최종 모델 저장 ---
model.save(os.path.join(MODEL_DIR, MODEL_NAME))
print(f"[INFO] CNN 모델 저장 완료: {MODEL_DIR}/{MODEL_NAME}")
