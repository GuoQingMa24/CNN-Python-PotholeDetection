import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 设置图片尺寸
IMG_SIZE = 100

# 数据路径
TRAIN_PATH = "./train/"
TEST_PATH = "./test/"
DETECT_PATH = "./detect_images/"
MODEL_PATH = "./model/pothole_cnn_model.h5"


# ------------------- 数据加载 -------------------
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels


def load_data():
    # 加载训练数据
    pothole_images, pothole_labels = load_images_from_folder(TRAIN_PATH + "Pothole", label=1)
    plain_images, plain_labels = load_images_from_folder(TRAIN_PATH + "Plain", label=0)

    # 合并训练数据
    X = np.array(pothole_images + plain_images)
    y = np.array(pothole_labels + plain_labels)

    # 数据打乱
    X, y = shuffle(X, y)

    # 数据归一化
    X = X / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # (batch_size, height, width, channels)

    # 标签独热编码
    y = to_categorical(y, num_classes=2)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ------------------- CNN 模型定义 -------------------
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 输出层，2类
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ------------------- 训练模型 -------------------
def train_model():
    X_train, X_val, y_train, y_val = load_data()
    model = create_cnn_model()
    model.summary()

    # 训练模型
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10, batch_size=32)

    # 保存模型
    if not os.path.exists('./model'):
        os.makedirs('./model')
    model.save(MODEL_PATH)
    print("模型保存在:", MODEL_PATH)

    # 绘制训练曲线
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()


# ------------------- 预测自定义图片 -------------------
def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_names = ['Plain Road', 'Pothole']
    print(f"Image: {image_path} -> Predicted Class: {class_names[class_index]}")


# ------------------- 主函数 -------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("训练模型中...")
        train_model()
    else:
        print("加载已存在的模型中...")
        model = load_model(MODEL_PATH)

    # 检测自定义图片
    print("开始检测图片...")
    for filename in os.listdir(DETECT_PATH):
        image_path = os.path.join(DETECT_PATH, filename)
        predict_image(image_path, model)
