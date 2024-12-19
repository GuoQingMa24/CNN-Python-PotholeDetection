import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 设置参数
MODEL_PATH = "./model/pothole_cnn_model.h5"  # 训练好的模型路径
DETECT_PATH = "./detect_images"  # 检测图像路径
IMG_SIZE = 100  # 模型输入的图片尺寸
WINDOW_SIZE = 100  # 滑动窗口尺寸
STRIDE = 20  # 滑动窗口的步长

# ------------------- 加载训练好的模型 -------------------
model = load_model(MODEL_PATH)

# ------------------- 滑动窗口检测 -------------------
def sliding_window_detection(image, model, window_size, stride):
    """
    使用滑动窗口检测坑洼区域。
    Args:
        image: 输入的完整图像。
        model: 已加载的 CNN 模型。
        window_size: 滑动窗口的尺寸 (宽, 高)。
        stride: 滑动窗口的步长。
    Returns:
        坑洼区域的坐标列表 [(x1, y1, x2, y2), ...]。
    """
    pothole_boxes = []  # 保存检测到的坑洼位置
    img_height, img_width = image.shape[:2]

    # 遍历图像中的每一个窗口
    for y in range(0, img_height - window_size, stride):
        for x in range(0, img_width - window_size, stride):
            # 提取窗口区域
            window = image[y:y+window_size, x:x+window_size]
            resized_window = cv2.resize(window, (IMG_SIZE, IMG_SIZE)) / 255.0  # 归一化
            resized_window = resized_window.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # 模型预测
            prediction = model.predict(resized_window, verbose=0)
            class_index = np.argmax(prediction)

            # 如果预测为坑洼，保存坐标
            if class_index == 1:  # 1表示坑洼
                pothole_boxes.append((x, y, x + window_size, y + window_size))

    return pothole_boxes


# ------------------- 检测图片并显示坑洼 -------------------
def detect_and_display_potholes(image_path, model):
    """
    检测图片中的坑洼并实时显示带标记的结果。
    Args:
        image_path: 输入图片路径。
        model: 训练好的模型。
    """
    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image: {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用滑动窗口检测坑洼区域
    pothole_boxes = sliding_window_detection(gray_image, model, WINDOW_SIZE, STRIDE)

    # 在原图上绘制检测框
    for box in pothole_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框表示坑洼

    # 显示结果图片
    cv2.imshow("Pothole Detection Result", image)
    cv2.waitKey(0)  # 等待用户按下键
    cv2.destroyAllWindows()


# ------------------- 主函数 -------------------
if __name__ == "__main__":
    print("Starting pothole detection...")

    # 遍历检测图像文件夹中的所有图片
    for filename in os.listdir(DETECT_PATH):
        image_path = os.path.join(DETECT_PATH, filename)
        print(f"Processing {filename}...")
        detect_and_display_potholes(image_path, model)

    print("Pothole detection completed.")
