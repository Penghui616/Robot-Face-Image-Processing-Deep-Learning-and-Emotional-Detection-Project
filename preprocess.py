import cv2
import os
import numpy as np
from PIL import Image

# 原始数据集路径
dataset_path = 'E:/archive'  # 替换为你的数据集路径
output_path = 'E:/preprocessed_vgg16'  # 处理后的数据存放路径

# VGG16 需要的图像参数
image_size = (224, 224)  # VGG16 需要 224x224
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# 确保文件夹存在
def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# 数据预处理
def preprocess_images():
    for folder_type in ['train', 'test']:
        input_folder = os.path.join(dataset_path, folder_type)
        output_folder = os.path.join(output_path, folder_type)
        
        for emotion_label in emotion_labels:
            input_emotion_folder = os.path.join(input_folder, emotion_label)
            output_emotion_folder = os.path.join(output_folder, emotion_label)

            ensure_folder(output_emotion_folder)  # 确保输出目录存在
            
            if not os.path.exists(input_emotion_folder):
                continue  # 如果文件夹不存在，跳过

            # 获取所有图片
            image_files = [f for f in os.listdir(input_emotion_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

            for idx, image_file in enumerate(image_files):
                input_image_path = os.path.join(input_emotion_folder, image_file)
                output_image_path = os.path.join(output_emotion_folder, f"{emotion_label}_{str(idx).zfill(6)}.jpg")

                # 读取图像
                img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图

                # 转换为 RGB（VGG16 需要 3 通道）
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # 调整大小为 224x224
                img_resized = cv2.resize(img_rgb, image_size)

                # 归一化到 [0, 255]（VGG16 需要）
                img_normalized = img_resized.astype('float32')

                # 保存处理后的图片
                Image.fromarray(np.uint8(img_normalized)).save(output_image_path)

if __name__ == "__main__":
    preprocess_images()
    print("✅ 数据预处理完成，已存入:", output_path)
