import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# ✅ 表情类别（5分类）
classes = ['angry', 'happy', 'neutral', 'sad', 'surprised']


# ✅ 图像预处理（和训练时一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ✅ 定义模型结构（5类）
class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes=5):
        super(ModifiedVGG16, self).__init__()
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        for param in self.vgg16.features[-8:].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.vgg16(x)

# ✅ 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModifiedVGG16(num_classes=5).to(device)
model.load_state_dict(torch.load("E:/best_model_5class.pth", map_location=device))
model.eval()

# ✅ 摄像头实时检测 + 情绪识别
def realtime_emotion_detection():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_pil = Image.fromarray(face_gray).convert("RGB")
            input_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                emotion = classes[predicted.item()]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Real-Time Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ✅ 启动摄像头实时检测
realtime_emotion_detection()
