# Robot-Face-Image-Processing-Deep-Learning-and-Emotional-Detection-Project
A real-time facial emotion recognition system using deep learning and image processing. Built with a modified VGG16 CNN model trained on the FER-2013 dataset, this project uses PyTorch and OpenCV to classify emotions (happy, sad, angry, surprised, neutral) from live webcam input.
📌 Project Overview
🎯 Goal: Accurately classify human facial emotions in real-time using a webcam stream.

📸 Input: Real-time grayscale video frames captured from webcam.

🧠 Model: Modified VGG16 CNN architecture trained on a cleaned version of the FER-2013 dataset.

😃 Output: One of 5 emotions: Happy, Sad, Neutral, Angry, or Surprised.

🛠 Features
✅ Real-time emotion detection using webcam

✅ CNN trained on 35,000+ labeled facial images

✅ Live face detection using Haar cascades

✅ Visual overlay of predicted emotion on video stream

✅ Robust performance with ~74% test accuracy and F1-score of 0.77

🧠 Model Architecture
Backbone: VGG16 (modified for 5-class emotion classification)

Preprocessing:

Convert to grayscale

Resize to 224x224

Normalize pixel values

Classifier:

Fully connected layers with ReLU and Dropout

Output layer with Softmax

Frameworks: PyTorch, OpenCV, Scikit-learn, Seaborn

📊 Performance
Metric	Value
Training Acc.	81.20%
Validation Acc.	~72.30%
Test Accuracy	74.05%
Weighted F1	0.77

The model performs best on happy (F1: 0.83) and surprised (F1: 0.78), with some confusion between neutral and sad expressions. See the confusion matrix and training graphs in /results.

🖥️ Live Demo Example
