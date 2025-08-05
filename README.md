# Robot-Face-Image-Processing-Deep-Learning-and-Emotional-Detection-Project
A real-time facial emotion recognition system using deep learning and image processing. Built with a modified VGG16 CNN model trained on the FER-2013 dataset, this project uses PyTorch and OpenCV to classify emotions (happy, sad, angry, surprised, neutral) from live webcam input.

1. Project Summary
   
Built with PyTorch and OpenCV for real-time facial emotion classification.

Modified VGG16 backbone with custom classifier layers.

Trained on preprocessed FER-2013 images, converted to RGB and resized to 224x224.

Final model achieves 74.05% test accuracy and F1-score of 0.77.

Includes a complete data pipeline, training script, evaluation module, and real-time inference demo.

2. Directory Structure


├── preprocess.py              # Preprocessing FER-2013 dataset

├── train_5class.py            # Model training script

├── test_5class.py             # Evaluation on test set

├── real_time_5class.py        # Real-time webcam inference

├── best_model_5class.pth

└── README.md

3. Installation

You also need to download and preprocess the FER-2013 dataset (see preprocess.py), and place it in the expected data/ directory structure
Links：https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

4. Model Architecture
Backbone: Pretrained VGG16 (torchvision.models)

Classifier: Replaced with:

Linear(25088 → 4096) → ReLU → Dropout(0.5)

Linear(4096 → 1024) → ReLU → Dropout(0.5)

Linear(1024 → 5)

Last 8 convolutional layers are optionally fine-tuned.

5. How to Use

5.1 Preprocess Dataset

python preprocess.py

This script converts grayscale images to RGB, resizes to 224×224, and saves them in the correct structure.

5.2 Train the Model

python train_5class.py

Uses 80% training, 20% validation split.

Trains for 10 epochs using Adam optimizer and CrossEntropyLoss.

Saves best model as best_model_5class.pth.

5.3 Evaluate on Test Set

python test_5class.py

Outputs test accuracy, classification report, and confusion matrix.

5.4 Real-Time Emotion Detection

python real_time_5class.py

Opens webcam, detects face, predicts emotion, and displays it in real time using OpenCV.

Press q to exit.

6. Results
Metric	Value
Train Accuracy	81.20%
Validation Acc.	~72.3%
Test Accuracy	74.05%
Weighted F1-Score	0.77

Model performs best on happy (F1: 0.83) and surprised (F1: 0.78), while neutral and sad are occasionally confused.

7. Dependencies
Python 3.8+

PyTorch

torchvision

OpenCV

scikit-learn

matplotlib

seaborn

Pillow


8. Model Weights
   
To run real-time detection, download the trained model:

best_model_5class.pth

This is my trained model links:https://drive.google.com/file/d/1sTD86qeprFU4XP6st91Zn1BvrB9SciJq/view?usp=drive_link

9.Result

<img width="868" height="498" alt="屏幕截图 2025-04-19 142021" src="https://github.com/user-attachments/assets/82397862-f354-4a0e-80d8-cb45cb08677d" />


The accuracy rate increased from 59% to 74%.


<img width="1242" height="470" alt="image" src="https://github.com/user-attachments/assets/bd5e2941-6778-43f1-b0ed-6b9b7909c16f" />





9. Model Weights
To run real-time detection, download the trained model:

best_model_5class.pth




