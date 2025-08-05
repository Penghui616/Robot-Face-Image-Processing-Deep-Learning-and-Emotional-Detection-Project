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

├── model/                     # Folder for storing .pth model

├── data/                      # Preprocessed data directory

├── results/                   # Accuracy, loss, confusion matrix (optional)

└── README.md
