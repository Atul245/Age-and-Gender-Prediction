
# Real-Time Age and Gender Prediction

A computer vision project that uses a pre-trained deep learning model to detect faces in an image or live webcam feed and predict the age and gender of each person.

---

## 📋 About The Project

This project leverages the power of Convolutional Neural Networks (CNNs) to perform real-time facial analysis. It detects human faces from a camera feed, crops the facial region, and passes it to a pre-trained model that classifies the person's gender (Male or Female) and predicts their age.

The model architecture was inspired by the research paper:  
**"Age and Gender Classification using Convolutional Neural Networks"** by Gil Levi and Tal Hassner.

---

## ✨ Features

- ✅ **Real-Time Detection**: Analyzes video streams from a webcam and performs predictions on the fly.
- ✅ **Gender Classification**: Classifies faces as 'Male' or 'Female'.
- ✅ **Age Prediction**: Estimates the person's age.
- ✅ **OpenCV Integration**: Uses OpenCV for efficient image and video processing.

---

## 🛠️ Technologies & Models Used

- **Python 3.x**
- **OpenCV (cv2)**: For video capture, face detection, and drawing annotations.
- **Keras (with TensorFlow backend)**: For loading and running the pre-trained deep learning model.
- **NumPy**: For numerical operations and image manipulation.

### Pre-trained Models

- **Face Detection**:  
  OpenCV's Caffe-based model (`deploy.prototxt.txt` and `res10_300x300_ssd_iter_140000.caffemodel`) is used for face localization.

- **Age & Gender Classification**:  
  A custom Keras model (`age_gender_prediction.h5`) trained specifically for this task.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x installed
- pip package manager installed

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Atul245/Age-and-Gender-Prediction.git
    cd Age-and-Gender-Prediction
    ```

2. Install required packages:
    ```bash
    pip install opencv-python numpy keras tensorflow
    ```

---

## 📖 Usage

To run the application:

```bash
python age_gender_prediction.py
```

- A window will appear displaying your webcam feed.
- The program will draw boxes around detected faces.
- Predicted gender and age will appear above each detected face.
- Press 'q' to close the window and stop the application.

---

## 📊 Dataset

This project uses a pre-trained model.  
The training data is not specified, but such models are typically trained on large public datasets like the **UTKFace Dataset**.  
You can explore the dataset here:  
[UTKFace Dataset](https://susanqq.github.io/UTKFace/)

---

## 🙏 Acknowledgments

- The age and gender model architecture is based on the work by **Gil Levi and Tal Hassner**.
- Face detection uses **OpenCV's Deep Neural Network (DNN)** module.

---

## 📄 License

This project is open source and available under the MIT License.
