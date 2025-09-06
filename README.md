Real-Time Age and Gender Prediction
A computer vision project that uses a pre-trained deep learning model to detect faces in an image or live webcam feed and predict the age and gender of each person.

!

📋 About The Project
This project leverages the power of Convolutional Neural Networks (CNNs) to perform real-time facial analysis. It detects human faces from a camera feed, crops the facial region, and passes it to a pre-trained model that classifies the person's gender (Male or Female) and predicts their age.

The model architecture was inspired by the research paper "Age and Gender Classification using Convolutional Neural Networks" by Gil Levi and Tal Hassner.

✨ Features
Real-Time Detection: Analyzes video streams from a webcam to perform predictions on the fly.

Gender Classification: Accurately classifies faces into 'Male' or 'Female'.

Age Prediction: Estimates the age of the person.

OpenCV Integration: Uses OpenCV for efficient image and video processing.

🛠️ Technologies & Models Used
Python 3.x

OpenCV (cv2): For capturing video, detecting faces, and drawing results on the screen.

Keras (with TensorFlow backend): For loading the pre-trained deep learning model.

NumPy: For numerical operations and image manipulation.

Pre-trained Models
Face Detection: A pre-trained Caffe model (deploy.prototxt.txt and res10_300x300_ssd_iter_140000.caffemodel) is used for localizing faces in the input frame.

Age & Gender Classification: A custom Keras model (age_gender_prediction.h5) trained for this specific task.

🚀 Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
You need to have Python and pip installed on your system.

Installation
Clone the repository:

git clone [https://github.com/Atul245/Age-and-Gender-Prediction.git](https://github.com/Atul245/Age-and-Gender-Prediction.git)
cd Age-and-Gender-Prediction

Install the required packages:

pip install opencv-python numpy keras tensorflow

📖 Usage
To run the application, simply execute the main Python script. The script will automatically launch your webcam and start the prediction process.

python age_gender_prediction.py

A window will appear showing your webcam feed.

The program will draw boxes around detected faces.

The predicted gender and age will be displayed above each box.

Press 'q' to close the application window and stop the script.

📊 Dataset
This project uses a pre-trained model. While the exact training data is not specified, models of this nature are typically trained on large, publicly available datasets of faces with age and gender labels, such as the UTKFace Dataset. You can find it here.

🙏 Acknowledgments
The model architecture is based on the work by Gil Levi and Tal Hassner.

The face detection model is provided by OpenCV's Deep Neural Network (DNN) module.
