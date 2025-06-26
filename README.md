🍅 Tomato Leaf Disease Detection Using CNN Models
This project focuses on detecting and classifying tomato leaf diseases using Convolutional Neural Networks (CNNs). It aims to assist farmers and agricultural experts in early identification of plant diseases to improve crop yield and reduce economic losses.

🚀 Project Objective
To build an accurate deep learning model using CNNs that can classify tomato leaf images into various disease categories or as healthy, enabling quick and reliable diagnosis.

🧠 Technologies Used
Python

TensorFlow / Keras

OpenCV

Matplotlib & Seaborn

NumPy & Pandas

📁 Dataset : https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf
The dataset contains labeled images of tomato leaves with multiple disease types including:

Tomato Bacterial Spot

Tomato Early Blight

Tomato Late Blight

Tomato Leaf Mold

Tomato Septoria Leaf Spot

Tomato Target Spot

Tomato Yellow Leaf Curl Virus

Tomato Mosaic Virus

Healthy Tomato Leaves

(Source: PlantVillage Dataset)

🛠️ Model Overview
Preprocessing: Resizing, normalization, and augmentation

CNN architecture: Multiple convolutional and pooling layers, followed by fully connected layers

Evaluation: Accuracy, precision, recall, and confusion matrix

📊 Results
The trained CNN model achieved high accuracy on both validation and test datasets, showcasing strong performance in disease detection.

📌 How to Use
Clone the repo

Install dependencies using pip install -r requirements.txt

Run model_train.py to train the model

Use predict.py to classify a new tomato leaf image

📝 Future Work
Deployment as a web/mobile application

Real-time detection using webcam input

Extension to other crops and diseases
