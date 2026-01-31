# Food-101 Image Classification using KNN
------------------------------------------

Team Members
---------------
* 1.Divya Jayaprakash 
* 2.Jayasri Dhanapal 
* 3.Reshma Karthikeyan Nair
* 4.MODI Gurpreetkaur Jaykumar
* 5.TARIGOPULA Abhilash

## Project Overview
--------------------

This project implements an image classification system using computer vision techniques to identify different food categories from images. The system uses deep learning for feature extraction and a classical machine learning algorithm for classification.

The goal of this project is to demonstrate how pre-trained convolutional neural networks can be combined with traditional classifiers to solve real-world image classification problems.


## 📖 Problem Statement
-------------------------
Manual identification of food items from images is time-consuming and error-prone. An automated food image classification system can help in applications such as diet tracking, restaurant recommendation systems, and smart food apps.

---

## Dataset Description
-----------------------
* **Dataset Name:** Food-101
* **Number of Classes:** 101 food categories
* **Description:**
  The Food-101 dataset contains real-world food images with variations in lighting, background, scale, and viewpoint. Each class has multiple images, making the dataset suitable for training and evaluating image classification models.

---
## Food-101 structure:
images/
   pizza/
   sushi/
   burger/
meta/
   train.txt
   test.txt

## Technologies & Tools Used
-----------------------------
* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

---

## Methodology
----------------
The project follows the below workflow:

1. **Image Preprocessing**

   * Image resizing
   * Normalization

2. **Feature Extraction**

   * Pre-trained MobileNetV2 is used to extract deep feature vectors from images

3. **Classification**

   * K-Nearest Neighbors (KNN) algorithm is applied on extracted features

4. **Evaluation**

   * Model performance is evaluated using accuracy
   * Visualization using confusion matrix and PCA plots

---

## Installation & Requirements
-------------------------------
Install the required Python libraries using the command below:

```bash
pip install numpy opencv-python tensorflow scikit-learn matplotlib
```

---

## How to Run the Project
---------------------------
1. Download or clone the project repository
2. Open the Jupyter Notebook file
3. Run each cell step by step
4. The model will process images, train the classifier, and display results

---

## Results
------------
* The KNN classifier provides reasonable classification accuracy when applied to deep features
* PCA visualization helps understand feature separability
* Confusion matrix highlights correct and incorrect predictions

---

## Applications
----------------
* Automated food recognition systems
* Diet and calorie monitoring applications
* Restaurant menu digitization
* Learning and academic computer vision projects

---

## Future Scope
----------------
* Use advanced classifiers such as SVM or Random Forest
* Implement end-to-end deep learning models
* Apply data augmentation for better accuracy
* Deploy the model as a web or mobile application

---

## Conclusion
---------------
This project demonstrates the effectiveness of combining deep feature extraction with a simple machine learning classifier. Using MobileNetV2 with KNN provides a practical and efficient solution for food image classification tasks.

---
