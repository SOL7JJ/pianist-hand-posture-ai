# pianist-hand-posture-ai
AI-powered computer vision system for detecting correct and incorrect pianist hand postures to support music education.
# AI-Powered Pianist Hand Posture Recognition 🎹🤖

## Overview

This project presents an **end-to-end machine learning system** for detecting **correct and incorrect pianist hand postures** from images using computer vision. The goal is to support **music education** by providing automated, objective feedback on hand posture ,  a critical but difficult-to-monitor aspect of piano technique.

The project was originally developed as part of a **Master’s degree in Artificial Intelligence** and has since been **refactored, extended, and engineered** to meet **industry ML Engineer standards**, including clean project structure, reproducibility, and deployable inference.

## Problem Statement

Poor hand posture can lead to:

* Reduced technical efficiency
* Increased injury risk
* Slower musical development

Traditional feedback relies on constant supervision by a teacher. This project explores how **machine learning and transfer learning** can assist by automatically classifying pianist hand posture as:

* ✅ Correct
* ❌ Incorrect

from static images captured during practice.

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **NumPy & pandas**
* **Google Colab** (training)
* **Streamlit** (deployment demo)

## Dataset

* Custom, self-collected image dataset of pianist hands
* Binary classification: *correct* vs *incorrect* posture
* Approximately **1,000+ images** (balanced classes)
* Images captured under varying:

  * Angles
  * Lighting conditions
  * Hand shapes and sizes

### Data Handling

* Images resized to **224×224**
* Normalisation applied
* **Data augmentation** used to improve generalisation:

  * Small rotations
  * Zoom
  * Brightness/contrast variation

> For privacy and licensing reasons, only a **small sample dataset** is included in this repository.


## Model Architecture

* **Transfer learning** using **MobileNetV2** pre-trained on ImageNet
* Frozen convolutional base during initial training
* Custom classification head
* Fine-tuning of upper layers

### Why MobileNetV2?

* Lightweight and efficient
* Strong performance on small–medium datasets
* Well-suited for real-time inference and deployment

---

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

This ensures balanced performance across both posture classes and avoids misleading results due to class imbalance.


## Explainability

To improve interpretability, **Grad-CAM** is used to visualise which regions of the image the model focuses on when making predictions. This is especially important in an educational context, where explainability builds trust with learners and teachers.


## Project Structure

pianist-hand-posture-ai/
│
├── data/
│   └── sample/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── inference/
│   └── predictor.py
│
├── app/
│   └── app.py
│
├── models/
│   └── best_model.h5
│
├── requirements.txt
├── README.md
└── LICENSE

## Demo Application

A **Streamlit web application** allows users to:

* Upload an image of a pianist’s hand
* Receive a posture classification
* View prediction confidence
* See Grad-CAM visual explanations

> Deployment instructions and demo screenshots will be added.

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Future Work

* Video-based hand posture recognition
* Real-time webcam feedback
* Multi-class classification of specific posture errors
* Integration into digital music learning platforms

---

## Author

**Jonathan James (Solis James)**
MSc Artificial Intelligence | Machine Learning Engineer (aspiring)

---

## License

This project is released under the MIT License.
