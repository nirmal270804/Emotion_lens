# EmotionLens: Multimodal Analysis for Comprehensive Emotion Recognition

EmotionLens is a real-time facial emotion recognition system developed as part of an academic capstone project under the Bachelor of Computer Applications (BCA) program at Nitte (Deemed to be University). The system combines classical and modern AI techniques—including CNN, SVM, and Transformers—to analyze facial expressions and recognize human emotions accurately.

---

## 📌 Project Overview

EmotionLens aims to detect and classify human emotions from facial expressions using:
- Convolutional Neural Networks (CNN)
- Support Vector Machines (SVM) with HOG features
- Transformer-based deep learning models

The system performs real-time inference from webcam input and provides on-screen feedback, enabling use in interactive and human-centric applications.

---

## 🎯 Objectives

- Train and compare multiple machine learning models (CNN, SVM, Transformer).
- Build a consistent image preprocessing pipeline.
- Enable real-time emotion detection using OpenCV.
- Evaluate each model's accuracy, loss, and real-time performance.
- Maintain modular and extensible code for future enhancements.
- Ensure the system performs under varied real-world conditions.

---

## 🔬 Dataset

- **FER-2013 Dataset**  
  Contains 35,000+ grayscale images (48×48), labeled into 7 emotions:  
  `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`.

---

## 📈 Model Performance

| Model        | Accuracy (%) | Test Loss | Remarks                  |
|--------------|--------------|-----------|---------------------------|
| CNN          | 52.33        | 3.4283    | Best performer overall    |
| SVM + HOG    | 52.08        | 1.2887    | Efficient but limited     |
| Transformer  | 24.63        | 0.2463    | Needs further tuning      |

---

## 🧠 Key Features

- **Real-time Webcam Input**: Live emotion prediction using OpenCV.
- **Comparative Evaluation**: In-depth analysis of classical and deep learning models.
- **Emotion Feedback Overlay**: Emotion labels are displayed on screen with bounding boxes.
- **Custom Preprocessing Pipeline**:
  - Grayscale conversion
  - Face detection (Haar cascades)
  - Normalization and resizing
  - Augmentation for training

---

## 🔧 Tech Stack

- **Language**: Python 3.x  
- **Libraries**:  
  - `TensorFlow/Keras` – For CNN & Transformer models  
  - `scikit-learn` – For SVM and HOG feature extraction  
  - `OpenCV` – For face detection and real-time video capture  
  - `NumPy`, `Matplotlib`, `Seaborn` – For data handling and visualization  
- **Development Environment**:  
  - Jupyter Notebook  
  - Google Colab  
  - VS Code  

---

## 🖥️ Hardware Requirements

- CPU: Intel i5/i7 or AMD Ryzen  
- GPU: NVIDIA GTX 1660 / RTX 3060 (for training and real-time inference)  
- RAM: 16 GB or more  
- Webcam: Basic HD camera (720p+)  

---

## 🛠️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-repo/emotionlens.git
   cd emotionlens
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run real-time demo**
   ```bash
   python src/realtime_demo.py --model cnn
   ```

---

## 🌍 Future Scope

- Integrate advanced face detectors (Dlib, MTSVM).
- Enhance transformer accuracy with larger datasets (RAF-DB, AffectNet).
- Mobile and edge deployment using model compression.
- Handle multi-user detection in group settings.
- Adaptation to environmental conditions (light, occlusion, mask-wearing).

---

## 📚 References

- FER-2013 Dataset
- [Survey Articles on Emotion Recognition](https://www.mdpi.com/1424-8220/24/11/3484)
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Scikit-learn](https://scikit-learn.org/)
- Research papers listed in the literature survey section of the project

---

## 👨‍🎓 Developed By

**Nirmal S Nair**  
**Joel Abraham Bijoy**  
**Varun Krishna M**  
Nitte Institute of Professional Education  
Nitte (Deemed to be University)
