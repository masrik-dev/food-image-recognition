# **Currently working on it. Please ignore the current readme file.**

# 🍲 Food Image Recognition

**Food Image Recognition** is a deep learning project focused on classifying different types of food from photographs.  
It uses the Food-101 dataset and leverages EfficientNet and CNN architectures to accurately identify 101 food categories.

---

## 📂 Dataset
- **Food-101** dataset sourced from [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101)
- Includes **101 classes** and **~101,000 labeled images**

---

## ⚙️ Features
- **Preprocessing & Augmentation**: Image resizing, normalization, and augmentation to enhance model generalization
- **Model Architectures**: EfficientNetB0 and EfficientNetB1 for feature extraction, CNNs built and trained from scratch
- **Training Enhancements**: Mixed precision training, model checkpoints, and early stopping to save and restore best models
- **Evaluation Tools**: Confusion matrix and accuracy plots to visualize performance
- **Prediction Examples**: Model predictions demonstrated on unseen images

---

## 📈 Results
- Achieved high accuracy across 101 food classes
- Visualized results using confusion matrix and accuracy/loss plots

---

## 🚀 Usage
1. Clone the repository  
   `git clone https://github.com/<your-username>/food-image-recognition.git`  
   `cd food-image-recognition`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Train the model  
   `python train.py`

4. Evaluate the model  
   `python evaluate.py`

5. Make predictions  
   `python predict.py --image path_to_image.jpg`

---

## 📌 Future Improvements
- Experiment with newer architectures (EfficientNetV2, Vision Transformers)
- Hyperparameter tuning for improved performance
- Deploy as a web application for real-time inference

---

## 📜 License
This project is licensed under the MIT License.
