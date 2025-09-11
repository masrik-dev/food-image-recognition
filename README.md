# 🍲 Food Image Recognition

Food Image Recognition is a deep learning project aimed at classifying types of food from photographs using the Food-101 dataset. It leverages EfficientNet and custom CNN architectures to achieve high accuracy across 101 distinct food categories.

## 📂 Dataset
* Food-101 Dataset sourced from Kaggle.
* Contains 101 classes and ~101,000 labeled images.

## ⚙️ Features
* Preprocessing & Augmentation: Resizing, normalization, image augmentation to improve generalization
* Model Architectures:
** EfficientNetB0 and EfficientNetB1 for feature extraction
** Custom CNN model trained from scratch
* Training Techniques:
** Mixed precision training for faster performance
** Early stopping and model checkpointing to save the best model
* Evaluation Tools:
** Confusion matrix and classification report
** Accuracy/loss plots over training epochs
* Prediction Examples on unseen test images

## 📈 Results
* Achieved high accuracy across multiple food categories
* Visualized performance using confusion matrix and accuracy plots

## 🚀 Usage

* Clone the repository:
`git clone https://github.com/<your-username>/food-image-recognition.git
cd food-image-recognition`

* Install dependencies:
`pip install -r requirements.txt`

*Train the model:
`python train.py`

* Evaluate model:
`python evaluate.py`

* Make predictions:
`python predict.py --image path_to_image.jpg`

## 📌 Future Improvements
* Experiment with newer architectures (e.g., EfficientNetV2, Vision Transformers)
* Hyperparameter tuning for improved accuracy
* Deploy as a web application

## 📜 License
This project is licensed under the MIT License.
