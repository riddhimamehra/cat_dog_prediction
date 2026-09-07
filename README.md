# Cat vs Dog Image Classification using MobileNetV2

A deep learning image classification project that uses **Transfer Learning with MobileNetV2** to classify images as either **Cat** or **Dog**.

## Project Overview

Instead of training a CNN completely from scratch, this project uses the **pre-trained MobileNetV2 architecture**, which was originally trained on the ImageNet dataset.

The pretrained convolutional base is initially frozen and used as a feature extractor. A custom classification head is then added for Cat vs Dog classification.

After the initial training, the MobileNetV2 model is fine-tuned by unfreezing layers and training with a very small learning rate.

## Live Demo

**[Cat vs Dog Classifier — Live Demo](https://catdogprediction-q5d44t3lbpggtbe6kdk7yj.streamlit.app/)**

## Model Architecture

The model consists of:

- MobileNetV2 pretrained on ImageNet
- Global Average Pooling
- Dense layer with 512 neurons
- Dropout (0.5)
- Output layer with Sigmoid activation

```text
Input Image (128 × 128 × 3)
          ↓
     MobileNetV2
   (Pretrained CNN)
          ↓
Global Average Pooling
          ↓
Dense (512, ReLU)
          ↓
Dropout (0.5)
          ↓
Dense (1, Sigmoid)
          ↓
    Cat / Dog
```

## Why MobileNetV2 / Transfer Learning?

The initial CNN approach was not performing reliably on individual images despite achieving around **95% accuracy** during evaluation. The main issue was that the dataset was relatively small, making it difficult for a CNN trained from scratch to learn robust visual features and generalize well to new images.

To address this, the project was switched to **Transfer Learning with MobileNetV2**. Since MobileNetV2 was already pretrained on the large ImageNet dataset, it provided useful image features that could be transferred to the Cat vs Dog classification task.

The MobileNetV2 base was initially frozen so that its pretrained weights were not changed during the first stage of training. The custom classification layers were then trained for the Cat vs Dog task.

The model was subsequently fine-tuned by unfreezing layers of MobileNetV2 while keeping the initial layers frozen. A very small learning rate (`1e-5`) was used during fine-tuning to allow the pretrained features to adapt to the new task.

This approach resulted in much more reliable image classification, with the final model achieving **100% accuracy on the 200-image test set**.

## Training

### Initial Training

- Epochs: 20
- Batch size: 32
- Optimizer: Adam
- Loss: Binary Crossentropy
- Validation split: 20%

### Fine-Tuning

- Epochs: 10
- Batch size: 32
- Optimizer: Adam
- Learning rate: `1e-5`
- Validation split: 20%

## Model Evaluation

The model was evaluated on a separate test set using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- Classification Report

### Test Results

**Accuracy: 100%**

Confusion Matrix:

```text
[[100   0]
 [  0 100]]
```

Classification Report:

```text
              precision    recall  f1-score   support

         Cat       1.00      1.00      1.00       100
         Dog       1.00      1.00      1.00       100

    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```

The model correctly classified all **200 test images**: 100 cats and 100 dogs.

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL

## Project Structure

```text
cat_dog_prediction/
│
├── cat_dog_prediction.ipynb
├── cat_dog_mobilenetv2.h5
└── README.md
```

## Key Concepts Learned

- Convolutional Neural Networks
- Transfer Learning
- MobileNetV2
- Pretrained Models
- Feature Extraction
- Fine-Tuning
- Image Preprocessing
- Binary Image Classification
- Confusion Matrix
- Precision, Recall and F1-score

## Future Improvements

- Add data augmentation
- Implement Early Stopping
- Experiment with different pretrained architectures
- Improve image preprocessing
- Deploy the model using Streamlit
