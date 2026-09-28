# Task 2 - Handwritten Digit Classification Using CNN

## Project Overview

This project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from images.

The model is trained using the MNIST handwritten digit dataset. It classifies input images into one of the ten digits from **0 to 9**.

The project also includes a separate prediction program that loads the saved trained model and predicts a digit from a new image provided by the user.

## Objective

* Build a CNN-based image classification model.
* Train the model on the MNIST dataset.
* Evaluate the model using test accuracy and classification metrics.
* Save the trained model.
* Use the saved model to predict digits from new images.

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Pillow

## Dataset

The project uses the **MNIST handwritten digit dataset**.

The dataset contains:

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* Grayscale images
* 10 classes: digits 0 to 9

The dataset is stored locally in:

```text
dataset/mnist.npz
```

## Project Structure

```text
task 2
│
├── dataset
│   └── mnist.npz
│
├── sample_images
│   ├── seven.png
│   └── six.png
│
├── mnist_digit_classifier.py
├── predict_digit.py
├── mnist_digit_model.keras
│
├── sample_digits.png
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── sample_predictions.png
│
├── mnist_digit_classifier.ipynb
├── requirements.txt
└── README.md
```

## CNN Architecture

The model consists of:

1. Input layer - 28 × 28 grayscale image
2. Convolutional layer - 32 filters
3. Max Pooling layer
4. Convolutional layer - 64 filters
5. Max Pooling layer
6. Flatten layer
7. Dense layer - 128 neurons
8. Dropout layer
9. Output layer - 10 classes

The output layer uses Softmax to calculate the probability of each digit.

## Preprocessing

The images are preprocessed before training:

* Pixel values are normalized from 0-255 to 0-1.
* A channel dimension is added for CNN input.
* The data is divided into training and testing sets.

## Model Training

Run:

```bash
python mnist_digit_classifier.py
```

The program:

1. Downloads the MNIST dataset if it is not already available.
2. Stores the dataset inside the `dataset` folder.
3. Preprocesses the images.
4. Builds the CNN model.
5. Trains the model.
6. Evaluates the model.
7. Generates graphs and evaluation results.
8. Saves the trained model as:

```text
mnist_digit_model.keras
```

## Generated Visualizations

The training program generates:

* `sample_digits.png` - Sample MNIST images
* `accuracy_curve.png` - Training and validation accuracy
* `loss_curve.png` - Training and validation loss
* `confusion_matrix.png` - Confusion matrix
* `sample_predictions.png` - Example predictions

## Predicting a New Image

After training, the saved model can be used to classify a new handwritten digit.

Run:

```bash
python predict_digit.py
```

The program asks for the image path:

```text
Enter the image path:
```

For example:

```text
sample_images/seven.png
```

The program loads:

```text
mnist_digit_model.keras
```

and predicts the digit.

Example output:

```text
Saved model loaded successfully!

-----------------------------
Prediction Result
-----------------------------
Predicted Digit : 7
Confidence      : 98.73%
-----------------------------
```

## Input Image Requirements

The prediction program works with common image formats such as:

* PNG
* JPG
* JPEG

The image should contain a single handwritten digit.

The program converts the image to grayscale, resizes it to 28 × 28 pixels, normalizes the pixel values, and sends it to the CNN model.

## Evaluation

The model is evaluated on the MNIST test dataset using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Training and validation curves are also generated to observe model performance during training.

## Saved Model

The trained CNN model is saved as:

```text
mnist_digit_model.keras
```

The saved model can be loaded later without retraining:

```python
from tensorflow.keras.models import load_model

model = load_model("mnist_digit_model.keras")
```

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

## How to Run

### Train the model

```bash
python mnist_digit_classifier.py
```

### Predict a new digit

```bash
python predict_digit.py
```

## Conclusion

A CNN-based handwritten digit classification system was developed using the MNIST dataset. The trained model learns visual patterns such as edges, curves, and shapes to classify handwritten digits from 0 to 9.

The trained model is saved and can be reused to predict digits from new input images without retraining the model.
