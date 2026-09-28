import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -----------------------------
# Load the saved model
# -----------------------------
MODEL_PATH = "mnist_digit_model.keras"

model = load_model(MODEL_PATH)

print("Saved model loaded successfully!")

# -----------------------------
# Take image path from user
# -----------------------------
image_path = input("Enter the image path: ").strip()

if not os.path.exists(image_path):
    print("Image not found!")
    exit()

# -----------------------------
# Load and preprocess image
# -----------------------------
image = Image.open(image_path).convert("L")

# Resize to MNIST size
image = image.resize((28, 28))

# Convert to NumPy array
image_array = np.array(image)

# MNIST uses white digit on black background.
# If the input has a white background, invert it.
if np.mean(image_array) > 127:
    image_array = 255 - image_array

# Normalize pixel values
image_array = image_array.astype("float32") / 255.0

# Reshape for CNN
image_array = image_array.reshape(1, 28, 28, 1)

# -----------------------------
# Make prediction
# -----------------------------
prediction = model.predict(image_array, verbose=0)

predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

# -----------------------------
# Display result
# -----------------------------
print("\n-----------------------------")
print("Prediction Result")
print("-----------------------------")
print("Predicted Digit :", predicted_digit)
print("Confidence      : {:.2f}%".format(confidence))
print("-----------------------------")

# -----------------------------
# Display input image
# -----------------------------
plt.imshow(image_array[0].reshape(28, 28), cmap="gray")
plt.title(f"Predicted Digit: {predicted_digit}")
plt.axis("off")
plt.show()