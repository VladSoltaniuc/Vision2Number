import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Normalize the images (as done during training)
x_test = x_test / 255.0

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h5')

# Choose a random image from the test set
random_index = np.random.randint(0, len(x_test))
img = x_test[random_index]
true_label = y_test[random_index]

# Reshape the image to match the model input
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Make the prediction
prediction = model.predict(img)
predicted_label = np.argmax(prediction)

# Plot the image and print the prediction result
plt.imshow(img[0], cmap='gray')
plt.title(f"Predicted: {predicted_label}, True: {true_label}")
plt.show()

# Print whether the prediction was correct or not
if predicted_label == true_label:
    print("Prediction is correct!")
else:
    print(f"Prediction is wrong. True label: {true_label}, Predicted: {predicted_label}")
