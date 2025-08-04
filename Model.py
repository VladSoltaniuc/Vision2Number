import tensorflow as tf
from tensorflow.keras import layers, models

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load the MNIST dataset (handwritten digits)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Preprocess the images: normalize them to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into 1D arrays
    layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (for each digit)
])

# Compile the model with optimizer, loss, and metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_acc}")

# Save the model
model.save("mnist_model.h5")
