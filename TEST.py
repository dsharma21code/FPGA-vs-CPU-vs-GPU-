import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)

# Load MNIST handwritten digit dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values from 0–255 to 0–1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Basic neural network
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(x_train, y_train, epochs=1, batch_size=32)

# Test model
loss, acc = model.evaluate(x_test, y_test)
print("Test accuracy:", acc)

# Save model
model.save("mnist_mlp.keras")
