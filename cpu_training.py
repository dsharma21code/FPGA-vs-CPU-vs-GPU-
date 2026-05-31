import time
import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Fake machine sensor data
X = np.random.rand(5000, 8).astype("float32")
y = np.random.randint(0, 2, 5000).astype("float32")

with tf.device("/GPU:0"):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

print("\nTraining on GPU...")
train_start = time.perf_counter()
model.fit(X, y, epochs=10, batch_size=32, verbose=1)
train_end = time.perf_counter()

print("\nBenchmarking GPU inference...")

# Warmup
model.predict(X[:100], verbose=0)

runs = 100
start = time.perf_counter()

for _ in range(runs):
    model.predict(X[:100], verbose=0)

end = time.perf_counter()

training_time = train_end - train_start
total_inference_time = end - start
avg_inference_time = total_inference_time / runs

print("\n===== GPU RESULTS =====")
print(f"Training time: {training_time:.4f} seconds")
print(f"Total inference time for {runs} runs: {total_inference_time:.4f} seconds")
print(f"Average inference time per 100 samples: {avg_inference_time:.6f} seconds")
print(f"Average inference time per sample: {avg_inference_time / 100:.8f} seconds")
