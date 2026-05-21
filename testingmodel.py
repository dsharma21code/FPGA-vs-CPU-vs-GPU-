#Testing the sample

sample = {
    "temperature": 70,
    "vibration": 0.60,
    "pressure": 128,
    "rpm": 1690,
    "current": 6.0,
    "voltage": 214,
    "noise": 68,
    "runtime_hours": 950,
    "load": 88,
    "humidity": 66,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}

predictions = inference_model.predict(input_dict)

predicted_class = tf.argmax(predictions[0]).numpy()

class_names = {
    0: "normal",
    1: "warning",
    2: "failure",
}

print("Prediction probabilities:", predictions[0])
print("Predicted class:", predicted_class)
print("Machine status:", class_names[predicted_class])
