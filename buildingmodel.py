# Building model

dict_inputs = feature_space.get_inputs()
encoded_features = feature_space.get_encoded_features()

x = keras.layers.Dense(4, activation="relu")(encoded_features)

predictions = keras.layers.Dense(3, activation="softmax")(x)

training_model = keras.Model(inputs=encoded_features, outputs=predictions)

training_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

inference_model = keras.Model(inputs=dict_inputs, outputs=predictions)
