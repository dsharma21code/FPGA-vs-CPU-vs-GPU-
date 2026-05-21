#Preparing the data from CSV file

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import FeatureSpace
import pandas as pd

# Load the CSV file from GitHub

file_url = "https://raw.githubusercontent.com/dsharma21code/FPGA-vs-CPU-vs-GPU-/main/failure.csv"
dataframe = pd.read_csv(file_url)

print(dataframe.shape)

dataframe.head()

#randomizing 12 samples for training and 3 for validation

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("status")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)


train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

feature_space = FeatureSpace(
    features={
        "temperature": "float_normalized",
        "vibration": "float_normalized",
        "pressure": "float_normalized",
        "rpm": "float_normalized",
        "current": "float_normalized",
        "voltage": "float_normalized",
        "noise": "float_normalized",
        "runtime_hours": "float_normalized",
        "load": "float_normalized",
        "humidity": "float_normalized",
    },
    output_mode="concat",
)

#Specify featurespace
#Status not included in featurespace - label column

feature_space = FeatureSpace(
    features={
        "temperature": FeatureSpace.float_normalized(),
        "vibration": FeatureSpace.float_normalized(),
        "pressure": FeatureSpace.float_normalized(),
        "rpm": FeatureSpace.float_normalized(),
        "current": FeatureSpace.float_normalized(),
        "voltage": FeatureSpace.float_normalized(),
        "noise": FeatureSpace.float_normalized(),
        "runtime_hours": FeatureSpace.float_normalized(),
        "load": FeatureSpace.float_normalized(),
        "humidity": FeatureSpace.float_normalized(),
    },
    output_mode="concat",
)

#Training data and Shape established

train_ds_with_no_labels = train_ds.map(lambda x, _: x)
feature_space.adapt(train_ds_with_no_labels)

for x, _ in train_ds.take(1):
    preprocessed_x = feature_space(x)
    print("preprocessed_x.shape:", preprocessed_x.shape)
    print("preprocessed_x.dtype:", preprocessed_x.dtype)

#Training and validation set established

preprocessed_train_ds = train_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_train_ds = preprocessed_train_ds.prefetch(tf.data.AUTOTUNE)

preprocessed_val_ds = val_ds.map(
    lambda x, y: (feature_space(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
preprocessed_val_ds = preprocessed_val_ds.prefetch(tf.data.AUTOTUNE)

