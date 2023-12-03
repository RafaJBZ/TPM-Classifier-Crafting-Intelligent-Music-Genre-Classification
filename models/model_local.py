import tensorflow as tf
from keras import layers
from keras import models
import numpy as np
import pandas as pd


labels = ["International", "Soul-RnB", "Instrumental", "Rock", "Jazz", "Folk", "Old-Time / Historic", "Blues",
          "Experimental", "Pop", "Electronic", "Hip-Hop", "Classical", "Spoken", "Country", "Easy Listening"]


input_shape = (25820,)
num_labels = len(labels)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Flatten(),
    layers.Reshape((20, 1291, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_labels, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


data = pd.read_parquet("./Data/1701184205.4851387.parquet")

data["vector"] = data["vector"].map(lambda x: x["values"])
data["top_genere"] = data["top_genere"].map(lambda x: labels.index(x))


x = np.asarray(data["vector"].to_list())
y = data["top_genere"].values

del data

print(x.shape)
print(y.shape)

model.fit(x, y, epochs=5)

model.save("./model.h5")

