from pyspark.sql import SparkSession
import tensorflow as tf
from keras import layers
from keras import models
from pyspark.sql.functions import row_number, lit
from pyspark.sql.window import Window
import numpy as np
from pyspark.ml.functions import vector_to_array

spark_conf = {
    "spark.driver.cores": "1",
    "spark.driver.memory": "1g",
    "spark.executor.memory": "1g"
}

labels = ['Rock', 'Folk', 'Experimental', 'Pop', 'Hip-Hop', 'International']
class DatasetGenerator:
    def __init__(self, spark_df, buffer_size, x_col, y_col):
        window = Window().partitionBy(lit("a")).orderBy(lit("a"))
        self.df = (
            spark_df.withColumn("index", row_number().over(window) - 1)
            .sort("index")
            .select([x_col, "index", y_col])
        )
        self.x_col = x_col
        self.y_col = y_col
        self.buffer_size = buffer_size

    def generate_data(self):
        idx = 0
        buffer_counter = 0
        buffer = self.df.withColumn("vector", vector_to_array("vector")).filter(
            f"index >= {idx} and index < {self.buffer_size}"
        ).toPandas()
        while len(buffer) > 0:
            if idx < len(buffer):
                X = np.array(buffer.iloc[idx][self.x_col])  # can be converted to array...
                y = labels.index(buffer.iloc[idx][self.y_col])

                idx += 1
                yield X.reshape((-1, 1)), y
            else:
                buffer = self.df.filter(
                    f"index >= {buffer_counter * self.buffer_size} "
                    f"and index < {(buffer_counter + 1) * self.buffer_size}"
                ).toPandas()
                idx = 0
                buffer_counter += 1

def create_keras_model():
    input_shape = (25820,)
    num_labels = 8

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Reshape((20, 1291, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("Train model")
    .config(map=spark_conf)
    .getOrCreate()
)


sc = spark.sparkContext

sc.setLogLevel("ERROR")
data = spark.read.parquet('/home/rafajbz/data/fma_vectors/1701049528.353486.parquet')

batch_size = 10
buffer_size = 10 * 2

data_generator = DatasetGenerator(data, buffer_size, x_col="vector", y_col="top_genere")

dataset = tf.data.Dataset \
    .from_generator(data_generator.generate_data, output_types=(tf.float32, tf.int32)) \
    .batch(batch_size, drop_remainder=True) \

model = create_keras_model()

model.fit(dataset, epochs=2)

model.save("./model.h5")


