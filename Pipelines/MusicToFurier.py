from pyspark import RDD   ## rdd es una dataset, resiliente y distribuido
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.functions import array_to_vector
import pyspark
from sklearn.preprocessing import StandardScaler
import librosa
import io
import numpy as np

spark_conf = {
    "spark.driver.cores": "1",
    "spark.driver.memory": "7g",
    "spark.executor.memory": "1g"
}

spark = (
    SparkSession.builder
    .master("local[24]")
    .appName("Word Count")
    .config(map=spark_conf)
    .getOrCreate()
)
sc = pyspark.SparkContext.getOrCreate()
binary_wave_rdd = sc.binaryFiles('hdfs://localhost:9000/data/tverde/fma_medium/*/' + '*.mp3')
fixed_length = 660984 # 30 seconds of audio at 22050 Hz

x_length = 1025
y_length = 1291
sr = 22050

audio_data_df = binary_wave_rdd.map(lambda x: librosa.load(io.BytesIO(x[1]))[0]) \
    .map(lambda x: librosa.util.fix_length(x, size=fixed_length)) \
    .map(lambda x: librosa.stft(x, n_fft=2048, hop_length=512)) \
    .map(lambda x: np.abs(x)) \
    .map(lambda x: librosa.feature.melspectrogram(sr=sr, S=x**2)) \
    .map(lambda x: librosa.feature.mfcc(S=librosa.power_to_db(x), n_mfcc=20)) \
    .map(lambda x: StandardScaler().fit_transform(x)) \
    .map(lambda x: x.flatten()) \
    .map(lambda x: x.tolist()) \
    .toDF(ArrayType(FloatType())).withColumn("value", array_to_vector("value")) \

# Show the first few rows of the DataFrame
audio_data_df.show()
audio_data_df.write.parquet("hdfs://localhost:9000/data/tverde/fma_vectors/stft.parquet")


