from pyspark import RDD  ## rdd es una dataset, resiliente y distribuido
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, StructType, StringType, StructField
from pyspark.ml.functions import array_to_vector
import pyspark
from sklearn.preprocessing import StandardScaler
import librosa
import io
import numpy as np
import time
from pyspark.sql.functions import regexp_extract, regexp_replace

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
binary_wave_rdd = sc.binaryFiles('hdfs://localhost:9000/data/tverde/fma_meduim/*' + '*.mp3')

fixed_length = 660984  # 30 seconds of audio at 22050 Hz

x_length = 1025
y_length = 1291
sr = 22050

df_struct = StructType([StructField("file_name", StringType(), True),
                        StructField("vector", ArrayType(FloatType()), True)])

# x[0] -> file name, x[1] bytes
audio_data_df = (binary_wave_rdd \
                 .map(lambda x: (x[0], librosa.load(io.BytesIO(x[1]))[0])) \
                 .map(lambda x: (x[0], librosa.util.fix_length(x[1], size=fixed_length))) \
                 .map(lambda x: (x[0], librosa.stft(x[1], n_fft=2048, hop_length=512))) \
                 .map(lambda x: (x[0], np.abs(x[1]))) \
                 .map(lambda x: (x[0], librosa.feature.melspectrogram(sr=sr, S=x[1] ** 2))) \
                 .map(lambda x: (x[0], librosa.feature.mfcc(S=librosa.power_to_db(x[1]), n_mfcc=20))) \
                 .map(lambda x: (x[0], StandardScaler().fit_transform(x[1]))) \
                 .map(lambda x: (x[0], x[1].flatten())) \
                 .map(lambda x: (x[0], x[1].tolist())) \
                 .toDF(df_struct) \
                 .withColumn("vector", array_to_vector("vector"))
                 .withColumn("file_name", regexp_replace(regexp_extract("file_name", r'.*/(\d+\.mp3)$', 1), "\.mp3", ""))
                 )

# Show the first few rows of the DataFrame
audio_data_df.write.parquet(f"hdfs://localhost:9000/data/tverde/fma_vectors?{time.time()}.parquet")
