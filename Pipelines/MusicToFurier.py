from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, StructType, StringType, StructField
from pyspark.ml.functions import array_to_vector
import pyspark
from sklearn.preprocessing import StandardScaler
import librosa
import io
import numpy as np
import time
from pyspark.sql.functions import regexp_extract, regexp_replace, udf
import pandas as pd

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
binary_wave_rdd = sc.binaryFiles('data/fma_small/000/' + '000002.mp3')
# Import metadata and features.
tracks = pd.read_csv('data/fma_metadata/tracks.csv', index_col=0, header=[0, 1])


get_genre_udf = udf(lambda x: tracks.loc[int(x)]['track']['genre_top'], StringType())


fixed_length = 660984  # 30 seconds of audio at 22050 Hz

x_length = 1025
y_length = 1291
sr = 22050

df_struct = StructType([StructField("top_genere", StringType(), True),
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
                 .withColumn("top_genere", regexp_replace(regexp_extract("top_genere", r'.*/(\d+\.mp3)$', 1), "\.mp3", ""))
                 .withColumn("top_genere", regexp_replace("top_genere", "^0+", ""))
                 .withColumn("top_genere", get_genre_udf("top_genere"))
                 )

# Show the first few rows of the DataFrame
audio_data_df.write.parquet(f"/home/rafajbz/data/fma_vectors/{time.time()}.parquet")
