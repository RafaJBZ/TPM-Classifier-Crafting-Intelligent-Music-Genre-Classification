from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = FastAPI()

model = load_model("../trained_models/model.h5")
def preprocess_audio(y, sr, fixed_length=660984):
    y = librosa.util.fix_length(y, size=fixed_length)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    mel_spectrogram = librosa.feature.melspectrogram(sr=sr, S=magnitude ** 2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=20)
    scaled_mfcc = StandardScaler().fit_transform(mfcc)
    flattened_mfcc = scaled_mfcc.flatten()
    return flattened_mfcc.tolist()
class ClassificationResult(BaseModel):
    class_name: str
    probability: float


@app.post("/classify")
async def classify_file(file: UploadFile = File(...)):

    mp3_content = await file.read()

    y, sr = librosa.load(io.BytesIO(mp3_content), sr=None)

    audio_data = preprocess_audio(y, sr)

    predictions = model.predict(np.expand_dims(audio_data, axis=0))[0]

    class_names = [
        "International", "Soul-RnB", "Instrumental", "Rock", "Jazz",
        "Folk", "Old-Time / Historic", "Blues", "Experimental", "Pop",
        "Electronic", "Hip-Hop", "Classical", "Spoken", "Country", "Easy Listening"
    ]

    prediction_tuples = list(zip(class_names, predictions))

    sorted_predictions = sorted(prediction_tuples, key=lambda x: x[1], reverse=True)

    top_5_predictions = sorted_predictions[:5]

    top_5_predictions_dict = {cn: f"{pred * 100:.4f}" for cn, pred in top_5_predictions}

    return top_5_predictions_dict

