import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load one song
X, sr = librosa.load('./data/fma_small/000/000002.mp3')
print(sr)

#%%
X.shape

stft = np.abs(librosa.stft(X, n_fft=2048, hop_length=512))
print(stft.shape)

mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
print(sr)
log_mel = librosa.amplitude_to_db(mel)
print(log_mel.shape)


mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
mfcc = StandardScaler().fit_transform(mfcc)
print(mfcc.shape)