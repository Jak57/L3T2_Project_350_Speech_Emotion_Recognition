import numpy as np
import librosa
import pickle

print("hello world")

def extract_feature(file_name, mfcc, chroma, mel):
  # with soundfile.SoundFile(file_name) as sound_file:
  # X = sound_file.read(dtype='float32')
  # sample_rate = sound_file.samplerate
  X, sample_rate = librosa.load(file_name)

  # print(X, sample_rate)

  if chroma:
    stft = np.abs(librosa.stft(X))

  result = np.array([])
  if mfcc:
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))

  if chroma:
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

  if mel:
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

  return result


# filename = "E:\SUBESCO\F_01_OISHI_S_10_ANGRY_1.wav"
# feature=extract_feature("E:\SUBESCO\F_01_OISHI_S_10_ANGRY_1.wav", mfcc=True, chroma=True, mel=True)

# feature=feature.reshape(1,-1)

# print(feature)


    # filename = 'modelForPrediction1.sav'
    # loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

    # feature=extract_feature("E:\SUBESCO\F_01_OISHI_S_10_ANGRY_1.wav", mfcc=True, chroma=True, mel=True)

    # print(feature)
    # feature=feature.reshape(1,-1)

    # prediction=loaded_model.predict(feature)
    # print(prediction)
    # return prediction


def final_prediction(audiofile):

    filename = 'modelForPrediction1.sav'
    loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

    feature=extract_feature(audiofile, mfcc=True, chroma=True, mel=True)

    # print(feature)
    feature=feature.reshape(1,-1)

    prediction=loaded_model.predict(feature)
    # print(prediction)
    return prediction


prediction = final_prediction("E:\SUBESCO\F_01_OISHI_S_10_ANGRY_1.wav")
print(prediction)