# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv


def create_audio_file():
    """
    Creates audio file by taking user input from microphone
    """
    # sampling frequency
    frequency = 44400

    # recording duration in seconds
    duration = 5.0

    print("Starting taking audio input...")

    # to record audio from sound-device into a NumPy
    recording = sd.rec(int(duration * frequency), samplerate=frequency, channels=2)

    # Wait for the audio to complete
    sd.wait()

    # Using scipy to save the recording in .wav format.
    # This will convert the NumPy array to an audio file with the given
    # sample frequency
    write("recording0.wav", frequency, recording)

    # Using wavio to save the recording in .wav format.
    # This will convert the NumPy array to an audio file with the 
    # given sampling frequency
    wv.write("recording1.wav", recording, frequency, sampwidth=2)

    print("End of taking audio input.")

create_audio_file()
