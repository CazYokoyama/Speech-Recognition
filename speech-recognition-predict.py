import librosa
import numpy as np
from tflite_runtime.interpreter import Interpreter
import sounddevice as sd
import scipy.signal

train_audio_path = 'input/tensorflow-speech-recognition-challenge/rpivoice/xcsoar/'
SAMPLE_RATE = 8000

# load command list, best_model.txt
def load_labels(path):
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]
labels = load_labels('best_model.txt')

# Load TFLite model and allocate tensors.
interpreter = Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

def tflite_predict(voice, sr):
    if sr != SAMPLE_RATE:
        voice = librosa.resample(voice, sr, SAMPLE_RATE)
    interpreter.set_tensor(input_details[0]['index'], voice.reshape(1, SAMPLE_RATE, 1))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index=np.argmax(output_data[0])
    return index, output_data[0]

samples, sample_rate = librosa.load(train_audio_path + 'stop/caz-stop-plantronics.wav', sr = SAMPLE_RATE)
index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))

samples, sample_rate = librosa.load(train_audio_path + 'yes/caz-yes-plantronics.wav', sr = SAMPLE_RATE)
index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))

rec_duration = 0.25
sample_rate = 16000
num_channels = 1

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs

    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
  global window

  # Remove 2nd dimension from recording sample
  rec = np.squeeze(rec).astype('float32')

  # Resample down to SAMPLE_RATE
  rec, new_fs = decimate(rec, sample_rate, SAMPLE_RATE)

  # Save recording onto sliding window
  window[:len(window)//4] = window[len(window)//4*1:len(window)//4*2]
  window[len(window)//4*1:len(window)//4*2] = window[len(window)//4*2:len(window)//4*3]
  window[len(window)//4*2:len(window)//4*3] = window[len(window)//4*3:]
  window[len(window)//4*3:] = rec
  index, prob = tflite_predict(window, SAMPLE_RATE)
  if prob[np.argmax(prob)] > 0.85:
    print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))
    window = np.zeros(len(window), dtype='float32')

# main start here
# create sliding window
window = np.zeros(int(rec_duration * SAMPLE_RATE) * 4, dtype='float32')

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
  while True:
    pass
