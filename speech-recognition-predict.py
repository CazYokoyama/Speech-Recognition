import librosa
import numpy as np
from tflite_runtime.interpreter import Interpreter

filepath='input/voice-commands/caz'
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

samples, sample_rate = librosa.load(filepath + '/stop/caz-stop.wav', sr = SAMPLE_RATE)
index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))

samples, sample_rate = librosa.load(filepath + '/learn/caz-learn.wav', sr = SAMPLE_RATE)
index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))

samples, sample_rate = librosa.load(filepath + '/yes/caz-yes.wav', sr = SAMPLE_RATE)
index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))
