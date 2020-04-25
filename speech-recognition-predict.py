
# coding: utf-8

# I strongly recommend to go through the article [here](https://www.analyticsvidhya.com/blog/2019/07/learn-build-first-speech-to-text-model-python/) to understand the basics of signal processing prior implementing the speech to text.
# 
# **Understanding the Problem Statement for our Speech-to-Text Project**
# 
# Let’s understand the problem statement of our project before we move into the implementation part.
# 
# We might be on the verge of having too many screens around us. It seems like every day, new versions of common objects are “re-invented” with built-in wifi and bright touchscreens. A promising antidote to our screen addiction is voice interfaces. 
# 
# TensorFlow recently released the Speech Commands Datasets. It includes 65,000 one-second long utterances of 30 short words, by thousands of different people. We’ll build a speech recognition system that understands simple spoken commands.
# 
# You can download the dataset from [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge).
# 
# **Implementing the Speech-to-Text Model in Python**
# 
# The wait is over! It’s time to build our own Speech-to-Text model from scratch.
# 
# **Import the libraries**
# 
# First, import all the necessary libraries into our notebook. LibROSA and SciPy are the Python libraries used for processing audio signals.

# In[ ]:


import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings

warnings.filterwarnings("ignore")


# In[ ]:


train_audio_path = 'input/tensorflow-speech-recognition-challenge/rpivoice/xcsoar/'
SAMPLE_RATE = 8000


# In[ ]:


os.listdir('input/')


# **Data Exploration and Visualization**
# 
# Data Exploration and Visualization helps us to understand the data as well as pre-processing steps in a better way. 
# 
# **Visualization of Audio signal in time series domain**
# 
# Now, we’ll visualize the audio signal in the time series domain:

# In[ ]:


samples, sample_rate = librosa.load(train_audio_path+'yes/0a7c2a8d_nohash_0.wav', sr = 16000)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + train_audio_path + 'yes/0a7c2a8d_nohash_0.wav')
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)


# **Sampling rate **
# 
# Let us now look at the sampling rate of the audio signals

# In[ ]:


ipd.Audio(samples, rate=sample_rate)


# In[ ]:


print(sample_rate)


# **Resampling**
# 
# From the above, we can understand that the sampling rate of the signal is 16000 hz. Let us resample it to 8000 hz since most of the speech related frequencies are present in 8000z 

# In[ ]:


samples = librosa.resample(samples, sample_rate, SAMPLE_RATE)
ipd.Audio(samples, rate=SAMPLE_RATE)


# Now, let’s understand the number of recordings for each voice command:

# In[ ]:


labels=os.listdir(train_audio_path)


# In[ ]:


#find count of each label and plot bar graph
no_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + label) if f.endswith('.wav')]
    no_of_recordings.append(len(waves))
    
#plot
plt.figure(figsize=(30,5))
index = np.arange(len(labels))
plt.bar(index, no_of_recordings)
plt.xlabel('Commands', fontsize=12)
plt.ylabel('No of recordings', fontsize=12)
plt.xticks(index, labels, fontsize=15, rotation=60)
plt.title('No. of recordings for each command')
plt.show()


# In[ ]:


labels=["yes", "stop", "learn"]


# **Duration of recordings**
# 
# What’s next? A look at the distribution of the duration of recordings:

# In[ ]:


duration_of_recordings=[]
for label in labels:
    waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
    for wav in waves:
        sample_rate, samples = wavfile.read(train_audio_path + '/' + label + '/' + wav)
        duration_of_recordings.append(float(len(samples)/sample_rate))
    
plt.hist(np.array(duration_of_recordings))


# **Preprocessing the audio waves**
# 
# In the data exploration part earlier, we have seen that the duration of a few recordings is less than 1 second and the sampling rate is too high. So, let us read the audio waves and use the below-preprocessing steps to deal with this.
# 
# Here are the two steps we’ll follow:
# 
# * Resampling
# * Removing shorter commands of less than 1 second
# 
# Let us define these preprocessing steps in the below code snippet:

# In[ ]:


all_wave = []
all_label = []
for label in labels:
    print(label)
    waves = [f for f in os.listdir(train_audio_path + label) if f.endswith('.wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(train_audio_path + label + '/' + wav, sr = SAMPLE_RATE)
        if(len(samples)== SAMPLE_RATE) : 
            all_wave.append(samples)
            all_label.append(label)


# Convert the output labels to integer encoded:

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)


# In[ ]:


best_model = open('best_model.txt', 'w')
for word in classes:
  best_model.write(word + '\n')
best_model.close()


# Now, convert the integer encoded labels to a one-hot vector since it is a multi-classification problem:

# In[ ]:


from keras.utils import np_utils
y=np_utils.to_categorical(y, num_classes=len(labels))


# Reshape the 2D array to 3D since the input to the conv1d must be a 3D array:

# In[ ]:


all_wave = np.array(all_wave).reshape(-1,SAMPLE_RATE,1)


# **Split into train and validation set**
# 
# Next, we will train the model on 80% of the data and validate on the remaining 20%:
# 

# In[ ]:


from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)


# **Model Architecture for this problem**
# 
# We will build the speech-to-text model using conv1d. Conv1d is a convolutional neural network which performs the convolution along only one dimension. 

# **Model building**
# 
# Let us implement the model using Keras functional API.

# In[ ]:


from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()

inputs = Input(shape=(SAMPLE_RATE,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu')(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu')(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu')(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu')(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()


# Define the loss function to be categorical cross-entropy since it is a multi-classification problem:

# In[ ]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# Early stopping and model checkpoints are the callbacks to stop training the neural network at the right time and to save the best model after every epoch:

# In[ ]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')


# Let us train the model on a batch size of 32 and evaluate the performance on the holdout set:

# In[ ]:


history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val,y_val))


# **Diagnostic plot**
# 
# I’m going to lean on visualization again to understand the performance of the model over a period of time:

# In[ ]:


from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# **Loading the best model**

# In[ ]:


from keras.models import load_model
model=load_model('best_model.hdf5')


# Define the function that predicts text for the given audio:

# In[ ]:


# load command list, best_model.txt
def load_labels(path):
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]
labels = load_labels('best_model.txt')


# In[ ]:


def predict(audio):
    prob=model.predict(audio.reshape(1,SAMPLE_RATE,1))
    index=np.argmax(prob[0])
    return labels[index], prob[0]


# Prediction time! Make predictions on the validation data:

# Let us now read the saved voice command and convert it to text:

# In[ ]:


filepath='input/voice-commands/caz'


# In[ ]:


#reading the voice commands
samples, sample_rate = librosa.load(filepath + '/stop/caz-stop.wav', sr = SAMPLE_RATE)
ipd.Audio(samples,rate=sample_rate)              


# In[ ]:


#converting voice commands to text
hiscore, prob = predict(samples)
print(hiscore, '%.2f %.2f %.2f' % (prob[0], prob[1], prob[2]))


# In[ ]:


#reading the voice commands
samples, sample_rate = librosa.load(filepath + '/learn/caz-learn.wav', sr = SAMPLE_RATE)
ipd.Audio(samples,rate=sample_rate)              


# In[ ]:


#converting voice commands to text
hiscore, prob = predict(samples)
print(hiscore, '%.2f %.2f %.2f' % (prob[0], prob[1], prob[2]))


# In[ ]:


#reading the voice commands
samples, sample_rate = librosa.load(filepath + '/yes/caz-yes.wav', sr = SAMPLE_RATE)
ipd.Audio(samples,rate=sample_rate)              


# In[ ]:


#converting voice commands to text
hiscore, prob = predict(samples)
print(hiscore, '%.2f %.2f %.2f' % (prob[0], prob[1], prob[2]))


# Congratulations! You have just built your very own speech-to-text model!

# In[ ]:


import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()


# In[ ]:


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[ ]:


# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
index=np.argmax(output_data[0])
print('%s %.2f %.2f %.2f' % (labels[index], output_data[0][0], output_data[0][1], output_data[0][2]))


# In[ ]:


def tflite_predict(voice, sr):
    if sr != SAMPLE_RATE:
        voice = librosa.resample(voice, sr, SAMPLE_RATE)
    interpreter.set_tensor(input_details[0]['index'], voice.reshape(1, SAMPLE_RATE, 1))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    index=np.argmax(output_data[0])
    return index, output_data[0]


# In[ ]:


samples, sample_rate = librosa.load(filepath + '/stop/caz-stop.wav', sr = SAMPLE_RATE)
ipd.Audio(samples,rate=sample_rate)


# In[ ]:


index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))


# In[ ]:


samples, sample_rate = librosa.load(filepath + '/learn/caz-learn.wav', sr = SAMPLE_RATE)
ipd.Audio(samples,rate=sample_rate)


# In[ ]:


index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))


# In[ ]:


samples, sample_rate = librosa.load(filepath + '/yes/caz-yes.wav', sr = SAMPLE_RATE)
ipd.Audio(samples,rate=sample_rate)              


# In[ ]:


index, prob = tflite_predict(samples, sample_rate)
print('%s %.2f %.2f %.2f' % (labels[index], prob[0], prob[1], prob[2]))

