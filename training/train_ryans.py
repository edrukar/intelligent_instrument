import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from dataset_functions import slice_sequence_examples, seq_to_singleton_format, split

# Training Hyperparameters:
VOCABULARY_SIZE = 130 # known 0-127 notes + 128 note_off + 129 no_event
SEQ_LEN = 32
STRIDE = 8
BATCH_SIZE = 64
HIDDEN_UNITS = 256
EPOCHS = 100

data_type = "ryans_transposed"
folder_name = "./" + data_type + "_len{}_stride{}".format(SEQ_LEN,STRIDE)
weights_folder_name = folder_name + "/weights"
history_folder_name = folder_name + "/history"

# Create directories
try:
    os.mkdir(folder_name)
    print("Directory " , folder_name ,  " Created ")
except FileExistsError: print("Directory " , folder_name , " already exists")

try:
    os.mkdir(weights_folder_name)
    print("Directory " , weights_folder_name ,  " Created ")
except FileExistsError: print("Directory " , folder_name , " already exists")

try:
    os.mkdir(history_folder_name)
    print("Directory " , history_folder_name ,  " Created ")
except FileExistsError: print("Directory " , folder_name , " already exists")

doc = []
doc.append("Dataset type:\t\t\t"+data_type)
doc.append("Batch size:\t\t\t{}".format(BATCH_SIZE))
doc.append("Sample length:\t\t\t{}".format(SEQ_LEN))
doc.append("Stride:\t\t\t\t{}".format(STRIDE))
file = open(folder_name + "/Specifications.txt", "w")
file.write("\n".join(doc))
file.close()

with np.load("ryans_transposed.npz") as array:
    data = array["train"]
train_data = split(data, SEQ_LEN+1, STRIDE+1)

x_train, y_train = seq_to_singleton_format(train_data)
x_train = np.array(x_train)
y_train = np.array(y_train)

# Define model architecture
model_train = Sequential()
model_train.add(Embedding(VOCABULARY_SIZE, HIDDEN_UNITS, input_length=SEQ_LEN))
model_train.add(LSTM(HIDDEN_UNITS,return_sequences=True))
model_train.add(LSTM(HIDDEN_UNITS))
model_train.add(Dense(VOCABULARY_SIZE, activation='softmax'))
model_train.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model_train.summary()

filepath=weights_folder_name+"/{epoch:02d}-loss_{loss:.2f}-val_loss_{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min', save_best_only=False)
callbacks_list = [checkpoint]

# Train
history = model_train.fit(x_train,
                y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                callbacks=callbacks_list,
                validation_split=0.1)

# Save loss and validation loss
loss = np.array(history.history['loss'])
validation_loss = np.array(history.history['val_loss'])
loss_name = history_folder_name+'/loss.npz'
val_loss_name = history_folder_name+'/validation_loss.npz'
np.savez(loss_name, loss=loss)
np.savez(val_loss_name, loss=validation_loss)
