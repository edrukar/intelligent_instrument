import numpy as np
import keras.utils as utils
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization


def create_model(voc_size, hidden_units, model_name):
    print("Creating model")
    # Build a decoding model (input length 1, batch size 1, stateful)
    model_dec = Sequential()
    model_dec.add(Embedding(voc_size, hidden_units, input_length=1, batch_input_shape=(1,1)))
    # LSTM part
    model_dec.add(LSTM(hidden_units, stateful=True, return_sequences=True))
    model_dec.add(LSTM(hidden_units, stateful=True))

    # project back to vocabulary
    model_dec.add(Dense(voc_size, activation='softmax'))
    model_dec.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    print("Loading weights")
    model_dec.load_weights(model_name)
    print("Weights loaded")
    return model_dec

    
def sample(preds, temperature=1.0):
    """ helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def sample_model(seed, model_name, length=400, temperature=1.0):
    '''Samples a musicRNN given a seed sequence.'''
    generated = []  
    next_index = seed
    for i in range(length+1):
        x = np.array([next_index])
        x = np.reshape(x,(1,1))
        preds = model_name.predict(x, verbose=0)[0]      
        next_index = sample(preds, temperature)
        generated.append(next_index)
    return np.array(generated)
