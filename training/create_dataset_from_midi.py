#from music21 import converter, instrument, note, chord, stream, midi
from dataset_functions import transpose, parse
from midi_functions import streamToNoteArray
import numpy as np
import pandas as pd
#import os
#import time
import sys
import h5py

def main():
    training_data = []
    midi_directory = "./Final_Fantasy_7/" # Change this to folder of your choice
    dataset_name = "ff7_transposed" # Change this to folder of your choice
    midi_streams = parse(midi_directory)
    print("Transposing")
    transposed_streams = transpose(midi_streams)
    print("Converting to integer format")
    for stream in transposed_streams:
        training_data.append(streamToNoteArray(stream))
    training_data = np.array(training_data)
    np.random.shuffle(training_data)
    print("Saving training data as {}.npz".format(dataset_name))
    np.savez("{}.npz".format(dataset_name), train=training_data)

if __name__ == "__main__":
    main()
