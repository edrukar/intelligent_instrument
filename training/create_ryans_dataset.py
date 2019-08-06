from music21 import converter, instrument, note, chord, stream, midi#, corpus
import numpy as np
from midi_functions import streamToNoteArray
from dataset_functions import scores_from_corpus, transpose

def main():
    training_data = []
    print("Load scores from corpus")
    score_list = scores_from_corpus('ryansMammoth')
    print("Transpose scores:")
    transposed_scores = transpose(score_list)
    print("Converting to integer format")
    for score in transposed_scores:
        training_data.append(streamToNoteArray(score))
    training_data = np.array(training_data)
    np.random.shuffle(training_data)
    np.savez('ryans_transposed.npz', train=training_data)
    print("Dataset saved as ryans_transposed.npz")

if __name__ == "__main__":
    main()
