from music21 import converter, instrument, note, chord, stream, midi, corpus
import numpy as np
from midi_functions import streamToNoteArray
from dataset_functions import filter_voices, transpose, scores_from_corpus

def main():
    training_data = []
    print("Load chorales from corpus")
    chorale_list = scores_from_corpus("bach")
    voices_list = filter_voices(chorale_list)
    print("Transpose voices:")
    transposed_voices = transpose(voices_list)
    for voice in transposed_voices:
        training_data.append(streamToNoteArray(voice))
    training_data = np.array(training_data)
    np.random.shuffle(training_data)
    np.savez('bach_transposed.npz', train=training_data)
    print("Dataset saved as bach_transposed.npz")


def chorales_from_corpus():
    """ Return a list of Bach chorales """
    chorales = corpus.getComposer('bach', 'xml')
    chorale_list = []
    for chorale in chorales:
        chorale_list.append(converter.parse(chorale))
    return chorale_list



if __name__ == "__main__":
    main()
