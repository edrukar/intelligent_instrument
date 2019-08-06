import numpy as np
import os
import time
from music21 import converter, instrument, note, chord, stream, midi, corpus

def slice_sequence_examples(sequence, num_steps, stride):
    """Slice a sequence into redundant sequences of length num_steps."""
    xs = []
    for i in range(0, len(sequence) - num_steps - 1, stride):
        example = sequence[i: i + num_steps]
        xs.append(example)
    return xs

def transpose(voices_list):
    """ Transpose voices to C major and A minor """
    majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("C#", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("F#", 6),("G", 5)])
    minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("C#", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("F#", 3),("G", 2)])
    transposed_streams = []

    for score in voices_list:
        key = score.analyze('key')
        print("old key", key)
        if key.mode == "major":
            halfSteps = majors[key.tonic.name]
            new_score = score.transpose(halfSteps)
            transposed_streams.append(new_score)

        elif key.mode == "minor":
            halfSteps = minors[key.tonic.name]
            new_score = score.transpose(halfSteps)
            transposed_streams.append(new_score)
        new_key = new_score.analyze('key')
        print("new key", new_key)
    return transposed_streams

def parse(midi_directory):
    streams = []
    print("Going to search:", midi_directory)
    midi_files = []
    start = time.time()
    for root, dirs, files in os.walk(midi_directory):
        for file in files:
            if ".mid" in file:
                midi_files.append(root + os.sep + file)

    print("Found", len(midi_files), "midi files.")
    print("Search took", time.time() - start)

    for file in midi_files:
        start = time.time()
        try:
            s = converter.parse(file)
            for part in s.parts:
                streams.append(part)
        except KeyboardInterrupt:
            print("Exiting")
            break
        except Exception as e:
            print("exception while parsing midi", e)
            continue
    return streams

def chorales_from_corpus():
    """ Return a list of Bach chorales """
    chorales = corpus.getComposer('bach', 'xml')
    chorale_list = []
    for chorale in chorales:
        chorale_list.append(converter.parse(chorale))
    return chorale_list

def scores_from_corpus(name):
    """ Return a list of scores """
    scores = corpus.getComposer(name)
    score_list = []
    for score in scores:
        score_list.append(converter.parse(score))
    return score_list


def split(array, sample_length, stride):
    sliced_array = []
    for seq in array:
        slices = slice_sequence_examples(seq, sample_length, stride)
        for sl in slices:
            sliced_array.append(sl)
    sliced_array = np.array(sliced_array)
    return sliced_array

def load_dataset(name, sample_length, stride):
    with np.load(name) as array:
        data = array["train"]

    spl = split(data, sample_length, stride)
    print("Subsequences before filter:",len(spl))
    spl = spl[~np.all(spl == 129, axis=1)]
    print("Subsequences after filter:",len(spl))
    #np.random.shuffle(spl)
    #validation_indices = np.arange(int(np.round(0.9 * len(spl))), len(spl))
    #validation_set = spl[validation_indices]
    #training_set = np.delete(spl, validation_indices,axis=0)
    return spl

def filter_voices(chorales):
    """ Filter out chorales with all four voices:
        Soprano
        Alto
        Tenor
        Bass
    """
    voices_list = []
    reference_set = set(["Soprano", "Alto", "Tenor", "Bass"])

    for i, chorale in enumerate(chorales):
        part_set = set([parts.id for parts in chorale])
        # Choose only chorales with all four voices.
        if reference_set.issubset(part_set):
            for voice in chorale:
                if voice.id in reference_set:
                    voices_list.append(voice)

    voices_list = np.array(voices_list)
    return voices_list

def seq_to_singleton_format(examples):
    """
    Return the examples in seq to singleton format.
    """
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs,ys)
