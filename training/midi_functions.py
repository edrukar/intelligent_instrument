"""
    Functions provided by Charles Martin.
    https://github.com/cpmpercussion/creative-prediction/blob/master/notebooks/3-zeldic-musical-RNN.ipynb
"""

from music21 import converter, instrument, note, chord, stream, midi, corpus
import glob
import numpy as np
import pandas as pd

# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128 # (stop playing all previous notes)
MELODY_NO_EVENT = 129 # (no change from previous event)
# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.

def streamToNoteArray(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    # = np.int(np.round(stream.flat.highestTime / 0.25)) # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append([np.round(element.offset / 0.25), max(np.round(element.quarterLength / 0.25),1), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), max(np.round(element.quarterLength / 0.25),1), max([p.midi for p in element.pitches])])
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame({'on': np_stream_list.T[0], 'off': np_stream_list.T[0] + np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    df = df.sort_values(['on','pitch'], ascending=[True, False]) # sort the dataframe properly
    df = df.drop_duplicates(subset=['on']) # drop duplicate values
    # part 2, convert into a sequence of note events
    output = np.zeros(df.off.max()+1, dtype=np.int16) + np.int16(MELODY_NO_EVENT)  # set array full of no events by default.
    # Fill in the output list
    for row in df.iterrows():
        output[row[1].on] = row[1].pitch  # set note on
        output[row[1].off] = MELODY_NOTE_OFF
    return output


def noteArrayToDataFrame(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df['offset'] = df.index
    df['duration'] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = df.duration.diff(-1) * -1 * 0.25  # calculate durations and change to quarter note fractions
    df = df.fillna(0.25)
    return df[['code','duration']]


def noteArrayToStream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = noteArrayToDataFrame(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = note.Rest() # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream
