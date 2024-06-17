import os
import csv
import pandas as pd
from music21 import *
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


def Load_Dataset(read_csv):
    if read_csv:
        Notes_List = []
        with open("Notes.csv", "r") as f:
            reader = csv.reader(f, delimiter="\n")
            for Notes in reader:
                Notes = Notes[0].split()
                Notes_List.append(Notes)
        return Notes_List
    else:
        Midi_List = []
        for dir in os.listdir("./Data"):
            path = f"./Data/{dir}"
            for file in os.listdir(path):
                filepath = path + f"/{file}"
                print(filepath)
                # Load data from midi file
                midi = converter.parse(filepath)
                Midi_List.append(midi)
        return Notes_Extraction(Midi_List)


def Notes_Extraction(Midi_List):
    Notes_List = []
    for midi in Midi_List:
        Notes = []
        # partition song with instrument
        songs = instrument.partitionByInstrument(midi)
        for part in songs.parts:
            if part.partName != "Piano":
                continue
            # get access to each element
            pick = part.recurse()
            for element in pick:
                # note[0]=Name, note[-1]=octave
                # note[1]=Flat or Sharp if have any
                if isinstance(element, note.Note):
                    # if this element is note, then get its pitch
                    Notes.append(str(element.pitch))
        if Notes:
            Notes = Remove_Rare(Notes)
            Notes_List.append(" ".join(Notes))
    # save as csv file
    df = pd.DataFrame(Notes_List)
    df.to_csv("Notes.csv", index=False, header=False)
    return Notes_List


def Notes_to_Index(Notes_List):
    Notes_vocab = []  # all vocabulary of notes
    Notes_index = []  # list of note index of each song
    Notes_map = {}  # map from note to index

    # list of all unique note from dataset
    unique_note = Counter(Note for Notes in Notes_List for Note in set(Notes))
    for Note in unique_note.keys():
        Notes_vocab.append(Note)

    # create map
    Notes_map = dict(
        sorted(
            {k: v for v, k in enumerate(Notes_vocab)}.items(), key=lambda item: item[1]
        )
    )

    # mapping
    for Notes in Notes_List:
        index = []
        for Note in Notes:
            index.append(Notes_map[Note])
        Notes_index.append(index)

    return Notes_index


def Index_to_Notes(Notes_List, input):
    Notes_vocab = []  # all vocabulary of notes
    Notes_index = []  # list of note index of each song
    Notes_map = {}  # map from note to index

    # list of all unique note from dataset
    unique_note = Counter(Note for Notes in Notes_List for Note in set(Notes))
    for Note in unique_note.keys():
        Notes_vocab.append(Note)

    # create map
    Notes_map = dict(
        sorted(
            {v: k for v, k in enumerate(Notes_vocab)}.items(), key=lambda item: item[1]
        )
    )

    # mapping

    return Notes_map[input]


def Remove_Rare(Corpus):
    count_unique = Counter(Corpus)

    Notes = list(count_unique.keys())
    Frequency = list(count_unique.values())

    Avg_Frequency = sum(Frequency) / len(Frequency)

    from math import ceil

    rare_note = []
    for index, (key, value) in enumerate(count_unique.items()):
        if value < ceil(Avg_Frequency):
            m = key
            rare_note.append(m)

    for element in Corpus:
        if element in rare_note:
            Corpus.remove(element)

    return Corpus


def Pitch_Diff(Corpus):
    ps_diff = []
    for i in range(1, len(Corpus)):
        former = pitch.Pitch(Corpus[i - 1]).ps
        latter = pitch.Pitch(Corpus[i]).ps
        ps_diff.append(str(latter - former))

    return ps_diff
