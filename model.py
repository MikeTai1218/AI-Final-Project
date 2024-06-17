import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from preprocess import *


def create_network(network_input, n_vocab):
    """create the structure of the neural network"""
    model = Sequential()
    model.add(
        LSTM(
            64,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model


def train(model, network_input, network_output):
    """train the neural network"""
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
    )
    callbacks_list = [checkpoint]

    model.fit(
        network_input,
        network_output,
        epochs=60,
        batch_size=64,
        callbacks=callbacks_list,
    )

    model.save_weights("model_weights.h5")


def eval(model, input, output, Notes_List, n_vocab):
    appear = []
    for i in range(0, n_vocab):
        appear.append(0)
    pred = model.predict(input)
    eql = 0
    uneql = 0
    for i in range(len(input)):
        max = 0
        for j in range(0, len(pred[i])):
            if pred[i][j] > pred[i][max]:
                max = j
        i_pred = max
        appear[max] = 1
        max = 0
        for j in range(0, len(output[i])):
            if output[i][j] > output[i][max]:
                max = j
        i_out = max
        # print(pred[i])
        print(
            i_pred,
            Index_to_Notes(Notes_List, i_pred),
            i_out,
            Index_to_Notes(Notes_List, i_out),
        )
        if i_pred == i_out:
            eql += 1
        else:
            uneql += 1
    print(eql, eql + uneql)
    print(eql / (eql + uneql))
    sum = 0
    for i in range(0, len(appear)):
        sum += appear[i]
    print(sum, n_vocab)
    print(sum / n_vocab)


def make(model, Notes_List, n_vocab):
    input = []
    for i in range(0, 5):
        input.append(random.randrange(n_vocab))
    for i in range(0, 490):
        a = [input[len(input) - 5 : len(input)]]
        temp = prepare_sequences_in(a, n_vocab)
        pred = model.predict(temp)
        max = 0
        for j in range(0, len(pred[0])):
            if pred[0][j] > pred[0][max]:
                max = j
        i_pred = max
        input.append(i_pred)
        if (
            input[len(input) - 5 : len(input)]
            == input[len(input) - 10 : len(input) - 5]
            or input[len(input) - 4 : len(input)]
            == input[len(input) - 8 : len(input) - 4]
            or input[len(input) - 1] == input[len(input) - 2]
            or input[len(input) - 3 : len(input)]
            == input[len(input) - 6 : len(input) - 3]
            or input[len(input) - 2 : len(input)]
            == input[len(input) - 4 : len(input) - 2]
        ):
            for i in range(0, 3):
                input.append(random.randrange(n_vocab))
    ans = ""
    for i in range(0, len(input)):
        ans = ans + str(Index_to_Notes(Notes_List, input[i])) + " "
    print(ans)
    offset = 0
    output_notes = []
    for pattern in input:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="test_output.mid")


def prepare_sequences(notes, n_vocab):
    """Prepare the sequences used by the Neural Network"""
    sequence_length = 5
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for song in notes:
        for i in range(0, len(song) - sequence_length, 1):
            sequence_in = song[i : i + sequence_length]
            sequence_out = song[i + sequence_length]
            network_input.append([note / float(n_vocab) for note in sequence_in])
            network_output.append(sequence_out)

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def prepare_sequences_in(notes, n_vocab):
    """Prepare the sequences used by the Neural Network"""
    sequence_length = 5
    network_input = []

    # create input sequences and the corresponding outputs
    for song in notes:
        for i in range(0, len(song) - sequence_length + 1, 1):
            sequence_in = song[i : i + sequence_length]

            network_input.append([note / float(n_vocab) for note in sequence_in])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    # normalize input
    network_input = network_input / float(n_vocab)

    return network_input
