import argparse
from preprocess import *
from model import *


def Get_Argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--readcsv",
        action="store_true",
        help="to read notes from csv file, default=False",
        default=False,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="to train model by yourself, default=False",
        default=False,
    )
    args = parser.parse_args()
    return args.readcsv, args.train


if __name__ == "__main__":
    # determine whether to get notes from csv and remove frequency
    read_csv, train_model = Get_Argument()
    # get the list of notes of each song
    Notes_List = Load_Dataset(read_csv)

    n_vocab = len(set([item for sublist in Notes_List for item in sublist])) - 2
    Notes_List = Notes_List[0:150]

    cp_Notes_List = Notes_List.copy()
    # map note to index
    Notes_Index_List = Notes_to_Index(Notes_List)

    network_input, network_output = prepare_sequences(Notes_Index_List, n_vocab)
    model = create_network(network_input, n_vocab)

    if train_model:
        train(model, network_input, network_output)
        model.load_weights("model_weights.h5")

        eval_List = Load_Dataset(read_csv)
        eval_List = eval_List[130:150]
        eval_Index_List = Notes_to_Index(eval_List)
        eval_input, eval_output = prepare_sequences(eval_Index_List, n_vocab)

        eval(model, eval_input, eval_output, cp_Notes_List, n_vocab)
    else:
        model.load_weights("model_weights.h5")

    make(model, cp_Notes_List, n_vocab)
