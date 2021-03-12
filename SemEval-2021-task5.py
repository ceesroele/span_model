"""

Solution to SemEval-2021 Task5: "Toxic Spans Detection"

See: https://sites.google.com/view/toxicspans

For details on data: https://competitions.codalab.org/competitions/25623#learn_the_details-data

And for the data itself:
https://github.com/ipavlopoulos/toxic_spans
"""

import os
import requests
import pandas as pd
import numpy as np
from ast import literal_eval
from pprint import pprint
from span.span_utils import (
    Fragment,
    fragment_train_test_split,
    filter_sentence_splitter_with_empty,
    filter_sentence_splitter
)
from span.span_model import SpanModel
import statistics

# Location of data files ('raw')
BASE_URL = 'https://raw.githubusercontent.com/ipavlopoulos/toxic_spans/master/data'

test_data_file = 'tsd_test.csv'
train_data_file = 'tsd_train.csv'
trial_data_file = 'tsd_trial.csv'


# ---- scorer from: https://github.com/ipavlopoulos/toxic_spans/blob/master/evaluation/metrics.py ----

def pairwise_operator(codes, method):
    """
    Pairwsise operator based on a method and a list of predictions (e.g., lists of offsets)
    >>> assert pairwise_operator([[],[],[]], f1) == 1
    :param codes: a list of lists of predicted offsets
    :param method: a method to use to compare all pairs
    :return: the mean score between all possible pairs (excl. duplicates)
    """
    pairs = []
    for i,coderi in enumerate(codes):
        for j,coderj in enumerate(codes):
            if j>i:
                pairs.append(method(coderi, coderj))
    return np.mean(pairs)


def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions)==0 else 0
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return nom/denom



# ----------------------------------------------------------------------------------------------------


def file_exists_or_geturl_and_save(filename):
    if not os.path.exists(filename):
        url = f'{BASE_URL}/{filename}'
        download = requests.get(url).content.decode('utf-8')
        with open(filename, 'w') as f:
            f.write(download)


def index_list_to_fragments(index_list: list):
    """Convert format [9, 10, 11, 12, 20, 21, 22, 23] to [Fragment(9,12,'Toxic'), Fragment(20,23, 'Toxic')]"""
    start = None
    latest = None
    fragments = []
    for i in index_list:
        if start is None:
            # start new fragment
            start = i
            latest = i
        elif i == latest + 1:
            # extend current fragment
            latest = i
        elif i > latest + 1:
            # next fragment; note that index_list is inclusive and Fragment.end exclusive
            fragments.append(Fragment(start, latest + 1, 'Toxic'))
            start = i
            latest = i
    if start is not None and latest is not None:
        # Note that index_list is inclusive and Fragment.end exclusive
        fragments.append(Fragment(start, latest + 1, 'Toxic'))

    return fragments


def fragments_to_index_list(fragment_list: list):
    """Convert format [Fragment(9,12,'Toxic'), Fragment(20,23, 'Toxic')] to format [9, 10, 11, 12, 20, 21, 22, 23] to"""
    index_list = []
    for f in fragment_list:
        index_list.extend(list(range(f.start, f.end)))
    return index_list


def get_data(filename):
    file_exists_or_geturl_and_save('tsd_train.csv')
    df = pd.read_csv("tsd_train.csv")
    df["spans"] = df.spans.apply(literal_eval)
    data = []
    counter = 0
    for index_list, text in df.itertuples(index=False, name=None):
        # convert index_list to a list of instances of the Fragment dataclass
        #index_list = eval(index_list)

        fragments = index_list_to_fragments(index_list)

        data.append({'id': f'id_{str(counter)}', 'orig': index_list, 'article': text, 'fragments': fragments})
        counter += 1

    return data


def get_model(encoder_decoder_type, name_or_path, split_in_sentences=False, use_cuda=True, args={}):
    return SpanModel(encoder_decoder_type=encoder_decoder_type,
                     encoder_decoder_name=name_or_path,
                     labels=LABELS,
                     split_in_sentences=split_in_sentences,
                     use_cuda=use_cuda,
                     args=args)


def train(model, data):
    train_data, eval_data = fragment_train_test_split(data, LABELS, test_size=0.2, shuffle=True)
    model.train_model(train_data, eval_data=eval_data)


if __name__ == '__main__':
    LABELS = ['Toxic']
    train_data = get_data('tsd_train.csv')

    data = filter_sentence_splitter_with_empty(train_data)

    args = dict(
        num_train_epochs=20,
        use_multiprocessing=False,
        overwrite_output_dir=True,
        output_dir='toxic'
    )
    model = get_model('bart', 'facebook/bart-base', split_in_sentences=True, use_cuda=True, args=args)

    train(model, data)

    test_data = get_data('tsd_test.csv')

    # The used parameters here have not been optimised for Task 5, they have been copied
    # from working settings for Task 6.
    prediction_args = dict(
        max_length=200,
        length_penalty=0.4,  # Found best value to be 0.4
        repetition_penalty=2.0,  # Found best value to be 2.0
        num_beams=5,  # Found best values in 5 and 3
        num_return_sequences=1,
        top_p=0.8,  # Found best value to be 0.8
        top_k=0,  # Set to 0 when using top_p
        do_sample=True
    )
    model = get_model('bart', 'toxic', split_in_sentences=True, use_cuda=True, args=prediction_args)
    outcomes = model.predict([d['article'] for d in test_data])

    f1_list = []
    for i in range(len(test_data)):
        td = test_data[i]
        o = outcomes[i]
        f1_list.append(
            f1(
                fragments_to_index_list(td['fragments']),
                fragments_to_index_list(o['fragments'])
            )
        )

    print('F1 average = ', statistics.mean(f1_list))

