"""
Train and evaluate a SpanModel for SemEval-2021-task6-subtask2

For data, see: https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus

"""

from span.span_model import SpanModel
from span.span_utils import (
    Fragment,
    filter_sentence_splitter,
    split_sentences_multi,
    count_fragments,
    split_in_sentences,
    decode,
    check_change,
    calibrate,
    surrounding_word,
    fragment_train_test_split,
    data_overview,
    span_data_to_dataframe
)


import json
import os
import requests
from pprint import pprint

TRAINING_ARTICLES_2020_DIR = '../semeval2020/propaganda_detection/datasets/train-articles'

LABELS_2021 = [
    "Appeal to authority",
    "Appeal to fear/prejudice",
    "Black-and-white Fallacy/Dictatorship",
    "Causal Oversimplification",
    "Doubt",
    "Exaggeration/Minimisation",
    "Flag-waving",
    "Glittering generalities (Virtue)",
    "Loaded Language",
    "Misrepresentation of Someone's Position (Straw Man)",
    "Name calling/Labeling",
    "Obfuscation, Intentional vagueness, Confusion",
    "Presenting Irrelevant Data (Red Herring)",
    "Reductio ad hitlerum",
    "Repetition",
    "Slogans",
    "Smears",
    "Thought-terminating cliché",
    "Whataboutism",
    "Bandwagon"
]

LABELS_2020_to_2021 = {
    'translate': {
        'Loaded_Language': 'Loaded Language',
        'Appeal_to_fear-prejudice': 'Appeal to fear/prejudice',
        'Appeal_to_Authority': 'Appeal to authority',
        'Causal_Oversimplification': 'Causal Oversimplification',
        'Thought-terminating_Cliches': 'Thought-terminating cliché',
        'Flag-Waving': 'Flag-waving',
        'Name_Calling,Labeling': 'Name calling/Labeling',
        'Doubt': 'Doubt',
        'Slogans': 'Slogans',
        'Repetition': 'Repetition',
        'Exaggeration,Minimisation': 'Exaggeration/Minimisation',
        'Black-and-White_Fallacy': 'Black-and-white Fallacy/Dictatorship'
    },
    'label_to_group': {
        'Bandwagon,Reductio_ad_hitlerum': ['Bandwagon', 'Reductio ad hitlerum'],
        'Whataboutism,Straw_Men,Red_Herring': [
            'Presenting Irrelevant Data (Red Herring)',
            'Misrepresentation of Someone\'s Position (Straw Man)',
            'Whataboutism'
        ]
    }
}

def normalize_label(label: str) -> list:
    """Normalize label to a standard set. Return a list"""
    if label in LABELS_2020_to_2021['translate'].keys():
        return [LABELS_2020_to_2021['translate'][label]]
    elif label in LABELS_2020_to_2021['label_to_group'].keys():
        # Originally: return all (new) labels that came from splitting an old one
        #return _LABELS['label_to_group'][label]
        # Now: ignore labels that have been split, so as not to learn false positives
        return []
    else:
        print("Untranslated: ", label)
        return [label]

@check_change
def filter_translate_2020_2021_1_2(data):
    """Normalize 2020 label data to 2021 label data"""
    new_data = []
    for d in data:
        new_fragments = []
        for f in d['fragments']:
            new_label = normalize_label(f.label)
            # We ony move labels if there is a translation for them, otherwise we ignore the fragment
            if new_label:
                f.label = new_label[0]
                new_fragments.append(f)
        d['fragments'] = new_fragments
        new_data.append(d)
    data = new_data
    return data


def read_article(article_id: str) -> str:
    """
    FIXME: using hard-coded path to directory with training articles
    :param self:
    :param article_id:
    :return:
    """
    global TRAINING_ARTICLES_2020_DIR
    p = os.path.join(TRAINING_ARTICLES_2020_DIR, f'article{article_id}.txt')
    with open(p, 'r', encoding='utf8') as f:
        return f.read()


def loadTAB2020(path: str):
    """Original label_identifier is 'semeval2020', translate to 'semeval2021'
    If label_identifier is set, we translate here in _get_data, so that happens
    only when parsing the original dataset
    """
    # self.labels = read_labels_from_file(label_identifier)

    # File containing labelled data
    #label_file = os.path.join(self.task_config['dir'], self.task_config['label_file'])
    with open(path, 'r') as f:
        lst = f.readlines()

    data = []
    prev_article_id = -1
    fragments = []
    for l in lst:
        article_id, p_type, start, end = l.strip().split('\t')

        article_id = int(article_id)
        if article_id == prev_article_id:
            # FIXME: hardcoded exceptions for label_identifiers
            #if label_identifier != 'semeval2020':
            #    for n_p_type in normalize_label(p_type):
            #        fragments.append(Fragment(int(start), int(end), n_p_type))
            #else:
            fragments.append(Fragment(int(start), int(end), p_type))
        else:
            if prev_article_id != -1:
                # Add the previous article
                # print("APPENDING article ", prev_article_id)
                fragments.sort()  # Sort on first tuple element, that is, 'start'
                data.append({
                    'id': prev_article_id,
                    'article': read_article(prev_article_id),
                    'fragments': fragments
                })

            # Prepare the new one
            prev_article_id = article_id
            fragments = []
            #if label_identifier != 'semeval2020':
            #    for n_p_type in normalize_label(p_type):
            #        fragments.append(Fragment(int(start), int(end), n_p_type))
            #else:
            #    fragments.append(Fragment(int(start), int(end), p_type))

    if fragments != []:
        # print("Appending ", prev_article_id)
        data.append({
            'id': prev_article_id,
            'article': read_article(prev_article_id),
            'fragments': fragments
        })
    print("Total fragments = ", sum([len(d['fragments']) for d in data]))
    return data


def loadJSON2021(path: str):
    """
    :param path: Comma separated list of paths to JSON files
    :return:
    """
    data = []
    file_list = [x.strip() for x in path.split(',')]
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf8') as f:
            json_data = json.load(f)

        for item in json_data:
            id = item['id']
            txt = item['text']
            fragments = []

            labels = item['labels']  # labels = [{start:, end:, technique:, text_fragment}, ...]
            # We skip that 'text_fragment' as it can be derived from text and start/end
            for frag in labels:
                if frag['text_fragment'].strip() == '':
                    # Deal with a '\n' fragment by skipping it
                    continue
                fragments.append(Fragment(frag['start'], frag['end'], frag['technique']))

            fragments.sort()
            data.append({'id': id, 'article': txt, 'fragments': fragments})
    return data

def get_all_dirty_labels(s):
    """Get any label from an outcome, irrespective of being matched or not"""

    return []


def train_ptc_2020(model, args={}):
    """To train with data from the Propaganda Techniques Corpus from
    SemEval-2020 Task 11, you will need to obtain and install it yourself.

    See: https://propaganda.qcri.org/semeval2020-task11/

    You'll have to modify the link below and the one in the `read_article` function.
    """

    # 2020 PTC data
    data_2020 = loadTAB2020('../semeval2020/propaganda_detection/datasets/train-task2-TC.labels')
    data_2020 = filter_translate_2020_2021_1_2(
                    filter_sentence_splitter(data_2020)
    )
    #train_data_2020, eval_data_2020 = train_test_split(data_2020, test_size=0.2, shuffle=True)
    train_data_2020, eval_data_2020 = fragment_train_test_split(data_2020, LABELS_2021,
                                                                test_size=0.2, shuffle=True)

    model.train_model(train_data_2020, eval_data=eval_data_2020, args=args)


def train_memes_2021(model, args={}):
    # 2021 'meme' data
    training_set_task2 = 'training_set_task2.txt'
    file_exists_or_geturl_and_save(training_set_task2)
    data_2021 = loadJSON2021(training_set_task2)
    #train_data_2021, eval_data_2021 = train_test_split(data_2021, test_size=0.2, shuffle=True)
    train_data_2021, eval_data_2021 = fragment_train_test_split(data_2021, LABELS_2021,
                                                                test_size=0.2, shuffle=True)

    print("TRAIN DATA 2021")
    pprint(data_overview(train_data_2021))


    print("EVAL DATA 2021")
    pprint(data_overview(eval_data_2021))

    model.train_model(train_data_2021, eval_data=eval_data_2021, random_mask=0.0, args=args)


def get_model(encoder_decoder_type, name_or_path, split_in_sentences=False, args={}):
    return SpanModel(encoder_decoder_type=encoder_decoder_type,
                     encoder_decoder_name=name_or_path,
                     labels=LABELS_2021,
                     split_in_sentences=split_in_sentences,
                     use_cuda=False,
                     args=args)

def file_exists_or_geturl_and_save(filename):
    if not os.path.exists(filename):
        url = f'https://raw.githubusercontent.com/di-dimitrov/SEMEVAL-2021-task6-corpus/main/data/{filename}'
        download = requests.get(url).content.decode('utf-8')
        with open(filename, 'w') as f:
            f.write(download)


if __name__ == '__main__':
    WITH_TRAINING_2020 = False
    WITH_TRAINING_2021 = True
    if WITH_TRAINING_2020:
        args = dict(
            num_train_epochs=30,
            use_multiprocessing=False,
            overwrite_output_dir=True,
            output_dir='ptc_2020'
        )
        model = get_model('bart', 'facebook/bart-base', args=args)
        train_ptc_2020(model, args=args)

    if WITH_TRAINING_2021:
        args = dict(
            num_train_epochs=1,
            use_multiprocessing=False,
            overwrite_output_dir=True,
            output_dir='memes_2021'
        )
        if WITH_TRAINING_2020:
            model_path = 'ptc_2020'
        else:
            model_path = 'facebook/bart-base'
        model = get_model('bart', model_path, args=args)
        train_memes_2021(model, args=args)

    prediction_args = dict(
        max_length=200,
        length_penalty=0.4,  # Found best value to be 0.4
        repetition_penalty=2.0,  # Found best value to be 2.0
        num_beams=5,  # Found best values in 5 and 3
        num_return_sequences=1,
        top_p=0.8,  # Found best value to be 0.8
        top_k=0,  # Set to 0
        do_sample=True
    )
    model = get_model('bart', 'memes_2021', split_in_sentences=True, args=prediction_args)

    test_set_task2_filename = 'test_set_task2.txt'
    file_exists_or_geturl_and_save(test_set_task2_filename)

    test_data_2021 = loadJSON2021(test_set_task2_filename)

    #print(model.eval_model(test_data_2021))

    sentences = [d['article'] for d in test_data_2021]
    outcome = model.predict(sentences)

    res = []
    for index in range(len(test_data_2021)):
        orig_input = test_data_2021[index]
        d = {'id': orig_input['id'], 'article': orig_input['article']}

        labels = []
        for f in outcome[index]['fragments']:
            labels.append({'start': f.start, 'end': f.end, 'technique': f.label,
                           'text_fragment': f.extract(orig_input['article'])})
        d['labels'] = labels
        res.append(d)

    predict_file = 'task2_prediction.txt'
    print(f'Writing outcome to "{predict_file}"')
    pprint(res)
    with open(predict_file, 'w', encoding='utf8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)





