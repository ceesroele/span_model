"""
Utilities for span_model module

"""

from dataclasses import dataclass

import regex as re
import pandas as pd
import numpy as np
import random
from pprint import pprint
from skmultilearn.model_selection import iterative_train_test_split
from statistics import mean, median

import logging
import sys


# Use this logger in all modules
logger = logging.getLogger("span")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


ERRORS = {
    'cwo': 'Closure without opening',
    'fpd': 'False Positive Double',  # Both opening and closing tags present
    'fpco': 'False Negative Closing only',  # Closing tag that is false positive
    'fpoo': 'False Positive Opening only',  # Opening tag that is false positive
    'fn': 'False Negative',  # Tag missing
}

@dataclass
class Fragment:
    """Object identifying a fragment by the begin and end of its span and its label"""
    start: int
    end: int
    label: str

    def __gt__(self, other):
        return self.start > other.start

    def __lt__(self, other):
        return self.start < other.start

    def __ge__(self, other):
        return self.start >= other.start

    def __le__(self, other):
        return self.start <= other.start

    def __len__(self):
        return self.end - self.start

    def __add__(self, n: int):
        return Fragment(self.start + n, self.end + n, self.label)

    def __sub__(self, n: int):
        return Fragment(self.start - n, self.end - n, self.label)

    def __and__(self, other):
        if self.start < other.start:
            a, b = self, other
        else:
            a, b = other, self
        if b.start >= a.end:
            return None
        else:
            start = b.start
            end = min(a.end, b.end)
            #intersection_start = b.start - a.start
            #intersection_end = end - a.start
            #txt = a.text[intersection_start:intersection_end]
            return Fragment(start, end, f'{a.label},{b.label}')

    def __or__(self, other):
        if self.start < other.start:
            a, b = self, other
        else:
            a, b = other, self
        if b.start >= a.end:
            start = a.start
            end = max(a.end, b.end)
            #text = a.text + '*' * (b.start - a.end) + b.text
            labels = f'{a.label},{b.label}'
        else:
            start = a.start
            end = max(a.end, b.end)
            #b_start = a.start + len(a.text) - b.start
            #text = a.text + b.text[b_start:]
            labels = f'{a.label},{b.label}'
        return Fragment(start, end, labels)

    def extract(self, article):
        """Slice the text string for this fragment out of the article"""
        return article[self.start:self.end]


def count_fragments(data: list) -> int:
    return sum([len(f) for d in data for f in d['fragments']])


def label_to_symbol(label: str, all_labels: list) -> str:
    """Convert a label to start and end of a special symbol to use as input or output for encoder/decoder"""
    index = all_labels.index(label)
    in_symbol = f"[i-{index}]"
    out_symbol = f"[o-{index}]"
    return in_symbol, out_symbol


def symbol_to_label(symbol: str, all_labels: list) -> str:
    """Convert a label to a special symbol to use as input or output for encoder/decoder"""
    m = re.search(r'[i-(\d+)]', symbol)
    n = re.search(r'[o-(\d+)]', symbol)
    if m is None and n is None:
        raise ValueError(f'Symbol {symbol} fails to match symbol regex')
    elif m is not None:
        return all_labels[m.group(1)]
    else:
        return all_labels[n.group(1)]


def get_tokens_from_labels(labels):
    """Get all begin and end tokens for a list of labels"""
    # Get begin and end tokens for our labels
    tokens = []
    for l in labels:
        begin_token, end_token = label_to_symbol(l, labels)
        tokens.extend([begin_token, end_token])
    return tokens


def _mask_ahead(article, index):
    """From index move ahead to the next word"""
    word_ahead = re.compile(r'^(\w+)\W')
    m = word_ahead.match(article[index:])
    if m:
        return index + len(m.group(1))
    else:
        return index

def _mask_back(article, index):
    """From index move back to the previous word"""
    word_back = re.compile(r'^(\w+)\W')
    m = word_back.match(article[:index])
    if m:
        return index - len(m.group(1))
    else:
        return index


def mask(article: str, fragments: list, all_labels: list, dropout=0.0, random_mask=0.0) -> str:
    label_types = list({f.label for f in fragments})
    label_dict = {l: [None] * len(article) for l in label_types}
    symbols = {l: label_to_symbol(l, all_labels) for l in label_types}
    _IN_, _OUT_ = 0, 1

    for f in fragments:
        label_dict[f.label][f.start] = symbols[f.label][_IN_]
        label_dict[f.label][f.end - 1] = symbols[f.label][_OUT_]

    # We now have a matrix with the article in the first column and then columns for each label
    # Let's put it together into a single list

    lst = []
    i = 0
    while i < len(article):
        remove_next = False
        # First we write IN symbols
        for l in label_types:
            sym = label_dict[l][i]
            guess = random.uniform(0, 1)
            if sym is not None and sym.startswith('[i-'):
                if guess >= (1 - random_mask):
                    remove_next = True
                else:
                    pass
        # Then content
        if not remove_next:
            lst.append(article[i])
        else:
            lst.append('<mask>')
            i = _mask_ahead(article, i)
        # And last we write the OUT symbols
        for l in label_types:
            sym = label_dict[l][i]
            guess = random.uniform(0, 1)
            if sym is not None and sym.startswith('[o-'):
                if guess >= (1 - random_mask):
                    if not remove_next:
                        if len(lst) > len('<mask>'):
                            i = _mask_back(article, i)
                            lst = lst[:i]
                            lst.append('<mask>')
                else:
                    pass
        i += 1
    s = "".join(lst)
    return s


def encode(article: str, fragments: list, all_labels: list, dropout=0.0) -> str:
    """Indicate fragments in an article through Begin and End identifiers per label

    :param dropout Probability of leaving out labels
    """
    label_types = list({f.label for f in fragments})
    label_dict = {l: [None] * len(article) for l in label_types}
    symbols = {l: label_to_symbol(l, all_labels) for l in label_types}
    _IN_, _OUT_ = 0, 1

    for f in fragments:
        label_dict[f.label][f.start] = symbols[f.label][_IN_]
        label_dict[f.label][f.end - 1] = symbols[f.label][_OUT_]

    # We now have a matrix with the article in the first column and then columns for each label
    # Let's put it together into a single list

    lst = []
    for i in range(len(article)):
        # First we write IN symbols
        for l in label_types:
            sym = label_dict[l][i]
            guess = random.uniform(0, 1)
            if sym is not None and sym.startswith('[i-'):
                lst.extend([' ', sym, ' '])
        # And last we write the OUT symbols
        lst.append(article[i])
        for l in label_types:
            sym = label_dict[l][i]
            guess = random.uniform(0, 1)
            if sym is not None and sym.startswith('[o-'):
                lst.extend([' ', sym, ' '])

    s = "".join(lst)
    return s


def decode(s: str, all_labels: list) -> tuple:
    """Convert a string with Begin and End markers into an article and a list of fragments."""
    article = []  # article is built up character by character
    label_dict = {label_index: None for label_index in range(len(all_labels))}
    re_start = r'^\[i-(\d+)\]\s?'
    re_end = r'^\s?\[o-(\d+)\]'
    index = 0
    fragments = []
    errors = 0  # number of errors during processing
    while index < len(s):
        m = re.match(re_start, s[index:])
        n = re.match(re_end, s[index:])
        if m is not None:
            # Set start marker for the label to the current length of the article
            label_index = int(m.group(1))
            if label_dict[label_index] is not None:
                print(f"Ignoring unclosed open marker to open new for '{all_labels[label_index]}'/{label_index} in\n(({s}))")
                errors += 1
            label_dict[label_index] = len(article)
            index += len(m.group(0))
        elif n is not None:
            label_index = int(n.group(1))
            cur_start = label_dict[label_index]
            if cur_start is None:
                # We found an end marker for which there is no start marker
                print(f"Ignoring end marker without start marker for '{all_labels[label_index]}'/{label_index} in\n"
                      f"(({s}))")
                errors += 1
            else:
                cur_end = len(article)
                while cur_start < cur_end and article[cur_start] in [' ', '\n', '\t']:
                    cur_start += 1
                if cur_end > cur_start:
                    f = Fragment(cur_start, cur_end, all_labels[label_index])
                    fragments.append(f)
                label_dict[label_index] = None
            index += len(n.group(0))
        else:
            # Regular character. We can just add, but then we have to do bookkeeping
            article.append(s[index])
            index += 1

    # Check if there are loose ends, that is, open markers without closure
    for l_index in range(len(all_labels)):
        if label_dict[l_index] is not None:
            print(f"Ignoring unclosed marker for label '{all_labels[l_index]}'/{l_index} in\n(({s}))")
            errors += 1

    return ''.join(article), fragments, errors


def span_data_to_dataframe(data, all_labels, random_mask=0.0):
    """
    Args:
        data: list of dictionaries with keys: ['id', 'article', 'fragments']

    Returns:
        Pandas dataframe with columns ['id', 'input_text', 'target_text']
    """
    if data is None:
        return None

    lst = []
    for d in data:
        if random_mask > 0.0:
            input_text = mask(d['article'], d['fragments'], all_labels, random_mask=random_mask)
        else:
            input_text = d['article']
        target_text = encode(d['article'], d['fragments'], all_labels)
        lst.append([d['id'], input_text, target_text])

    df = pd.DataFrame(lst, columns=['id', 'input_text', 'target_text'])
    print('Length of input_text: ', len(lst))
    pprint(df.head())
    return df


def offset_fragments(fragments, offset):
    """Correct indices of fragments with an offset.

    When an article is stripped of its beginning, all fragments need to be reindexed
    with the length of what was stripped.
    """
    res = []
    for f in fragments:
        assert f.start >= offset, f'{str(f)} >= {offset}, fragments = {str(fragments)}'
        res.append(f - offset)  # Note: substraction from the Fragment subtracts from its start and end indexes
    return res


def split_sentences_multi(id: str, article: str, fragments: list, include_empty=False, level=0):
    """Split an article into multiple sections, each with its own list of fragments

    :param: include_empty: include items for which there are no fragments with labels (that is, no labels match)
    """
    #print("****** ID = ", id, 'level', level, 'start index', start_index, 'len article', len(article), "*******")
    data = []
    number_of_fragments = len(fragments)
    fragment_texts = [f.extract(article) for f in fragments]

    #print('original fragments', fragment_texts, fragments)
    if fragments == []:
        # Just ignore this
        # FIXME: not correct, these should be added too if 'include_empty' is True
        print(f"Ignoring article with length {len(article)} for empty fragments")
        # return data
    else:
        # forcibly sort fragments, shouldn't be necessary as they are supposed to be sorted already...
        fragments.sort()

        # First deal with sentences without fragment
        sentences, start = preceding_non_labelled_sentences(id, article, fragments)

        # Note that here 'sentences' really only contains the strings, not [id, string] pairs
        data.extend([{'id': f'{id}p{i}', 'article': s, 'fragments': []} for i, s in enumerate(sentences)])

        # Now add the first sections that contains labels.
        # Note that we must deal with the 'start' offset coming from preceding_non_labelled_sentences
        # in both article and fragments.
        start_text, start_fragments, rest_text, rest_fragments, new_start = \
            split_sentences_multi_until(
                article[start:],
                offset_fragments(fragments, start)
            )
        data.append({'id': f'{id}x', 'article': start_text, 'fragments': start_fragments})

        # Now recursively deal with the rest
        if rest_text == '':
            # Nothing to be done, just return the input data
            pass
        elif rest_fragments == []:
            # if there are no more fragments, split article in sentences up to the end
            new_data = [{'id': f'{id}r{i}', 'article': s, 'fragments': []} for i, s in enumerate(split_in_sentences(rest_text)) if s.strip() != '']
            #print("split in sentences, new data: ", new_data)
            data.extend(new_data)
        else:
            #print("now recursively dealing with ", rest_text, rest_fragments)
            data.extend(split_sentences_multi(f'{id}_{level}', rest_text, rest_fragments, include_empty=include_empty, level=level+1))

        # Remove empty fragments
        if not include_empty:
            new_data = [d for d in data if d['fragments'] != []]
            data = new_data

        new_fragment_texts = [f.extract(d['article']) for d in data for f in d['fragments']]

        for t in new_fragment_texts:
            if level == 0 and t not in fragment_texts:
                found = False
                for ft in fragment_texts:
                    if ft.startswith(t):
                        found = True
                        print(f"+++ Output fragment {[t]} is truncated \n--- from {[ft]}")
                        break
                if not found:
                    print(f"=== Got output level {level} fragment text {str([t])} not in input fragments ", fragment_texts,
                          ' ** total output is ', new_fragment_texts)
                    print("All original ", len(fragments), "fragments: ", fragments) # , "original text: ", article)


        n = count_fragments(data)

        #print(f"Level {level} output of {n} fragments for input of input: {number_of_fragments}")

    return data


def next_nl_or_end(s, n=0):
    """Next newline or end of text"""
    # first identify starting newlines, we pass them
    start = n
    while start < len(s) - 1 and s[start] == '\n':
        start += 1
    p = s.find('\n', start + 1)
    if p > -1:
        # Another newline found, continue until no more newlines or end of string
        while p < len(s) and s[p] == '\n':
            p += 1
        return p
    else:
        # No (more) newlines found, return length of string as end index
        return len(s)


def split_sentences_multi_until(article: str, fragments: list, start_index=0):
    """Split into sentences until sentence not covered with any label

    :param article:
    :param fragments:
    :param start_index:
    :param X:
    :param y:
    :return:
    """
    index = 0
    fragment_texts = [f.extract(article) for f in fragments]
    if fragments == []:
        # FIXME: to be implemented
        print("FIXME:  not dealing with empty fragments")
        assert False, "Should never get here with empty fragments"
    else:
        new_start = 0
        max_end_so_far = 0
        while index < len(fragments):
            max_end_so_far = max(max_end_so_far, fragments[index].end)
            new_start = next_nl_or_end(article, max_end_so_far)
            if index + 1 < len(fragments) and new_start > fragments[index+1].start:
                index += 1
            else:
                break
        start_text = article[:new_start]
        rest_text = article[new_start:]
        start_fragments = offset_fragments(fragments[:index+1], start_index)
        rest_fragments = offset_fragments(fragments[index+1:], new_start)

        start_fragment_texts = [f.extract(start_text) for f in start_fragments]

        for t in start_fragment_texts:
            if t not in fragment_texts:
                found = False
                for ft in fragment_texts:
                    if ft.startswith(t):
                        found = True
                        print(f"+++ MULTI UNTIL Output fragment {[t]} is truncated \n--- from {[ft]}")
                        break
                if not found:
                    print(f"=== MULTI UNTIL Got output level fragment text {str([t])} not in input fragments ", fragment_texts,
                          ' ** total output is ', start_fragment_texts)
                    print("All original ", len(fragments), "fragments: ", fragments) # , "original text: ", article)


        return start_text, start_fragments, rest_text, rest_fragments, new_start


def preceding_non_labelled_sentences(id: str, article: str, fragments: list):
    if fragments == []:
        return [], 0
    else:
        # fragments are sorted by start field, get the first. Fragment is a (start,end,p_type) tuple
        first_index = fragments[0].start
        if first_index > 0:
            substr = article[:first_index]
            if substr.find('\n') > 0:
                r = substr.rindex('\n')
                # Note: here we remove any newlines from after sentences,
                # where elsewhere they are left.
                # Reason for leaving them in is that this way fragment indices can be reconstructed.
                sentences = split_in_sentences(substr[:r+1]) #[x for x in substr[:r].split('\n') if x != '']
                return sentences, r+1
            else:
                return [], 0
        else:
            return [], 0


def merge_short_sentences(lst: list) -> list:
    """Merge subsequent sentences that consist of only a single word."""
    last_is_short = False
    out_lst = []
    for s in lst:
        if s.find(' ') == -1:
            if last_is_short:
                out_lst[-1] = out_lst[-1] + s
            else:
                last_is_short = True
                out_lst.append(s)
        else:
            last_is_short = False
            out_lst.append(s)
    return out_lst


def split_in_sentences(s):
    """Split into sentences, without loosing newlines. Original newlines remain after each sentence"""
    if s == '':
        return []
    elif s.find('\n') == -1:
        return [s]
    else:
        if s[-1] == '\n':
            i = len(s) -1
            while s[i] == '\n':
                i -= 1
            if not s[:i].find('\n'):
                # Only closing newlines, return s as one sentence
                return [s]
            else:
                j = s[:i].rfind('\n')
                lst = split_in_sentences(s[:j+1])
                rest = s[j+1:]
                lst.append(rest)
        else:
            j = s.rfind('\n')
            lst = split_in_sentences(s[:j + 1])
            rest = s[j + 1:]
            lst.append(rest)
        return lst


def surrounding_word(s: str, index: int, with_line_end=False):
    """Find the start and end index of the word around the current index.

    :param s String in which we seek word
    :param index Current index within the string, identifying a position within the sought word
    :param with_line_end If set, a trailing line end character will be included with the word

    Returns: (start, end) of word identified by 'index' or None if index is not a position in a word."""
    is_word = re.compile(r'\w')
    #print(f' s=[{s}], index={index}')
    if index >= len(s):
        index = len(s) - 1

    if is_word.match(s[index]) is None:
        return None
    else:
        start = index
        end = index
        # 'start' is inclusive, that is, index of part of the word
        while start - 1 > 0 and is_word.match(s[start - 1]):
            start -= 1
        # 'end' is exclusive, that is, its index is NOT part of the word
        while end < len(s) and (is_word.match(s[end]) or (with_line_end and s[end] in ['.', '?', '!'])):
            end += 1
        return start, end


def calibrate(fragment: Fragment, match_text: str, orig_text:str, distance=3) -> Fragment:
    """Calibrates a fragment to another text, assuming it to be nearly right.

    Effectively, it moves a fragment a short distance to better match the original text"""
    try:
        fragment_text = fragment.extract(match_text).lower()

        #print(f'INPUT FRAGMENT = [{fragment_text}]')

        first_word_re = re.compile(r'\A\W*(\w+)\W.*', re.MULTILINE | re.DOTALL)
        # Include trailing end of line interpunction
        last_word_re = re.compile(r'.*\W(\w+[!.?]?)\W*\Z', re.MULTILINE | re.DOTALL)

        m = first_word_re.match(fragment_text+' ')
        assert m is not None, f"First word matching failed for [{fragment_text} ] for {fragment}"
        first_word = m.group(1)

        # Deal with aberrant single letter words, skip them unless they are 'i' or 'a'
        # Commented out: this had a negative effect
        if len(first_word) == 1 and first_word not in ['i', 'a']:
            fragment.start += 1
            fragment_text = fragment_text[1:]
            m = first_word_re.match(fragment_text+' ')
            assert m is not None, f"First word matching failed (b) for [{fragment_text} ] for {fragment}"
            first_word = m.group(1)

        n = last_word_re.match(' ' + fragment_text)
        assert n is not None, f"Last word matching failed for [ {fragment_text}] for {fragment}"
        last_word = n.group(1)

        have_set_first_word = False
        have_set_last_word = False

        start, end = fragment.start, fragment.end
        startpos = max(start - distance, 0)
        for i in range(startpos, start + distance):
            if orig_text[i:].lower().startswith(first_word):
                start = i
                have_set_first_word = True
                break

        endpos = min(end + distance, len(orig_text))
        for i in range(end - distance, endpos):
            if orig_text[:i].lower().endswith(last_word):
                end = i
                have_set_last_word = True
                break

        if not have_set_first_word:
            res = surrounding_word(orig_text, start)
            if res is None:
                print("starting in empty space")
            else:
                start = res[0]
        if not have_set_last_word:
            res = surrounding_word(orig_text, end, with_line_end=True)
            if res is None:
                print("ending in empty space")
            else:
                end = res[1]

        fragment.start = start
        fragment.end = end

        # print(f'OUTPUT FRAGMENT = [{fragment.extract(orig_text)}]')

        return fragment
    except AssertionError as e:
        print(e)
        print(f'Failed to calibrate: returning original fragment {fragment}')
        return fragment


def fragment_train_test_split(data: list, labels: list, test_size=0.2, shuffle=True):
    """Treating the data as multilabel to do iterative train_test_split

    arguments:
      data: list of objects from the `Fragment` dataclass
      labels: list of used labels
      text_size: part of the data to be used as `test` set
      shuffle: if True, shuffle the dataset before splitting.
    """
    def label_to_int(l, labels):
        return labels.index(l)

    if shuffle:
        random.shuffle(data)

    # For train_test_split we use an X containing only the index of the data items
    X = [[index] for index in range(len(data))]
    y = []
    num_labels = len(labels)
    # One hot encoding of multiple categories
    for d in data:
        label_list = list({label_to_int(f.label, labels) for f in d['fragments']})
        y.append([1 if l in label_list else 0 for l in range(num_labels)])

    nX = np.array(X)
    ny = np.array(y)

    # We use iteractive_train_test_split to split with an even division for
    # labels.
    #
    # NOTE!
    # multilearn returns for iterative_train_test_split are:
    #   X_train, y_train, X_test, y_test
    # unlike sklearn train_test_split which returns
    #   X_train, X_test, y_train, y_test
    X_train, _, X_test, _ = iterative_train_test_split(nX, ny, test_size=test_size)
    X_train = np.squeeze(X_train)
    X_test = np.squeeze(X_test)
    print("Train:", X_train.shape, "Test:", X_test.shape)

    train = []
    test = []
    for key in list(X_train):
        train.append(data[key])
    for key in list(X_test):
        test.append(data[key])

    return train, test


########## Score ##########################################


def compute_prec_rec_f1(prec_numerator, prec_denominator, rec_numerator, rec_denominator, print_results=False):
    logger.debug("P=%f/%d, R=%f/%d"%(prec_numerator, prec_denominator, rec_numerator, rec_denominator))
    p = r = f1 = 0
    if prec_denominator > 0:
        p = prec_numerator / prec_denominator
    if rec_denominator > 0:
        r = rec_numerator / rec_denominator
    if print_results:
        logger.debug("Precision=%f/%d=%f\tRecall=%f/%d=%f" % (
            prec_numerator, prec_denominator, p, rec_numerator, rec_denominator, r))
    if prec_denominator == 0 and rec_denominator == 0:
        f1 = 1.0
    if p > 0 and r > 0:
        f1 = 2 * (p * r / (p + r))
    if print_results:
        logger.info("F1=%f" % (f1))
    return p, r, f1


def calc_c(s, t, h):
    """
    :param s: Fragment prediction
    :param t: Fragment ground truth
    :param h: Normalizing factor
    :return:
    """
    intersection = s & t
    if intersection is None or s.label != t.label:
        return 0
    else:
        return len(intersection) / h


def count_fragments(data, label=None):
    if label is None:
        return len([f for d in data for f in d['fragments']])
    else:
        count = 0
        for d in data:
            count += len([f for f in d['fragments'] if f.label == label])
        return count


def compute_FLC_score(predictions, ground_truths, labels, per_article_evaluation=False):
    total_fragments_prediction = 0
    total_fragments_ground = 0

    cumulative_label_precision = {lab: 0 for lab in labels}
    cumulative_label_recall = {lab: 0 for lab in labels}

    cumulative_precision = 0
    cumulative_recall = 0

    f1_articles = []

    for pred, ground in zip(predictions, ground_truths):
        count_article_fragments_prediction = len(pred['fragments'])
        count_article_fragments_ground = len(ground['fragments'])

        total_fragments_prediction += count_article_fragments_prediction
        total_fragments_ground += count_article_fragments_ground

        article_cumulative_precision = 0
        article_cumulative_recall = 0
        for s_fragment in pred['fragments']:
            for t_fragment in ground['fragments']:
                precision = calc_c(s_fragment, t_fragment, len(s_fragment))
                recall = calc_c(s_fragment, t_fragment, len(t_fragment))

                article_cumulative_precision += precision
                article_cumulative_recall += recall

                if s_fragment.label == t_fragment.label:
                    # if there is no overlap, e.g. because there are several spans with the label, p and r will be 0
                    cumulative_label_precision[s_fragment.label] += precision
                    cumulative_label_recall[s_fragment.label] += recall

        cumulative_precision += article_cumulative_precision
        cumulative_recall += article_cumulative_recall

        p_article, r_article, f1_article = compute_prec_rec_f1(
            article_cumulative_precision,
            count_article_fragments_prediction,
            article_cumulative_recall,
            count_article_fragments_ground,
            print_results=False)

        f1_articles.append(f1_article)

    p, r, f1 = compute_prec_rec_f1(
        cumulative_precision, total_fragments_prediction,
        cumulative_recall, total_fragments_ground
    )

    if per_article_evaluation:
        logger.info("Per article evaluation F1=%s" % (",".join([str(f1_value) for f1_value in f1_articles])))

    f1_per_technique = []
    for label in labels:
        precision_for_label, recall_for_label, f1_for_label = compute_prec_rec_f1(
            cumulative_label_precision[label],
            count_fragments(predictions, label=label),
            cumulative_label_recall[label],
            count_fragments(ground_truths, label=label),
            False)
        f1_per_technique.append(f1_for_label)
        logger.debug("%s: P=%f R=%f F1=%f" % (label, precision_for_label, recall_for_label, f1_for_label))

    return p, r, f1, f1_per_technique


def FLC_score(predictions: list, ground_truths: list, labels: list):
    """Fragment Level Classification score of prediction compared to ground_truth.

    See: See: https://propaganda.qcri.org/semeval2020-task11/data/propaganda_tasks_evaluation.pdf

    Args:
        predictions: list of Fragments
        ground_truths: list of Fragments
        labels: list of labels for classes
    """
    precision, recall, f1, f1_per_label = compute_FLC_score(predictions, ground_truths, labels, per_article_evaluation=False)
    res_for_screen = "\nF1=%f\nPrecision=%f\nRecall=%f\n%s\n" % (
        f1,
        precision,
        recall,
        "\n".join(["F1_" + pr + "=" + str(f) for pr, f in zip(labels, f1_per_label)])
    )
    f1_for_labels = {labels[i]: f1_per_label[i] for i in range(len(labels))}
    res_for_script = {'f1': f1, 'precision': precision, 'recall': recall, 'f1_for_labels': f1_for_labels}

    return res_for_screen, res_for_script


########## Data Exploration ###############################

def statistics(data: list, name='<no name>') -> dict:
    """Dictionary with statistics for a list of articles with their fragments"""

    def word_count(s):
        """Count the number of words in a text"""
        # remove interpunction
        s = s.replace(',', ' ').replace('-', ' ').replace('(', ' ').\
            replace(')', ' ').replace('.', ' ').replace('!', ' ').replace('?', ' ').replace('\n', ' ')
        return len([w for w in s.split(' ') if w.strip() != ''])

    def sentence_count(s):
        """Count the number of sentences in a text"""
        return len([t for t in s.split('\n') if t.strip() != ''])

    def upperlower_ratio(s):
        """Number of uppercase characters divided
        by the sum of the number of uppercase characters and number of lowercase characters"""
        upper = sum(1 for c in s if c.isupper())
        lower = sum(1 for c in s if c.islower())
        return upper / (upper + lower)

    output = {'name': name}
    fragments = [(f, f.extract(d['article'])) for d in data for f in d['fragments']]
    lengths = [len(f[0]) for f in fragments]
    word_counts = [word_count(t[1]) for t in fragments]
    sentence_counts = [sentence_count(t[1]) for t in fragments]
    upperlower = [upperlower_ratio(t[1]) for t in fragments]
    # Now we get some measures
    output['fragments'] = len(fragments)
    if lengths:
        output['average length'] = mean(lengths)
        output['median length'] = median(lengths)
    else:
        output['average length'] = 0.0
        output['median length'] = 0.0

    if word_counts:
        output['average wordcount'] = mean(word_counts)
        output['median wordcount'] = median(word_counts)
    else:
        output['average wordcount'] = 0.0
        output['median wordcount'] = 0.0

    if sentence_counts:
        output['average sentencecount'] = mean(sentence_counts)
        output['median sentencecount'] = median(sentence_counts)
        output['longer than 1 sentence'] = sum([1 for x in sentence_counts if x != 1])
    else:
        output['average sentencecount'] = 0.0
        output['median sentencecount'] = 0.0
        output['longer than 1 sentence'] = 0

    if upperlower:
        output['upper/lower ratio'] = mean(upperlower)
    else:
        output['upper/lower ratio'] = 0.0

    return output


def data_overview(data, labels=None):
    """Create pandas DataFrame with overview of data, total and per label"""
    def filter_by_label(data, label):
        """"Retain only the fragments for the given label"""
        filtered_data = []
        for d in data:
            d_for_label = {
                'article': d['article'],
                'id': d['id'],
                'fragments': [f for f in d['fragments'] if f.label == label]
            }
            filtered_data.append(d_for_label)
        return filtered_data

    if labels is None:
        labels = list({f.label for d in data for f in d['fragments']})

    stats = [statistics(data, name='Total')]  # stats on entire dataset
    for l in labels:
        data_for_label = filter_by_label(data, l)
        d = statistics(data_for_label, name=l)  # stats per label
        stats.append(d)

    return pd.DataFrame(stats)


def analyse_prediction(input_text, correct_text, raw_prediction, processed_prediction, labels):
    def check_current_balance(balance, label_index, labels):
        start_markers, end_markers = balance[label_index]
        if end_markers > start_markers:
            d_str = f'closing marker before opening marker for label ' \
                        f'{labels[label_index]}/{label_index}: {str(balance[label_index])}'
            return d_str

    def count_tags(s):
        start_re = re.compile(r'\[i-(\d+)\]')
        end_re = re.compile(r'\[o-(\d+)\]')
        start_or_end = re.compile(r'\[([io])-(\d+)\]')
        start_tags = start_re.findall(s)
        #print('start tags', start_tags)

        end_tags = end_re.findall(s)
        #print('end tags', end_tags)

        all_tags = start_or_end.findall(s)

        balance = {label_index: (0, 0) for label_index in range(len(labels))}
        errs = []
        for t in all_tags:
            i_o = t[0]
            label_index = int(t[1])

            cur = balance[label_index]
            if i_o == 'i':
                balance[label_index] = (cur[0] + 1, cur[1])
            else:
                balance[label_index] = (cur[0], cur[1] + 1)

            e_str = check_current_balance(balance, label_index, labels)
            if e_str is not None:
                errs.append(e_str)

        return errs

    count_tags_res = count_tags(raw_prediction)
    return count_tags_res


########## Filters #########################################
"""
In configuration, filters are defined under 'data' and presented as a list.
The 'filter_' prefix is not presented, it serves only to clarify here in code what kind of functions these are.
"""

def check_change(func):
    """Decorator for filters: logs change in number of items"""
    def wrapper_check_change(data):
        length_before = len(data)
        output_data = func(data)
        length_after = len(output_data)
        if length_before == length_after:
            msg = f'Number of items unchanged'
        elif length_before > length_after:
            msg = f'removed {length_before - length_after} to new total of {length_after} items'
        else:
            msg = f'added {length_after - length_before} to new total of {length_after} items'
        logger.info(f'{func.__name__}: {msg}')
        return output_data
    return wrapper_check_change


@check_change
def filter_sentence_splitter(data):
    """
    Split articles into sets of sentences covering fragments.


    :param data:
    :return: data
    """
    new_data = []
    for d in data:
        id = d['id']
        article = d['article']
        fragments = d['fragments']
        lst = split_sentences_multi(id, article, fragments, include_empty=False)
        new_data.extend(lst)
    data = new_data
    return data

@check_change
def filter_sentence_splitter_with_empty(data):
    """
    Split articles into sets of sentences covering fragments.


    :param: data
    :return: data
    """
    new_data = []
    for d in data:
        id = d['id']
        article = d['article']
        fragments = d['fragments']
        lst = split_sentences_multi(id, article, fragments, include_empty=True)
        new_data.extend(lst)
        # If the article length is small, also include the original article
        #if len(article) < 400 and len(lst) > 1:
        #    new_data.append(d)
    data = new_data
    return data


@check_change
def filter_eliminate_short(data):
    """Filter out items with too few characters as they are unlikely to represent a technique"""
    MIN_CHARACTERS = 10
    new_data = []
    for d in data:
        if len(d['article']) > MIN_CHARACTERS:
            new_data.append(d)
    data = new_data
    return data


@check_change
def filter_eliminate_long(data):
    """Filter out items with too many characters as they are unlikely to represent a technique"""
    MAX_CHARACTERS = 300
    new_data = []
    for d in data:
        if len(d['article']) <= MAX_CHARACTERS:
            new_data.append(d)
    data = new_data
    return data


@check_change
def filter_lowercase(data):
    """Converts text of all items to lowercase"""
    new_data = []
    for d in data:
        d['article'] = d['article'].lower()
        new_data.append(d)
    data = new_data
    return data


@check_change
def filter_duplicate_fragments(data):
    """
    Extract fragments from sentences and add them to the dataset.

    Standard data contains (sets of) sentences covering fragments.
    Now we add the fragments phrases as independent data items.
    :return:
    """
    MIN_CHARS = 10
    added_data = []
    skipped = 0
    for d in data:
        fragments = d['fragments']
        for index, (start, end, p_type) in enumerate(fragments):
            add_id = d['id'] + '_' + str(index)
            add_article = d['article'][start:end]
            add_fragments = [(0, len(add_article), p_type)]
            if len(add_article) > MIN_CHARS:
                added_data.append({'id': add_id, 'article': add_article, 'fragments': add_fragments})
            else:
                skipped += 1

    if skipped > 0:
        logger.info(f'duplicate_fragments: skipped {skipped} fragments as they are shorter than {MIN_CHARS} chars')
    data = data + added_data
    return data


