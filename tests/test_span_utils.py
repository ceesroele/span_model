import pickle

from span.span_utils import (
    preceding_non_labelled_sentences,
    next_nl_or_end,
    split_sentences_multi_until,
    split_sentences_multi,
    split_in_sentences,
    merge_short_sentences,
    calibrate,
    encode,
    decode,
    surrounding_word,
    Fragment,
    FLC_score
)

article = 'abcd\n1234\npqrs'
article_extra = 'abcd\n\n1234\n\npqrs'

f1 = Fragment(1, 2, 'f1')
f2 = Fragment(6, 8, 'f2')
f3 = Fragment(11, 13, 'f3')

article_2 = 'aaaaa\n\nbbbbb\nccccc\n\n01234'
fragments_2 = [
    Fragment(2, 6, 'foo'),
    Fragment(4, 8, 'bar'),
    Fragment(21, 23, 'foo')
]


def test_next_nl_or_end():
    s = 'abcd\n\npqrs'
    n = next_nl_or_end(s)
    assert n == 6
    s = 'abcd\n\npqrs\n'
    n = next_nl_or_end(s, 4)
    assert n == 11
    s = 'abcd\n\npqrs'
    n = next_nl_or_end(s, 8)
    assert n == 10
    s = 'abcd\n\npqrs\n\nvwxyz'
    n = next_nl_or_end(s, 4)
    assert n == 12
    assert s[:n] == 'abcd\n\npqrs\n\n'
    assert s[n:] == 'vwxyz'

    n2 = next_nl_or_end(article_2, 9)
    assert n2 == 13


def test_preceding_non_labelled_sentences():
    id = '111'
    section = f2.extract(article_extra)
    sentences, start = preceding_non_labelled_sentences(id, article_extra, [f2])
    new_article = article_extra[start:]
    assert sentences == ['abcd\n\n']
    assert new_article == '1234\n\npqrs'
    assert start == 6
    section_after = (f2 - start).extract(article_extra[start:])
    assert section == section_after


def test_split_sentences_multi_until():
    id = '111'
    start_text, start_fragments, continuation_text, rest_fragments, new_start = \
        split_sentences_multi_until(article_2, fragments_2, start_index=0)

    assert start_text == 'aaaaa\n\nbbbbb\n'
    assert start_fragments == [Fragment(2, 6, 'foo'), Fragment(4, 8, 'bar')]
    assert continuation_text == 'ccccc\n\n01234'
    assert rest_fragments == [Fragment(8, 10, 'foo')]
    assert new_start == 13
    assert article_2[new_start] == 'c'
    assert article_2[new_start-1] == '\n'



def test_split_sentences_multi():
    id = '111'
    part = fragments_2[2].extract(article_2)
    assert part == '12'

    data = split_sentences_multi(id, article_2, fragments_2, include_empty=True) #, start_index=0, data=[])

    print(data)

    assert data[0] == {'id': '111x', 'article': 'aaaaa\n\nbbbbb\n', 'fragments': [Fragment(2, 6, 'foo'), Fragment(4, 8, 'bar')]}

    assert data[1] == {'id': '111_0p0', 'article': 'ccccc\n\n', 'fragments': []}

    #assert get_fragment(data[2]['article'], data[2]['fragments'][0]) == '12'
    assert data[2] == {'id': '111_0x', 'article': '01234', 'fragments': [Fragment(1, 3, 'foo')]}

    # Add a fragment that starts after the previous fragments,
    # but starts before the end of the sentence included with the previous fragments.
    fragments_2.append(Fragment(9, 10, 'tolstoy'))
    fragments_2.sort()

    data = split_sentences_multi(id, article_2, fragments_2, include_empty=True)
    assert data[0] == {
        'id': '111x',
        'article': 'aaaaa\n\nbbbbb\n',
        'fragments': [Fragment(2, 6, 'foo'), Fragment(4, 8, 'bar'), Fragment(9, 10, 'tolstoy')]}

    assert data[1] == {'id': '111_0p0', 'article': 'ccccc\n\n', 'fragments': []}

    assert data[2]['fragments'][0].extract(data[2]['article']) == '12'
    assert data[2] == {'id': '111_0x', 'article': '01234', 'fragments': [Fragment(1, 3, 'foo')]}


def test_split_in_sentences():
    s = split_in_sentences('')
    assert s == []

    s = split_in_sentences('a\n')
    assert s == ['a\n']

    s = split_in_sentences('aaa\n')
    assert s == ['aaa\n']

    s = split_in_sentences('aaa\n\n')
    assert s == ['aaa\n\n']


    a = 'aaa\nbbb\n123'
    b = a + '\n'
    c = a + '\n\n'
    s = split_in_sentences(a)
    assert s == ['aaa\n', 'bbb\n', '123']

    s2 = split_in_sentences(b)
    assert s2 == ['aaa\n', 'bbb\n', '123\n']

    s3 = split_in_sentences(c)
    assert s3 == ['aaa\n', 'bbb\n', '123\n\n']


def test_calibrate():
    match_text = 'hello sad big world!'
    orig_text = 'goodbye sad big world!'
    fragment = Fragment(6, 13, 'foo')

    assert fragment.extract(match_text) == 'sad big'
    res = calibrate(fragment, match_text, orig_text, distance=3)
    print('new fragment = ', res)
    assert res == Fragment(8, 15, 'foo')

    match_text = ' hello sad big world!'
    orig_text = 'hello sad big world!'
    fragment = Fragment(1, 6, 'foo')

    assert fragment.extract(match_text) == 'hello'
    res = calibrate(fragment, match_text, orig_text, distance=3)
    print('new fragment = ', res)
    assert res == Fragment(0, 5, 'foo')

def test_surrounding_word():
    s = ' hello, world!'

    index = 10
    assert s[index] == 'r'
    start, end = surrounding_word(s, index)
    assert start == 8
    assert end == 13
    assert s[start:end] == 'world'

    index = 10
    assert s[index] == 'r'
    start, end = surrounding_word(s, index, with_line_end=True)
    assert start == 8
    assert end == 14
    assert s[start:end] == 'world!'

    index = 7
    assert s[index] == ' '
    res = surrounding_word(s, index)
    assert res is None

    index = 4
    assert s[index] == 'l'
    start, end = surrounding_word(s, index)
    assert start == 1
    assert end == 6
    assert s[start:end] == 'hello'



def test_decode():
    all_labels = ['one', 'two']
    s = "abc [i-0] def [o-0] ghi [i-1] jk [o-1]"
    article, fragments, errors = decode(s, all_labels)
    print(article)
    assert len(article) == 14
    assert len(fragments) == 2
    assert fragments[0].extract(article) == 'def'
    assert fragments[1].extract(article) == 'jk'

    all_labels1 = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'elven',
                   'twelves', 'thirteen', 'fourteen','fifteen','sixteen', 'seventeen']
    s1 = 'How to [i-1] vaccinate [o-1], ahem test'
    article1, fragments1, errors1 = decode(s1, all_labels1)
    print(article1)
    assert len(article1) == 27
    assert len(fragments1) == 1
    print(fragments1[0])
    assert fragments1[0].extract(article1) == 'vaccinate'

    s2 = '[i-16]  Got 195,000 + Americans [i-8] KILLED [o-8].\nFought scientists and doctors'
    article2, fragments2, errors2 = decode(s2, all_labels1)
    print(article2)
    assert len(article2) == 62
    assert len(fragments2) == 1
    print(fragments2[0])
    assert fragments2[0].extract(article2) == 'KILLED'



def test_merge_short_sentences():
    s = "\"LAW AND ORDER\"\n\nPLEADED GUILTY\n\nCONVICTED\n\nCONVICTED\n\nCONVICTED\n\nCONVICTED\n\nINDICTED"
    lst = split_in_sentences(s)

    out_lst = merge_short_sentences(lst)
    print('outlst', out_lst)

    assert len(out_lst) == 3

    assert out_lst[2] == 'CONVICTED\n\nCONVICTED\n\nCONVICTED\n\nCONVICTED\n\nINDICTED'

def test_find_all_labels_matching_or_not():
    s = ""


def test_FLC_score():
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
    with open('tests/ground_truths_dev_task6_subtask2.pkl', 'rb') as f:
        ground_truths = pickle.load(f)
    with open('tests/predictions_dev_task6_subtask2.pkl', 'rb') as f:
        predictions = pickle.load(f)

    for_screen, for_script = FLC_score(predictions, ground_truths, LABELS_2021, output_for_script=True)
    lst = [float(x) for x in for_script.split('\t')]

    f1 = lst[0]
    precision = lst[1]
    recall = lst[2]

    assert f1 == 0.390338
    assert precision == 0.555802
    assert recall == 0.300792

    index = LABELS_2021.index('Loaded Language')

    f1_label = lst[3 + index]
    assert f1_label == 0.5491378695017832


