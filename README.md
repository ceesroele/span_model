# SpanModel

A sequence-to-sequence model for simultaneously identifying a span within a text and classifying it.


## Installation

```
Create a conda environment:

conda create --name "semeval2021-task2"

Activate it:
conda activate semeval2021-task2

Install python in the conda environment:
# Use python version 3.8 as 3.9 is not supported by torchvision
conda install python=3.8

Install python packages:
pip install -r requirements.txt
```

## Example applications

There are two example applications.

### Detecting Toxic Spans

Task 5 of SemEval-2021.

```
python SemEval-2021-task5.py
```

Running this results in an average F1 score of 0.857 on the test set.


### Detecting Manipulation Techniques in Text

Task 6 subtask 2 of SemEval-2021.

```
python SemEval-2021-task6-2.py
```

## Future work

Deal with the two main causes of systemic errors:
1. Begin and end tags are not matching
2. Words or characters are introduced in the generated sentences that were not in the input

Ideas are:
* Train with half-masked sentences consisting only of begin and end tags (pre-training for tags)
* Add functionality to the generator code in Transformers to prevent tokens other than
tags in the output that are not in the input.