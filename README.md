# SpanModel

A sequence-to-sequence model for simultaneously identifying a span within a text and classifying it.



## Installation

```buildoutcfg
Create a conda environment:

conda create --name "semeval2021-task2"

Activate it:
conda activate semeval2021-task2

Install python in the conda environment:
# Use python version 3.8 as 3.9 is not supported by torchvision
conda install python=3.8

Install python packages:
pip install -r requirements.txt


Run it with:
python SemEval-2021-task6-subtask2.py

Outcome will be the generation of a submission file as used in the SemEval task.
```