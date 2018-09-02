# Perceptron dependency parser
A graph-based dependency parser, trained on linguistic features with the (averaged) structured-perceptron.
This project is an mash-up of the following sources of ingredients:

### Linguistic features
These I improvised. Probably loads of room for improvement, but an extremely basic feature set already performs remarkably well (just dep-head pairs without any context, but with a distance feature). See [features.py](features.py) for the basic feature set.

For the Universal Dependencies dataset we can also make use of the `lemma` and `feats` fields, but I haven't come around to this yet.

### Structured perceptron algorithm
Inspired by and partially based on the spaCy blog posts [Parsing English in 500 Lines of Python](https://explosion.ai/blog/parsing-english-in-python) and [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

### Graph based dependency parsing
Dependecy parsing with an MST algorithm from [McDonald et al. 2006](https://www.seas.upenn.edu/~strctlrn/bib/PDF/nonprojectiveHLT-EMNLP2005.pdf), and the training objective of [Dozat and Manning 2017](https://arxiv.org/pdf/1611.01734.pdf) (for each token predict a head).

## Usage
For now we assume you have the PTB in train/dev/test splits stored somewhere in one directory and that they are named `train.conll`, `dev.conll`, `test.conll`.

To train the perceptron for 5 epochs, type:
```bash
./main.py train --data path/to/ptb/dir --epochs 5
```
The training can be halted at any point with `cntrl-c`. The trained model and feature-set are saved at `models/model.pkl` resp. `models/features.pkl` by default. To specify these paths use `--model path/to/model.pkl` resp. `--features path/to/features.pkl`.

To train the perceptron for 5 epochs with already extracted features, type:
```bash
./main.py train --data path/to/ptb/dir --epochs 5 --features path/to/features
```

To evaluate the trained perceptron, type:
```bash
./main.py eval --data path/to/ptb/dir
```

To plot heatmaps of the predicted score matrices for five sentences in the dev set (like those in [image](image)) type:
```bash
./main.py plot --data path/to/ptb/dir
```

## Speed
Making the full feature set for the training set (~11 million for the basic features) takes about 5 minutes. One epoch with these features on the training set takes around 8 minutes.

## Accuracy
No full results yet.

Training UAS after 5 epochs is around 70. Dev UAS then is around 50.

## TODO
- [ ] Predict labels. Maybe a second perceptron altogether for that?
- [ ] Make integration with Universal Dependencies easier. (Now only using conllx format)
- [ ] Understand which features matter.
- [ ] Perform full training till convergence.
