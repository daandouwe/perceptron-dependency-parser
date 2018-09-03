# Perceptron dependency parser
A graph-based dependency parser, trained on linguistic features with the (averaged) structured-perceptron.
This project is an mash-up of the following ingredients:

### Graph based dependency parsing
Dependecy parsing with an MST algorithm from [McDonald et al. 2006](https://www.seas.upenn.edu/~strctlrn/bib/PDF/nonprojectiveHLT-EMNLP2005.pdf), and the training objective of [Dozat and Manning 2017](https://arxiv.org/pdf/1611.01734.pdf) (for each token predict a head).

### Structured perceptron algorithm
Inspired by and partially based on the spaCy blog posts [Parsing English in 500 Lines of Python](https://explosion.ai/blog/parsing-english-in-python) and [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

### Linguistic features
The feature-set is largely taken from [McDonald et al. 2005](https://www.seas.upenn.edu/~strctlrn/bib/PDF/dependencyACL2005.pdf). For the Universal Dependencies dataset we can also make good use of the `lemma` and `feats` fields, but I haven't come around to this yet.

### Data handling
All code to do handle `conllu` and `conllx` files is taken from [bastings](https://github.com/bastings/parser/tree/extended_parser) parser.

## Usage
For now we assume you have the PTB in train/dev/test splits in conll-format, stored somewhere in one directory, and that they are named `train.conll`, `dev.conll`, `test.conll`. For later we will to include a data script that downloads some of the Universal Dependencies languages, so we don't have this manual step.

To train the perceptron for 5 epochs, type:
```
./main.py train --data path/to/ptb/dir --epochs 5
```
The training can be halted at any point with `cntrl-c`. The trained model and feature-set are saved at `models/model.pkl` resp. `models/features.pkl` by default. To specify these paths use `--model` resp. `--features`.

To train the perceptron for 5 epochs with already extracted features, type:
```
./main.py train --data path/to/ptb/dir --epochs 5 --features path/to/features
```

To evaluate the trained perceptron, type:
```
./main.py eval --data path/to/ptb/dir
```

To plot heatmaps of the predicted score matrices for five sentences in the dev set (like those in [image](image)) type:
```
./main.py plot --data path/to/ptb/dir
```

## Features
The implementation lets you choose between a basic, and more rich feature-set.

The basic features are all of the followinf form:
```
head dep pos pos=VBN VBZ
head dep word word=is a
head dep pos word=VBN have
head dep suffix suffix=ing is
head dep shape shape=Xxxx dd/dd/ddd
head dep shape shape=xxxx xx
```
With shape inspired by spaCy's `token.shape` feature. This feature-set has no positional, context or otherwise sentence-level features.

Optionally you can add distance:
```
head dep pos pos=VBN have (-1)
head dep word word=is a (1)
```
With `(-1)` indicating the linear distance from the head to the dependent. This is a cheap way of giving some sentence-level information to the model.

Optionally you can add left and right surrounding pos tags for context:
```
head dep i i+1/i-1 i=DT NNS/VBZ VBG
```
with `i i+1` meaning the word itself and its right neighbor.

Finally there is an 'in-between' feature that finds all tags linearly in between head and dependent:
```
head between dep=DT JJ NNS (2 1)
```
With `(2 1)` indicating respectively the distance from head to between, and from between to dependent.

## Speed and size
Making the full feature set for the training set (~66 million for the basic features) takes about 14 minutes. One epoch with these features on the training set also takes around 15 minutes (40 sentences per second). After training, we prune the model by removing weights smaller than a certain threshold (1e-3 by default):
```
Pruning weights with threshold 0.001...
Number of weights: 66,475,707 (64,130,339 exactly zero).
Number of pruned weights: 2,343,424.
```
Due to the sheer enormity of the feature-set, the model saved model is still pretty big: ~140 MB!

## Accuracy
A fully converged training run (15 epochs) on the minimal feature-set gave the following results:
```
Train UAS 96.43
Dev UAS 81.98
Test UAS 81.58
```
Averaging the weights makes quite a difference on the dev-set: from 78.48 to 81.98.

## Interpretation
Fun fact one: The trained weights of the features are extremely interpretable. These are the largest ones:
```
head dep pos pos=VBN MD (-2) 32.0216
head dep pos pos=NNS NN (-1) 28.4403
head dep pos pos=NNS PRP$ (-2) 27.7557
head dep pos pos=VBN PRP (-2) 27.3990
head dep pos pos=VBN PRP (-3) 26.8825
head dep pos word=VB did (-2) 26.3881
head dep pos pos=VBN WDT (-2) 26.3819
head dep pos pos=VB MD (-1) 26.3348
head dep pos word=VBN be (-1) 26.2219
head dep pos pos=VBN MD (-3) 25.8771
head dep pos word=VB does (-2) 25.5444
head dep pos word=VBN have (-1) 25.0681
head dep pos word=VBN have (-2) 24.5689
head dep pos word=VBN has (-3) 24.5264
head dep pos pos=NNS CD (-1) 24.1591
head dep pos pos=VB NNS (1) 24.0786
head dep pos word=VBN had (-2) 23.9201
head dep pos pos=NNS JJ (-2) 23.8582
head dep pos pos=NNS DT (-1) 23.8089
head dep pos pos=VBD PRP (-1) 23.7837
head dep pos word=VB do (-2) 23.1994
head dep pos pos=VB PRP (-1) 23.1278
head dep pos pos=VBN VB (2) 23.0798
head dep pos pos=$ CD (1) 22.8169
head dep pos pos=NN NN (-1) 22.7906
head dep word pos=Inc NNP (-2) 22.7582
head dep pos pos=VB PRP (-2) 22.4573
head dep word pos=Inc. NNP (-2) 22.3508
head dep pos pos=NNS PRP$ (-1) 22.2894
head dep pos pos=VBG PRP (-2) 22.0252
```
Fun fact two: We can make some nifty [heatmaps](image) out of the score matrices.

## Requirements
```
python>=3.6.0
numpy
matplotlib
tqdm
```

## TODO
- [ ] Predict labels. Maybe a second perceptron altogether for that?
- [ ] Make integration with Universal Dependencies easier. (Now only using conllx format)
- [ ] Make data loading less name-dependent.
- [ ] Understand which features matter.
- [X] Perform full training till convergence.
- [ ] Make training parallel ('hogwild'). Really easy, and perhaps even some regularization.
- [X] Prune the averaged weights by removing all features that are exactly 0.
