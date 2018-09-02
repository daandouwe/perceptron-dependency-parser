# Perceptron dependency parser
A graph-based dependency parser, trained on linguistic features with the (averaged) structured-perceptron.
This project is an mash-up of the following sources of ingredients:

### Graph based dependency parsing
Dependecy parsing with an MST algorithm from [McDonald et al. 2006](https://www.seas.upenn.edu/~strctlrn/bib/PDF/nonprojectiveHLT-EMNLP2005.pdf), and the training objective of [Dozat and Manning 2017](https://arxiv.org/pdf/1611.01734.pdf) (for each token predict a head).

### Structured perceptron algorithm
Inspired by and partially based on the spaCy blog posts [Parsing English in 500 Lines of Python](https://explosion.ai/blog/parsing-english-in-python) and [A Good Part-of-Speech Tagger in about 200 Lines of Python](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python).

### Linguistic features
These I improvised. Probably loads of room for improvement, but an extremely basic feature set already performs remarkably well (just dep-head pairs without any context, but with a distance feature). See [features.py](features.py) for the basic feature set. For the Universal Dependencies dataset we can also make use of the `lemma` and `feats` fields, but I haven't come around to this yet.

### Data handling
All code to do handle `conllu` and `conllx` files is taken from [bastings](https://github.com/bastings/parser/tree/extended_parser) parser.

## Usage
For now we assume you have the PTB in train/dev/test splits in conll-format, stored somewhere in one directory, and that they are named `train.conll`, `dev.conll`, `test.conll`. For later I want to include a data script that downloads some of the Universal Dependencies languages, so you don't have this manual step.

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
No fully converged results yet, but after 6 epochs, training UAS is around 75, and dev UAS is around 71.
(Averaging the weights has a huge impact on dev UAS: from 61 to 75!)

## Interpretation
Fun fact one: The trained weights of the features are extremely interpretable. These are the largest ones:
```
head dep pos=VBD . 24.5849
head dep pos=VBZ . 23.3874
head dep pos=VBN . 20.1195
head dep pos=VBP . 18.6637
head dep pos=VBN MD 16.6646
head dep word=interested in 15.2804
head dep shape=XXXXX . 15.2345
distance=1 15.1817
head dep pos=VBG . 15.0368
head dep pos=VBD VBD 14.9565
head dep pos=VB MD 14.7098
head dep pos=VB . 14.4930
distance=-1 14.4745
head dep pos=ROOT VBD 14.4062
head dep pos=VB TO 14.2382
head dep pos=VBN WDT 13.9254
head dep pos=VBD , 13.7076
head dep word=yielding , 13.5674
head dep pos=NNS PRP$ 13.5411
head dep shape=Xxxx.-xxxxx , 13.5285
head dep pos=VBD WRB 13.4672
head dep pos=VBN VBD 13.3486
head dep pos=ROOT VBZ 13.2660
head dep word=trading New 13.1774
head dep word=is thing 12.9891
head dep pos=VBZ VBZ 12.9506
head dep word=accused of 12.8340
head dep pos=VBD IN 12.7977
head dep pos=VBN IN 12.7394
head dep pos=NN PRP$ 12.6646
```
Fun fact two: We can make some nifty [heatmaps](image) out of the score matrices.

## Requirements
```
numpy
matplotlib
tqdm
```

## TODO
- [ ] Predict labels. Maybe a second perceptron altogether for that?
- [ ] Make integration with Universal Dependencies easier. (Now only using conllx format)
- [ ] Make data loading less name-dependent.
- [ ] Understand which features matter.
- [ ] Perform full training till convergence.
- [ ] Make training parallel ('hogwild'). Really easy, and perhaps even some regularization.
