# Unsupervised text style transfer with monolingual data using back translation

This project is an attempt at text style transfer and machine translation (work in progress) using only monolingual dataset (non parallel sentences). This project is strongly inspired from papers like "Style Transfer from Non-Parallel Text by Cross-Alignment" and "Usupervised Machine Translation Using Monolingual Corpora Only" among other related works. Before we jump into the intro to the project and usage, I would like to emphasize on the fact that this project is not to be considered as an ready-to-use library for translation problems (at least for now). It is a personal project which I work on after my day job, as such, I have not yet tried it on variety of problems or benchmarks.

The main script is "train.py" which reads and preprocesses data, saves the computed artifacts like vocabulary, tokens, tensors for sentences etc and finally, trains the model. You can use "eval.py" script to evaluate a saved model from any of the runs/experiments.

You can the supported arguments and their explanations using :
```
python train.py -h
```

and same for eval.py as well. I will skip the full list for brevity. Some notable arguments for train.py are input source file (file with source type sentences in each new line), input target file, source embedding, target embedding, experiment id/name etc. All are optional though.

### TODO
* Test the script for machine translation with FastText aligned word embeddings
* model performance on longer and complicated sentence is bad.
* try better architectures (may be Transformers?)

P.S. This document will updated gradually with more info.
