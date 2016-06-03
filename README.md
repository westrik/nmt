# Machine translation with RNNs

Implementation of the language model described in "Neural Machine Translation by
Jointly Learning to Align and Translate" by D. Bahdanau, K. Cho, and Y. Bengio.
([arXiv](http://arxiv.org/abs/1409.0473))

Implemented in Python using [Theano](https://github.com/Theano/Theano) and
[Lasagne](https://github.com/Lasagne/Lasagne).


## Model

RNN encoder-decoder with soft attention mechanism.
[...]
<!---
Insert clear explanation of model along with diagrams 
-->

## Usage

###Translation
[...]

#####Using precomputed weights
[...]


###Training
[...]


####Data preparation

While developing this model, I used the development data sets (news-test2012 and
news-test2013) from the [WMT14 shared task](http://www.statmt.org/wmt14/translation-task.html). 

Then to train, I used the 10^9 French-English corpus from the [WMT10 shared
task](http://www.statmt.org/wmt10/training-giga-fren.tar).

To test the model, I used the news-test2008 development set, also from the WMT14
shared task.

To tokenize the data, run this command on each corpus:

```
perl tools/normalize-punctuation.perl [en/fr] < data.[en/fr] 
    | perl tools/tokenizer.perl -l [en/fr] 
    > data.tok.[en/fr]
```

####Train the model
[...]
