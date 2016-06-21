# Machine translation with RNNs

Implementation of an LSTM encoder-decoder sequence-to-sequence translation model 
similar to the one described in "Sequence to Sequence Learning with Neural
Networks" by I. Sutskever, et al. 2014
([arXiv](https://arxiv.org/abs/1409.3215))

Implemented in Python using [Theano](https://github.com/Theano/Theano) and
[Keras](https://github.com/fchollet/keras).

<!--- ////////////////// -->

## Usage

###Translation
[...]

####Using precomputed weights
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


<!--- ////////////////// -->

## Model

RNN encoder-decoder with soft attention mechanism.
[...]
<!---
Insert clear explanation of model along with diagrams 
-->

