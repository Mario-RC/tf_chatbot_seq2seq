# Tensorflow chatbot
### (with seq2seq + attention + dict-compress + beam search)


> ####[Update 2017-03-14]
> 1. Included TensorBoard.   
> 2. Added configurations.
> 3. Removed anti-language model, reinforcement learning mdoe and facebook messenger server.
> 4. Better performance options for tensorflow-gpu v1.0.0.

> ####[Update 2019-07-24]
> 1. Upgrade to tensorflow v1.0.0, no backward compatible since tensorflow have changed so much.   
> 2. A pre-trained model with twitter corpus is added, just `./go_example` to chat! (or preview my [chat example](https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm/blob/master/example_chat.md))
> 3. You could start from tracing this `go_example` script to know how things work!

## Briefing
This is a [seq2seq model][a1] modified from [tensorflow example][a2].

1. The original tensorflow seq2seq has [attention mechanism][a3] implemented out-of-box.
2. And speedup training by [dictionary space compressing][a4], then decompressed by projection the embedding while decoding.
3. This work add option to do [beam search][a5] in decoding procedure, which usually find better, more interesting response.


[a1]: http://arxiv.org/abs/1406.1078
[a2]: https://www.tensorflow.org/versions/r0.10/tutorials/seq2seq/index.html
[a3]: http://arxiv.org/abs/1412.7449
[a4]: https://arxiv.org/pdf/1412.2007v2.pdf
[a5]: https://en.wikipedia.org/wiki/Beam_search
[a6]: https://arxiv.org/abs/1510.03055
[a7]: https://arxiv.org/abs/1606.01541
[a8]: http://flask.pocoo.org/
[a9]: https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm/blob/master/README2.md


## Just tell me how it works

#### Clone the repository

    git clone github.com/Mario-RC/tf_chatbot_seq2seq_antilm.git
    
#### Prepare for Corpus
You may find corpus such as twitter chat, open movie subtitle, or ptt forums from [this chat corpus repository][b1]. You need to put it under path like:

    tf_chatbot_seq2seq_antilm/works/<YOUR_MODEL_NAME>/data/train/chat.txt

And hand craft some testing sentences (each sentence per line) in:

    tf_chatbot_seq2seq_antilm/works/<YOUR_MODEL_NAME>/data/test/test_set.txt
    
#### Train the model

    python3 main.py --mode train --model_name <MODEL_NAME>
    
#### Run some test example and see the bot response

after you trained your model until perplexity under 50 or so, you could do:

    python3 main.py --mode test --model_name <MODEL_NAME>


**[Note!!!] if you put any parameter overwrite in this main.py commmand, be sure to apply both to train and test, or just modify in lib/config.py for failsafe.**


## Introduction

Seq2seq is a great model released by [Cho et al., 2014][c1]. At first it's used to do machine translation, and soon people find that anything about **mapping something to another thing** could be also achieved by seq2seq model. Chatbot is one of these miracles, where we consider consecutive dialog as some kind of "mapping" relationship.

Here is the classic intro picture show the seq2seq model architecture, quote from this [blogpost about gmail auto-reply feature][c2].

[![seq2seq][c3]][c3]


The problem is, so far we haven't find a better objective function for chatbot. We are still using [MLE (maximum likelyhood estimation)][c4], which is doing good for machine translation, but always generate generic response like "me too", "I think so", "I love you" while doing chat.

These responses are not informative, but they do have large probability --- since they tend to appear many times in training corpus. We don't won't our chatbot always replying these noncense, so we need to find some way to make our bot more "interesting", technically speaking, to increase the "perplexity" of reponse.

Here we reproduce the work of [Li. et al., 2016][c5] try to solve this problem. The main idea is using the same seq2seq model as a language model, to get the candidate words with high probability in each decoding timestamp as a anti-model, then we penalize these words always being high probability for any input. By this anti-model, we could get more special, non-generic, informative response.

The original work of [Li. et al][c5] use [MERT (Och, 2003)][c6] with [BLEU][c7] as metrics to find the best probability weighting (the **λ** and **γ** in
**Score(T) = p(T|S) − λU(T) + γNt**) of the corresponding anti-language model. But I find that BLEU score in chat corpus tend to always being zero, thus can't get meaningful result here. If anyone has any idea about this, drop me a message, thanks!


[c1]: http://arxiv.org/abs/1406.1078
[c2]: http://googleresearch.blogspot.ru/2015/11/computer-respond-to-this-email.html
[c3]: http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png
[c4]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation
[c5]: http://arxiv.org/pdf/1510.03055v3.pdf
[c6]: http://delivery.acm.org/10.1145/1080000/1075117/p160-och.pdf
[c7]: https://en.wikipedia.org/wiki/BLEU


## Parameters

There are some options to for model training and predicting in lib/config.py. Basically they are self-explained and could work with default value for most of cases. Here we only list something you  need to config:

**About environment**

name | type | Description
---- | ---- | -----------
mode | string | work mode: train/test/chat
model_name | string | model name, affects your working path (storing the data, nn_model, result folders)
scope_name | string | In tensorflow if you need to load two graph at the same time, you need to save/load them in different namespace. (If you need only one seq2seq model, leave it as default)
vocab_size | integer | depends on your corpus language: for english, 60000 is good enough. For chinese you need at least 100000 or 200000.
gpu_usage | float | tensorflow gpu memory fraction used, default is 1 and tensorflow will occupy 100% of your GPU. If you have multi jobs sharing your GPU resource, make it 0.5 or 0.3, for 2 or 3 jobs.

**About decoding**

name | type | default | Description
---- | ---- | ------- | -------
beam_size | int | 10 | beam search size, setting 1 equals to greedy search 

[d1]: http://arxiv.org/pdf/1510.03055v3.pdf


## Requirements

1. For training, GPU is recommended since seq2seq is a large model, you need certain computing power to do the training and predicting efficiently, especially when you set a large beam-search size.

2. DRAM requirement is not strict as CPU/GPU, since we are doing stochastic gradient decent.

3. If you are new to deep-learning, setting-up things like GPU, python environment is annoying to you, here are dockers of my machine learning environment:  
  [(non-gpu version docker)][e1]  /  [(gpu version docker)][e2]  

[e1]: https://github.com/Marsan-Ma/docker_mldm
[e2]: https://github.com/Marsan-Ma/docker_mldm_gpu


## References

Seq2seq is a model with many preliminaries, I've been spend quite some time surveying and here are some best materials which benefit me a lot:

1. The best blogpost explaining RNN, LSTM, GRU and seq2seq model: [Understanding LSTM Networks][f1] by Christopher Olah.

2. This work [sherjilozair/char-rnn-tensorflow][f2] helps me learn a lot about language model and implementation graph in tensorflow.

3. If you are interested in more magic about RNN, here is a MUST-READ blogpost: [The Unreasonable Effectiveness of Recurrent Neural Networks][f3] by Andrej Karpathy.

4. The vanilla version seq2seq+attention: [nicolas-ivanov/tf_seq2seq_chatbot][f4]. This will help you figure out the main flow of vanilla seq2seq model, and I build this repository based on this work.

[f1]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[f2]: https://github.com/sherjilozair/char-rnn-tensorflow
[f3]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[f4]: https://github.com/nicolas-ivanov/tf_seq2seq_chatbot


## TODOs
1. Currently I build beam-search out of graph, which means --- it's very slow. There are discussions about build it in-graph [here][g1] and [there][g2]. But unfortunately if you want add something more than beam-search, you need much more than just beam search to be in-graph.

[g1]: https://github.com/tensorflow/tensorflow/issues/654#issuecomment-196168030
[g2]: https://github.com/tensorflow/tensorflow/pull/3756

# A more detailed explaination about "the tensorflow chatbot"

Here I'll try to explain some algorithm and implementation details about [this work][a1] in layman's terms.

[a1]: https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm
 

## Sequence to sequence model

### What is a language model?

Let's say a language model is ...   
a) Trained by a lot of corpus.  
b) It could predict the **probability of next word** given foregoing words.  
=> It's just conditional probability, **P(next_word | foregoing_words)**  
c) Since we could predict next word:   
=> then predict even next, according to words just been generated  
=> continuously, we could produce sentences, even paragraph.

We could easily achieve this by simple [LSTM model][b1].


### The seq2seq model architecture

Again we quote this seq2seq architecture from [Google's blogpost]
[![seq2seq][b2]][b3]

It's composed of two language model: encoder and decoder. Both of them could be LSTM model we just mentioned.

The encoder part accept input tokens and transform the whole input sentence into an embedding **"thought vector"**, which express the meaning of input sentence in our language model domain. 

Then the decoder is just a language model, like we just said, a language model could generate new sentence according to foregoing corpus. Here we use this **"thought vector"** as kick-off and receive the corresponding mapping, and decode it into the response.


### Reversed encoder input and Attention mechanism

Now you might wonder:  
a) Considering this architecture, wil the "thought vector" be dominated by later stages of encoder?  
b) Is that enough to represent the meaning of whole input sentence into just a vector?  


For (a) actually, one of the implement detail we didn't mention before: the input sentence will be reversed before input to the encoder. Thus we shorten the distance between head of input sentence and head of response sentence. Empirically, it achieves better results. (This trick is not shown in the architecture figure above, for easy to understanding)

For (b), another methods to disclose more information to decoder is the [attention mechanism][b4]. The idea is simple: allowing each stage in decoder to peep any encoder stages, if they found useful in training phase. So decoder could understand the input sentence more and automagically peep suitable positions while generating response.



[b1]: http://colah.github.io/posts/2015-08-Understanding-LSTMs
[b2]: http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s640/2TFstaticgraphic_alt-01.png
[b3]: http://googleresearch.blogspot.ru/2015/11/computer-respond-to-this-email.html
[b4]: http://arxiv.org/abs/1412.7449



## Techniques about language model

### Dictionary space compressing and projection

A naive implementation of language model is: suppose we are training english language model, which a dictionary size of 80,000 is roughly enough. As we one-hot coding each word in our dictionary, our LSTM cell should have 80,000 outputs and we will do the softmax to choose for words with best probability...

... even if you have lots of computing resource, you don't need to waste like that. Especially if you are dealing with some other languages with more words like Chinese, which 200,000 words is barely enough.

Practically, we could reduce this 80,000 one-hot coding dictionary into embedding spaces, we could use like 64, 128 or 256 dimention to embed our 80,000 words dictionary, and train our model with only by this lower dimention. Then finally when we are generating the response, we project the embedding back into one-hot coding space for dictionary lookup.


### Beam search

The original implementation of tensorflow decode response sentence greedily. Empirically this trapped result in local optimum, and result in dump response which do have maximum probability in first couple of words. 

So we do the beam search, keep best N candidates and move-forward, thus we could avoid local optimum and find more longer, interesting responses more closer to global optimum result.

In [this paper][b4], Google Brain team found that beam search didn't benefit a lot in machine translation, I guess that's why they didn't implement beam search. But in my experience, chatbot do benefit a lot from beam search.
