# Sequence-to-Sequence Models

Recurrent neural networks can learn to model language, as already discussed
in the [RNN Tutorial](../../tutorials/recurrent/index.md)
러커런트 뉴로 네트웍은  [RNN 자습서](../../tutorials/recurrent/index.md)에서 이미 이야기 했던것 처럼 언어를 모델화하는 것을 학습할 수 있다.


(if you did not read it, please go through it before proceeding with this one).
This raises an interesting question: could we condition the generated words on
some input and generate a meaningful response? For example, could we train
a neural network to translate from English to French? It turns out that
the answer is *yes*.

(그것을 읽지 않았다면, 이 자습서를 읽기전이 그것을 먼저 읽기를 권한다). 흥미로운 질문이 있다: 몇몇의 입력으로 생성된 단어들을 조건화 할수 있고  어떤 의미있는 응답을 생성할수 있을까? 예를 들어, 영어를 프랑스어로 번역하기 위한 뉴럴 네트웍을 학습할수 있을까? 답은 *예*가 된다.

This tutorial will show you how to build and train such a system end-to-end.
We are assuming you have already installed via the pip package, have cloned the
tensorflow git repository, and are in the root of the git tree.

이 자습서는 그런 앤드-투-앤드 (end-to-end)시스템을 만들고 학습하는 방법 보여줄 것이다.  pip를 통해 tensorflow를 이미 설치했고 tensorflow git 저장소를 클론(clone)했고, git 트리루트에 있다고 가정한다.

You can then start by running the translate program:

그리고 나서, 번역 프로그램을 실행함으로 시작할수 있다:

```
cd tensorflow/models/rnn/translate
python translate.py --data_dir [your_data_directory]
```

It will download English-to-French translation data from the
[WMT'15 Website](http://www.statmt.org/wmt15/translation-task.html)
prepare it for training and train. It takes about 20GB of disk space,
and a while to download and prepare (see [later](#lets-run-it) for details),
so you can start and leave it running while reading this tutorial.

그 프로그램은 [WMT'15 웹사이트](http://www.statmt.org/wmt15/translation-task.html)에서 영어-프랑스어 번역 데이타를 다운받을 것이고 학습를 위해 그것을 준비하고 학습할 것이다. 그 데이타는 20기가 공간을 차지할 것이고, 다운로드하고 준비하기 위해서 좀 시간이 걸린다. 그래서 이 자습서를 읽는 동안 프로그램을 구동시키고 실행해 두자.

This tutorial references the following files from `models/rnn`.

이 자습서는 `models/rnn`에서 아래 파일을 참고한다.

File | What's in it?
--- | ---
`seq2seq.py` | Library for building sequence-to-sequence models.
`translate/seq2seq_model.py` | Neural translation sequence-to-sequence model.
`translate/data_utils.py` | Helper functions for preparing translation data.
`translate/translate.py` | Binary that trains and runs the translation model.


## Sequence-to-Sequence Basics
## 기본적인 시퀸스-투-시퀸스(Sequence-to-Sequence)

A basic sequence-to-sequence model, as introduced in
[Cho et al., 2014](http://arxiv.org/abs/1406.1078)
([pdf](http://arxiv.org/pdf/1406.1078.pdf)), consists of two recurrent neural
networks (RNNs): an *encoder* that processes the input and a *decoder* that
generates the output. This basic architecture is depicted below.

[Cho et al., 2014](http://arxiv.org/abs/1406.1078)
([pdf](http://arxiv.org/pdf/1406.1078.pdf))에서 소개된 기본적인 시퀸스투시퀸스 모델은 두개의 레커런트 뉴럴 네트웍(RNN)으로 구성된다: 입력을 처리하는 *인코더*와 결과를 생성하는 *디코더*. 이 기본 구조는 아래와 같이 묘사된다.

<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/basic_seq2seq.png" />
</div>

Each box in the picture above represents a cell of the RNN, most commonly
a GRU cell or an LSTM cell (see the [RNN Tutorial](../../tutorials/recurrent/index.md)
for an explanation of those). Encoder and decoder can share weights or,
as is more common, use a different set of parameters. Multi-layer cells
have been successfully used in sequence-to-sequence models too, e.g. for
translation [Sutskever et al., 2014](http://arxiv.org/abs/1409.3215)
([pdf](http://arxiv.org/pdf/1409.3215.pdf)).

위 그림의 각 박스는 가장 일반적으로 GRU 쎌이거나 LSTM 쎌인 RNN의 쎌을 나타낸다([RNN Tutorial](../../tutorials/recurrent/index.md를 참조하길 바란다). 인코더와 디코더는 가중치를 공유 할수 있거나, ??? 다른 파라미터 집합을 사용한다. 다중층 쎌들은 역시 시퀸스투시퀸스에서 성공적으로 사용되어져 왔다. 예로  [Sutskever et al., 2014](http://arxiv.org/abs/1409.3215)
([pdf](http://arxiv.org/pdf/1409.3215.pdf))


In the basic model depicted above, every input has to be encoded into
a fixed-size state vector, as that is the only thing passed to the decoder.
To allow the decoder more direct access to the input, an *attention* mechanism
was introduced in [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473)
([pdf](http://arxiv.org/pdf/1409.0473.pdf)).
We will not go into the details of the attention mechanism (see the paper),
suffice it to say that it allows the decoder to peek into the input at every
decoding step. A multi-layer sequence-to-sequence network with LSTM cells and
attention mechanism in the decoder looks like this.

위에 묘사된 기본 모델에서, 디코더에 전달되는 유일한 것이기 때문에 모든 입력은 고정된 크기를 가진 상태 벡터로 인코딩 되어져야 한다. 디코더가 입력에 더 직접적인 접근을 가능케 하기 위해, *어텐션(attention) 메카니즘이 [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473)([pdf](http://arxiv.org/pdf/1409.0473.pdf))에서 소개된다. 어탠션 메카나즘에 대해서 상세히 보지 않을 것이다(논문을 참고), 그것은 디코더가 모든 디코딩 단계에서 입력을 엿보게 해주는 것이라고 언급하는 것만으로도 충분하다. LSTM쎌을 가진 여러층의 시퀸스-투-시퀸스 네트웍과 디코더안에 어탠션 메카니즘은 이처럼 보인다.


<div style="width:80%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/attention_seq2seq.png" />
</div>


## TensorFlow seq2seq Library
## TensorFlow seq2seq 라이브러리

As you can see above, there are many different sequence-to-sequence
models. Each of these models can use different RNN cells, but all
of them accept encoder inputs and decoder inputs. This motivates
the interfaces in the TensorFlow seq2seq library (`models/rnn/seq2seq.py`).
The basic RNN encoder-decoder sequence-to-sequence model works as follows.

위에서 볼수 있듯이, 다른 많은 시퀸스-투-시퀸스 모델들이 있다. 각각 이러한 모델은 다른 RNN 쎌들을 사용할 수 있다, 그러나 모두 인코더 입력과 디코디 입력을 받아야 한다. 이것은 텐스플로우 seq2seq 라이브러리(`models/rnn/seq2seq.py`) 인터페이스의 동기가 된다.

```python
outputs, states = basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
```

In the above call, `encoder_inputs` are a list of tensors representing inputs
to the encoder, i.e., corresponding to the letters *A, B, C* in the first
picture above. Similarly, `decoder_inputs` are tensors representing inputs
to the decoder, *GO, W, X, Y, Z* on the first picture.

위에 호출에서, `encoder_inputs`는 인코더에 입력을 나타내는 텐서(tensor) 리스트이다. 예로, 위에 첫번째 그림에서 문자 *A, B, C*에 해당된다. 유사하게, `decoder_inputs`들은 디코더에 입력을 나타내는 텐서들이다. 첫번째 그림에 *GO, W, X, Y, Z* 이다. 


The `cell` argument is an instance of the `models.rnn.rnn_cell.RNNCell` class
that determines which cell will be used inside the model. You can use
an existing cell, such as `GRUCell` or `LSTMCell`, or you can write your own.
Moreover, `rnn_cell` provides wrappers to construct multi-layer cells,
add dropout to cell inputs or outputs, or to do other transformations,
see the [RNN Tutorial](../../tutorials/recurrent/index.md) for examples.

`cell` 인수는 `models.rnn.rnn_cell.RNNCell`클래스의 인스탄스이다. 게다가, `rnn_cell`은 여러층 쎌을 만들고, 쎌 입력과 결과에 드랍아웃(dropout)를 추가하거나 다른 변환을 하기 위한 랩퍼(wrapper)들을 제공한다. 예제들을 보기위해서 [RNN Tutorial](../../tutorials/recurrent/index.md)를 참고 하기 바란다.

The call to `basic_rnn_seq2seq` returns two arguments: `outputs` and `states`.
Both of them are lists of tensors of the same length as `decoder_inputs`.
Naturally, `outputs` correspond to the outputs of the decoder in each time-step,
in the first picture above that would be *W, X, Y, Z, EOS*. The returned
`states` represent the internal state of the decoder at every time-step.

`basic_rnn_seq2seq` 호출은 두개의 인수를 리턴한다: `outputs` 와 `states`. 그것 둘다 `decoder_inputs`와 동일한 크기의 탠서 리스트이다. 자연스럽게, `outputs`는 각 시간 단계(time-step)에서 디코더 결과에 해당된다, 위 첫번째 그림에서 그것은 *W, X, Y, Z, EOS*가 된다. 리턴된 `states`는 모든 시간단계에서 디코더의 내부 상태를 나타낸다.

In many applications of sequence-to-sequence models, the output of the decoder
at time t is fed back and becomes the input of the decoder at time t+1. At test
time, when decoding a sequence, this is how the sequence is constructed.
During training, on the other hand, it is common to provide the correct input
to the decoder at every time-step, even if the decoder made a mistake before.
Functions in `seq2seq.py` support both modes using the `feed_previous` argument.
For example, let's analyze the following use of an embedding RNN model.

시퀸스-투-시퀸스 모델의 많은 어플리케이션에서, 시간 t에 디코더 결과는 다시 시간 t+1에 디코더의 입력으로 전달된다. 테스트 시간에, 하나의 시퀸스를 디코딩할때, 이것은 그 시퀸스가 만들어지는 방법이다. 다른면으로, 학습동안, 디코더가 전에 실수를 있었다고 할찌라도, 모든 시간 단계에 디코더에 올바른 입력을 제공하는 것이 일반적인이다. `seq2seq.py`에 함수들은 `feed_previous` 인수를 사용해서 두가지 모드를 지원한다. 예를 들어, 아래에 임베드 RNN모델 사용을 분석해 보자.

```python
outputs, states = embedding_rnn_seq2seq(
    encoder_inputs, decoder_inputs, cell,
    num_encoder_symbols, num_decoder_symbols,
    output_projection=None, feed_previous=False)
```

In the `embedding_rnn_seq2seq` model, all inputs (both `encoder_inputs` and
`decoder_inputs`) are integer-tensors that represent discrete values.
They will be embedded into a dense representation (see the
[Vectors Representations Tutorial](../../tutorials/word2vec/index.md) for more details
on embeddings), but to construct these embeddings we need to specify
the maximum number of discrete symbols that will appear: `num_encoder_symbols`
on the encoder side, and `num_decoder_symbols` on the decoder side.

` embedding_rnn_seq2seq`모드에서, 모든 입력들은(`encoder_inputs` 과 `decoder_inputs`) 이산 값(discrete value)을 나타내는 정수-텐서(integer-tensor)들이다. 그것들은 덴스 표현으로 임베드되어 질 것이다(임베딩에 대해 더 자세한 설명을 위해 [Vectors Representations Tutorial](../../tutorials/word2vec/index.md 를 보길 권한다), 그러나 이러한 임베딩을 만들기 위해서 나타나는 분산 심볼의 최대 수를 지정할 필요가 있다: 인코더 쪽에 `num_encoder_symbols`, 그리고 디코더 쪽에 `num_decoder_symbols`. 

In the above invocation, we set `feed_previous` to False. This means that the
decoder will use `decoder_inputs` tensors as provided. If we set `feed_previous`
to True, the decoder would only use the first element of `decoder_inputs`.
All other tensors from this list would be ignored, and instead the previous
output of the encoder would be used. This is used for decoding translations
in our translation model, but it can also be used during training, to make
the model more robust to its own mistakes, similar
to [Bengio et al., 2015](http://arxiv.org/abs/1506.03099)
([pdf](http://arxiv.org/pdf/1506.03099.pdf)).

위 호출에서, `feed_previous`를 False로 설정했다. 이것은 디코더가 제공되어진 `decoder_inputs`를 사용할 것이다. `feed_previous`가 True로 설정되면, 디코더는 `decoder_inputs`의 단지 첫번째 원소만 사용할 것이다. 이 리스트에 모든 다른 텐서들은 무시 되어지고, 그대신 디코더의 이전 결과가 사용되어 질 것이다. 이것은 우리의 번역 모델에서 번역을 디코딩 하기 위해서 사용되어진다. [Bengio et al., 2015](http://arxiv.org/abs/1506.03099)
([pdf](http://arxiv.org/pdf/1506.03099.pdf)와 유사하게 그 모델이 그 자신의 실수에 더 견고하기 하기 위해 또한 학습동안에도 사용되어 질수 있다. 

One more important argument used above is `output_projection`. If not specified,
the outputs of the embedding model will be tensors of shape batch-size by
`num_decoder_symbols` as they represent the logits for each generated symbol.
When training models with large output vocabularies, i.e., when
`num_decoder_symbols` is large, it is not practical to store these large
tensors. Instead, it is better to return smaller output tensors, which will
later be projected onto a large output tensor using `output_projection`.
This allows to use our seq2seq models with a sampled softmax loss, as described
in [Jean et. al., 2014](http://arxiv.org/abs/1412.2007)
([pdf](http://arxiv.org/pdf/1412.2007.pdf)).

위에서 사용되어진 더 중요한 한 인수는 `output_projection`이다. 그것이 명시되지 않는다면, 임베딩 모델의 결과는 그것이 각 생성된 심볼에 대해 logits를 나타내는 것 처럼 batch-size x `num_decoder_symbols`의 형태를 가진 텐서가 될 것이다. 아주 큰 사전을 가지고 모델을 학습할때, 예를 들어, `num_decoder_symbols`이 아주 클때, 아주 큰 이러한 탠서들을 저장하는것은 비실용적이다. 그대신, 좀더 작은 결과 탠서를 리턴하는 것이 좋다. 그 텐서는 나중에 `output_projection`를 사용해서 큰 결과 텐서에 프로젝트되어진다. 이것은 [Jean et. al., 2014](http://arxiv.org/abs/1412.2007)
([pdf](http://arxiv.org/pdf/1412.2007.pdf))에서 설명하는 대로 샘플 소프트맥스 로스(sampled softmax loss)를 가진 seq2seq모델을 사용할수 있게 해준다. 

In addition to `basic_rnn_seq2seq` and `embedding_rnn_seq2seq` there are a few
more sequence-to-sequence models in `seq2seq.py`, take a look there. They all
have similar interfaces, so we will not describe them in detail. We will use
`embedding_attention_seq2seq` for our translation model below.

`basic_run_seq2seq`와 `embedding_rnn_seq2seq`에 더해서, `seq2seq.py`에 좀더 많은 시퀸스-투-시퀸스 모델들이 있고, 그것을 볼 것이다. 그것들은 유사한 인터페이스를 가진다, 그래서 상세히 설명하지는 않을 것이다. 아래에 우리의 번역 모델을 위해 `embedding_attention_seq2seq`를 사용할 것이다.

## Neural Translation Model

While the core of the sequence-to-sequence model is constructed by
the functions in `models/rnn/seq2seq.py`, there are still a few tricks
that are worth mentioning that are used in our translation model in
`models/rnn/translate/seq2seq_model.py`.

시퀸스-투-시퀸스(sequence-to-sequence) 의 핵심은 `models/rnn/seq2seq.py`에 있는 함수에 의해 구성되어지지만, `models/rnn/translate/seq2seq_model.py`에 있는 우리의 번역 모델에서 사용되어지는 것을 언급하는 것이 가치 있는 몇가지 트릭이 여전히 있다.??? 

### Sampled softmax and output projection
### 샘플 소프트맥스와 결과 프로젝션(Sampled softmax and output projection)

For one, as already mentioned above, we want to use sampled softmax to
handle large output vocabulary. To decode from it, we need to keep track
of the output projection. Both the sampled softmax loss and the output
projections are constructed by the following code in `seq2seq_model.py`.

하나는, 위에서 이미 언급한 대로, 우리는 아주 큰 사전을 다루기 위해서 샘플 소프트맥스를 사용하고 싶다. 그것으로 부터 디코딩하기 위해 결과 프로젝션를 기억할 필요가 있다. 샘플 소프트맥스 손실(loss)과 결과 프로젝트 둘다는 `seq2seq_model.py`에 있는 다음 코드에 의해 구성되어진다. 

```python
  if num_samples > 0 and num_samples < self.target_vocab_size:
    w = tf.get_variable("proj_w", [size, self.target_vocab_size])
    w_t = tf.transpose(w)
    b = tf.get_variable("proj_b", [self.target_vocab_size])
    output_projection = (w, b)

    def sampled_loss(inputs, labels):
      labels = tf.reshape(labels, [-1, 1])
      return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                        self.target_vocab_size)
```

First, note that we only construct a sampled softmax if the number of samples
(512 by default) is smaller than the target vocabulary size. For vocabularies
smaller than 512, it might be a better idea to just use a standard softmax loss.

첫번째, 샘플 수(디폴트가 512)가 목적(target) 사전 크기보다 작다면, 단지 샘플 소프트맥스를 구성한다는 것을 기억하자. 512보다 작은 사전에 대해, 단지 표준 소프트맥스 손실(lost)를 사용하는 것이 더 나은 생각일지도 모른다.

Then, as you can see, we construct an output projection. It is a pair,
consisting of a weight matrix and a bias vector. If used, the rnn cell
will return vectors of shape batch-size by `size`, rather than batch-size
by `target_vocab_size`. To recover logits, we need to multiply by the weight
matrix and add the biases, as is done in lines 124-126 in `seq2seq_model.py`.

```python
if output_projection is not None:
  self.outputs[b] = [tf.matmul(output, output_projection[0]) +
                     output_projection[1] for ...]
```

그리고 나서, 알다시피, 결과 프로젝트를 구성한다. 그것은 가중치 행렬과 바이어스 벡터로 구성된 순서쌍이 된다. 사용되어지면, rnn쎌은 `target_vocab_size`에 의한 배치 크기가 아니라 `size`에 의한 배치 크기의 벡터들이 리턴될 것이다. logit를 복구하기 위해, `seq2seq_model.py`에 124-126라인에서 했던 것처럼 가중치 행렬에 의해서 곱해지고, 바이어스가 더해질 필요가 있다. 

### Bucketing and padding

In addition to sampled softmax, our translation model also makes use
of *bucketing*, which is a method to efficiently handle sentences of
different lengths. Let us first clarify the problem. When translating
English to French, we will have English sentences of different lengths L1
on input, and French sentences of different lengths L2 on output. Since
the English sentence is passed as `encoder_inputs`, and the French sentence
comes as `decoder_inputs` (prefixed by a GO symbol), we should in principle
create a seq2seq model for every pair (L1, L2+1) of lengths of an English
and French sentence. This would result in an enormous graph consisting of
many very similar subgraphs. On the other hand, we could just pad every
sentence with a special PAD symbol. Then we'd need only one seq2seq model,
for the padded lengths. But on shorter sentence our model would be inefficient,
encoding and decoding many PAD symbols that are useless.

샘플 소프트맥스에 더해서, 우리의 번역 모델은 또한 *버컷링(bucketing)*를 사용한다, 그것은 다른 크기의 문장을 효과적으로 다루기 위한 방법이다. 우선 그 문제를 명확히 해 보자. 영어에서 프랑스어로 번역할때, 입력으로 L1 크기의 영어문장과 결과로 L2 크기의 프랑스 문장이 있을 것이다. 영어 문장이 `encoder_inputs`으로 전달되고 프랑스어 문장은 `decoder_inputs`(GO 심볼이 앞에 붙혀진다) 로 나오기 때문에, 기본적으로 (L1, L2+1)의 모든 순서쌍에 대해 seq2seq 모델을 만들어야 한다. 이것은 아주 거대한 그래프를 만들게 될 것이다. 다른한편으로 특별한 PAD 심볼을 가지고 모든 문장에 단지 삽입할수 있다. 그러면, 우리는 그 패딩된 크기에 대한 단지 하나의 seq2seq 모델만 만들 필요있게 된다.  

As a compromise between constructing a graph for every pair of lengths and
padding to a single length, we use a number of *buckets* and pad each sentence
to the length of the bucket above it. In `translate.py` we use the following
default buckets.
모든 길이의 순서쌍에 대해 그래프를 생성하는 것과 하나의 길이로 패딩하는 것 사이에 절충으로써 몇개의 *버켓(bucket)를 사용하고 각 문장을 그 위에서 있는 버켓의 길이로 메워 넣는다.`translate.py`에서 다음과 같은 기본적인 버켓들을 사용한다.

```python
buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
```

This means that if the input is an English sentence with 3 tokens,
and the corresponding output is a French sentence with 6 tokens,
then they will be put in the first bucket and padded to length 5 for
encoder inputs, and length 10 for decoder inputs. If we have an English
sentence with 8 tokens and the corresponding French sentence has 18 tokens,
then they will not fit into the (10, 15) bucket, and so the (20, 25) bucket
will be used, i.e. the English sentence will be padded to 20, and the French
one to 25.

이것은 3개의 토근을 가진 영어 문장이 입력이고 6개의 토근을 가진 프랑스어 문장이 그 입력에 대한 결과라고 한다면 그것들은 첫번째 버켓에 들어가게 되고 인코더 입력에 길이 5까지 메워넣고, 디코더 입력에 대해서는 길이 10까지 메워넣게 된다. 8개의 토근을 가진 영어문장이고 그에 상응하는 프랑스어 문장이 18개의 토근을 가진다면, 버켓 (10, 15)에 맞지 않고, 버켓 (20, 25)이 사용될 것이다. 예를 들어, 그 영어 문장은 길이 20까지 메워 질것이고, 프랑스어 문장은 길이 25까지 메워진다.

Remember that when constructing decoder inputs we prepend the special `GO`
symbol to the input data. This is done in the `get_batch()` function in
`seq2seq_model.py`, which also reverses the input English sentence.
Reversing the inputs was shown to improve results for the neural translation
model in [Sutskever et al., 2014](http://arxiv.org/abs/1409.3215)
([pdf](http://arxiv.org/pdf/1409.3215.pdf)).
To put it all together, imagine we have the sentence "I go.", tokenized
as `["I", "go", "."]` as input and the sentence "Je vais." as output,
tokenized `["Je", "vais", "."]`. It will be put in the (5, 10) bucket,
with encoder inputs representing `[PAD PAD "." "go" "I"]` and decoder
inputs `[GO "Je" "vais" "." EOS PAD PAD PAD PAD PAD]`.

디코더 입력을 만들때, 특별한 `GO`심볼을 입력 데이타 앞에 붙히는 것을 기억하자. 이것은  `seq2seq_model.py`에 `get_batch()`함수 안에서 이루어진다. 그것은 또한 영어 문장 입력을 반대로 변환한다. 입력을 반대로 변환하는 것은 [Sutskever et al., 2014](http://arxiv.org/abs/1409.3215)
([pdf](http://arxiv.org/pdf/1409.3215.pdf))에서 뉴럴 번역 모델에 대한 결과를 향상시키는 것을 보여주었다. 이것 모든 것을 다 넣기 위해, "I go." 라는 문장이 있다고 가정하고, 입력으로 `["I", "go", "."]`로 토근화 되고 결과로  "Je vais"가 되고 `["Je", "vais", "."]`로 토근화 된다. 그것은 `[PAD PAD "." "go" "I"]`로 인코더 입력이고, `[GO "Je" "vais" "." EOS PAD PAD PAD PAD PAD]`는 디코더 입력으로  버켓 (5,10)에 들어가게 될것이다.


## Let's Run It
## 그것을 실행해보자

To train the model described above, we need to a large English-French corpus.
We will use the *10^9-French-English corpus* from the
[WMT'15 Website](http://www.statmt.org/wmt15/translation-task.html)
for training, and the 2013 news test from the same site as development set.
Both data-sets will be downloaded to `data_dir` and training will start,
saving checkpoints in `train_dir`, when this command is run.

위에 설명되어진 대로 모델을 훈련하기 위해, 우리는 아주 큰 영어-프랑스어 코퍼서가 필요하다. [WMT'15 Website](http://www.statmt.org/wmt15/translation-task.html)에서 *10^9-French-English corpus*를 훈련을 위해 사용할 것이고, 개발 집합로 그 동일한 사이트에서 2013 news test를 사용할 것이다.

```
python translate.py
  --data_dir [your_data_directory] --train_dir [checkpoints_directory]
  --en_vocab_size=40000 --fr_vocab_size=40000
```

It takes  about 18GB of disk space and several hours to prepare the training
corpus. It is unpacked, vocabulary files are created in `data_dir`, and then
the corpus is tokenized and converted to integer ids. Note the parameters
that determine vocabulary sizes. In the example above, all words outside
the 40K most common ones will be converted to an `UNK` token representing
unknown words. So if you change vocabulary size, the binary will re-map
the corpus to token-ids again.

약 18기가 디스크 공간을 차지 하고 훈련 코퍼스를 준비하기 위해 몇 시간이 걸린다. 그것이 풀려지면, 사전 파일들이 `data_dir`에 만들어 진다, 그리고 나서 그 코퍼스는 토근화 되어지고 정수 아이디로 변환될것이다. 사전 크기를 결정하는 매개변수들을 주시하자. 위 예에서, 가장 일반적인 단어 4만개에 있는 않는 모든 단어들은 모르는 단어를 나타내는 `UNK`토근으로 변환 될것이다. 사전 크기를 변경한다면 그 바이너리는 토근 아이를 그 코퍼스에 다시 연결 할것이다.

After the data is prepared, training starts. Default parameters in `translate`
are set to quite large values. Large models trained over a long time give good
results, but it might take too long or use too much memory for your GPU.
You can request to train a smaller model as in the following example.

데이타가 준비된 뒤에, 훈련이 시작된다. `translate`에 있는 디폴트 매개변수는 꽤 큰 값으로 설정된다. 아주 긴 시간 동안 학습된 큰 모델은 좋은 결과를 준다, 그러나 너무 긴 시간과 GPU에 너무 많은 메모리를 사용할찌도 모른다. 다음 예처럼 더 작은 모델을 학습하는 것을 요청할 수 있다. 

```
python translate.py
  --data_dir [your_data_directory] --train_dir [checkpoints_directory]
  --size=256 --num_layers=2 --steps_per_checkpoint=50
```

The above command will train a model with 2 layers (the default is 3),
each layer with 256 units (default is 1024), and will save a checkpoint
every 50 steps (the default is 200). You can play with these parameters
to find out how large a model can be to fit into the memory of your GPU.

위 명령은 256 유닛(디폴트는 1024)을 가진 각 2개층(디폴트는 3)을 가진 모델을 학습할 것이고, 50 단계(디폴트는 200) 마다 체크포인트를 저장할 것이다. 여러분의 GPU의 메모리에 얼마나 큰 모델이 맞게 할 수 있는지 알기 위해 이러한 매개변수를 통해 할수 있다.

During training, every `steps_per_checkpoint` steps the binary will print
out statistics from recent steps. With the default parameters (3 layers
of size 1024), first messages look like this.

학습 동안, 모든 `steps_per_checkpoint` 단계에서 바이너리는 최근 단계로 부터 통계를 프린터 할것이다. 디폴트 매개변수(크기 1024의 3개층)와 함께 첫번째 메시지는 이처럼 보인다.

```
global step 200 learning rate 0.5000 step-time 1.39 perplexity 1720.62
  eval: bucket 0 perplexity 184.97
  eval: bucket 1 perplexity 248.81
  eval: bucket 2 perplexity 341.64
  eval: bucket 3 perplexity 469.04
global step 400 learning rate 0.5000 step-time 1.38 perplexity 379.89
  eval: bucket 0 perplexity 151.32
  eval: bucket 1 perplexity 190.36
  eval: bucket 2 perplexity 227.46
  eval: bucket 3 perplexity 238.66
```

You can see that each step takes just under 1.4 seconds, the perplexity
on the training set and the perplexities on the development set
for each bucket. After about 30K steps, we see perplexities on short
sentences (bucket 0 and 1) going into single digits.
Since the training corpus contains ~22M sentences, one epoch (going through
the training data once) takes about 340K steps with batch-size of 64. At this
point the model can be used for translating English sentences to French
using the `--decode` option.

각 단계는 단지 1.4초보다 빠르게 걸린다는 것과 훈련 셋에 퍼플렉시티(perplexity)와 개발 셋에 퍼플렉시티들을 볼수 있다. 대략 3만 단계 뒤에, 짧은 문장(버켓이 0과 1)에 대한 퍼플렉시티가 한 자릿수로 이동한다. 훈련 코퍼스는 22M 문장을 포함하기 땜누에, 각 에포(훈련 데이타 전체 한번 통과)은 64의 배치크기로 대략 34만 단계가 걸린다. 이 시점에, 그 모델은 `--decode` 옵션을 사용해서 영어 문장에서 프랑스 문장으로 변역하기 위해 사용될어 질 수 있다.

```
python translate.py --decode
  --data_dir [your_data_directory] --train_dir [checkpoints_directory]

Reading model parameters from /tmp/translate.ckpt-340000
>  Who is the president of the United States?
 Qui est le président des États-Unis ?
```

## What Next?

The example above shows how you can build your own English-to-French
translator, end-to-end. Run it and see how the model performs for yourself.
While it has reasonable quality, the default parameters will not give you
the best translation model. Here are a few things you can improve.

위 예는 앤드-투-앤드(end-to-end)로 영어-에서-프랑스어(English-to-French) 번역기를 만드는 방법을 보여준다. 스스로 그것을 실행하고 그 모델이 어떻게 수행하는지 봐라. 그것이 꽤 괜찮은 성능을 보이지만, 디플트 매개변수로 최고의 번역 모델이 주어지지 않는다. 여기 여러분이 그것을 향상 시킬수 있는 몇가지 방법이 있다.

First of all, we use a very primitive tokenizer, the `basic_tokenizer` function
in `data_utils`. A better tokenizer can be found on the
[WMT'15 Website](http://www.statmt.org/wmt15/translation-task.html).
Using that tokenizer, and a larger vocabulary, should improve your translations.

무엇보다, `data_utils`안에 `basic_tokenizer`인 아주 기본적인 토큰생성자를 사용한다. 더 좋은 토큰 생성자는 [WMT'15 Website](http://www.statmt.org/wmt15/translation-task.html)에서 찾을수 있다. 저 토근 생성자와 더 큰 사전을 사용하는 것은 번역을 향샹 시킬 것이다.

Also, the default parameters of the translation model are not tuned.
You can try changing the learning rate, decay, or initializing the weights
of your model in a different way. You can also change the default
`GradientDescentOptimizer` in `seq2seq_model.py` to a more advanced one, such
as `AdagradOptimizer`. Try these things and see how they improve your results!

또한, 그 번역 모델의 디폴트 매개변수는 최적화 되지 않았다. 학습 비율이나 디케이를 변경하거나, 다른 방법으로 여러분의 모델의 가중치들을 초기하는 것을 시도 할수 있다. 또한 `seq2seq_model.py`에 있는 디폴트인 `GradientDescentOptimizer`를 더 혁신적인 `AdagradOptimizer`로 변경할 수 있다. 이러한 것들을 시도하고 결과가 어떻게 향샹 되는지 보자.

Finally, the model presented above can be used for any sequence-to-sequence
task, not only for translation. Even if you want to transform a sequence to
a tree, for example to generate a parsing tree, the same model as above can
give state-of-the-art results, as demonstrated in
[Vinyals & Kaiser et al., 2014](http://arxiv.org/abs/1412.7449)
([pdf](http://arxiv.org/pdf/1412.7449.pdf)).
So you can not only build your own translator, you can also build a parser,
a chat-bot, or any program that comes to your mind. Experiment!

마지막으로, 위에서 나타난 모델은 번역뿐만 아니라 어떤 시퀸스-투-시퀸스 작업에 대해서도 사용되어 질수 있다. 하나의 시퀸스을 트리로 변환하기 원한다고 할찌라도, 예를 들어 파싱 트리를 생성하기, 위처럼 동일한 모델은[Vinyals & Kaiser et al., 2014](http://arxiv.org/abs/1412.7449)
([pdf](http://arxiv.org/pdf/1412.7449.pdf))에서 보여진대로 최고의 결과를 준다. 그래서 여러분 자신의 번역기를 만들수 있을 뿐만 아니라 또한 파서, 챗봇, 또는 당신 머리에 떠오르는 어떤 프로그램도 만들수 있다. 실험해보라!

