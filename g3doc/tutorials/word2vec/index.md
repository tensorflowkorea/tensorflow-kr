# Vector Representations of Words
# 단어들의 벡터 표현
 
In this tutorial we look at the word2vec model by
[Mikolov et al.](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
This model is used for learning vector representations of words, called "word
embeddings".
이 튜토리얼에서, [Mikolov et al.](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 의 word2vec 모델을 살펴본다. 이 모델은 "word embeddings" 라 불리는 단어들의 벡터 표현의 학습에 사용된다.

## Highlights
## Highlights

This tutorial is meant to highlight the interesting, substantive parts of
building a word2vec model in TensorFlow.
이 튜토리얼은 TensorFlow 에서 word2vec 모델을 형성하는 흥미있고 실질적인 부분들을 강조한다.

* We start by giving the motivation for why we would want to
represent words as vectors.
* We look at the intuition behind the model and how it is trained
(with a splash of math for good measure).
* We also show a simple implementation of the model in TensorFlow.
* Finally, we look at ways to make the naive version scale better.
* 왜 단어들을 벡터들로 표현해야 하는지에 대한 주어진 동기에서 시작한다.
* 모델 뒤의 직관과 어떻게 학습되었는지(정확한 측정을 위한 수학을 사용)를 알아본다.
* 마지막으로, 초기 버전 수준을 더 잘 만들수 있는 방법을 알아본다.

We walk through the code later during the tutorial, but if you'd prefer to dive
straight in, feel free to look at the minimalistic implementation in
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
This basic example contains the code needed to download some data, train on it a
bit and visualize the result. Once you get comfortable with reading and running
the basic version, you can graduate to
[tensorflow/models/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py)
which is a more serious implementation that showcases some more advanced
TensorFlow principles about how to efficiently use threads to move data into a
text model, how to checkpoint during training, etc.
후에 튜토리얼에서 코드를 경험할 수 있게 보여줄 것이다. 하지만 좀 더 자세히 알고 싶다면, [tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) 의 최소화된 구현을 참고하자. 이 기본 예제는 데이타를 다운로드를 요청하고, 이것을 학습하고, 결과를 시각화하는 코드를 포함한다. 기본 버전을 읽고 실행하는데 익숙해지면, 쓰레드를 이용하여 어떻게 효율적으로 데이터를 텍스트 모델로 이동시키는지, 학습하는 동안 어떻게 체크하는지 등에 대한 더 심화된 TensorFlow 원리들을 보여주는 심화 구현된 [tensorflow/models/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py) 을 시작할 수 있다.

But first, let's look at why we would want to learn word embeddings in the first
place. Feel free to skip this section if you're an Embedding Pro and you'd just
like to get your hands dirty with the details.
하지만 우선, 초반 왜 우리가 word embeddings 에 대해 배워야 하는지 알아보자. Embedding 을 잘 알고 자세한 설명들이 혼란스럽다고 생각한다면, 이 부분을 넘어가도 된다.

## Motivation: Why Learn Word Embeddings?
## 동기 : 왜 Word Embeddings 를 배워야하지?

Image and audio processing systems work with rich, high-dimensional datasets
encoded as vectors of the individual raw pixel-intensities for image data, or
e.g. power spectral density coefficients for audio data. For tasks like object
or speech recognition we know that all the information required to successfully
perform the task is encoded in the data (because humans can perform these tasks
from the raw data).  However, natural language processing systems traditionally
treat words as discrete atomic symbols, and therefore 'cat' may be represented
as  `Id537` and 'dog' as `Id143`.  These encodings are arbitrary, and provide
no useful information to the system regarding the relationships that may exist
between the individual symbols. This means that the model can leverage
very little of what it has learned about 'cats' when it is processing data about
'dogs' (such that they are both animals, four-legged, pets, etc.). Representing
words as unique, discrete ids furthermore leads to data sparsity, and usually
means that we may need more data in order to successfully train statistical
models.  Using vector representations can overcome some of these obstacles.
이미지, 오디오 처리 시스템들은 이미지 데이터에 대한 각각 가공되지 않은 픽셀-강도값의 벡터들이나 오디오 데이터에 대한 파워스펙트럴밀도 로 저장된 대량의 고차원 dataset 들과 함께 한다. 객체나 연설 인식과 같은 문제(task)에서 문제(task)를 성공적으로 수행하기 위해 필요한 모든 정보는 데이터에 부호화 된다는 것을 알고있다.(사람은 이러한 문제를 가공되지 않은 데이터로부터 수행하기 때문이다.) 그러나, 자연어 처리 시스템들은 일반적으로 이산 원자 기호들(discrete atomic symbols)의 단어들로 다뤄진다. 따라서 'cat' 은 'Id537'로, 'dog' 은 'Id143'로 표현된다. 이러한 부호화 자료들은 임의적이며, 각각의 기호(symbol)들 간에 존재하는 관계에 관한 시스템과는 의미없는 정보로 제공된다. 이것은 'dogs' 라는 처리 데이터에 대해서 'cats' 을 배우는 것이 영향력이 매우 적음을 의미한다.(이들 모두 동물, 네 다리, 애완동물 등) 유일하고 이산의(discrete) id 들로 단어들을 표현하는 것은 데이터를 더욱 드문드문(sparsity) 하게 이끌고, 대체적으로 통계적 모델들을 성공적으로 학습하기 위해 더 많은 데이터가 필요함을 의미한다. 벡터 표현들을 사용하는 것은 다음과 같은 장애들을 해결할 수 있다.

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/audio-image-text.png" alt>
</div>

[Vector space models](https://en.wikipedia.org/wiki/Vector_space_model) (VSMs)
represent (embed) words in a continuous vector space where semantically
similar words are mapped to nearby points ('are embedded nearby each other').
VSMs have a long, rich history in NLP, but all methods depend in some way or
another on the
[Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis),
which states that words that appear in the same contexts share
semantic meaning. The different approaches that leverage this principle can be
divided into two categories: *count-based methods* (e.g.
[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)),
and *predictive methods* (e.g.
[neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models)).
[벡터공간 모델(Vector space models)](https://en.wikipedia.org/wiki/Vector_space_model) (VSMs) 은 의미상 유사한 단어들은 가까운 지점으로 매핑되어지는 연속된 벡터 공간의 단어들로 표현(내포)한다.(서로 가깝게 의미하는) VSMs 은 NLP 에서 길고, 깊은 역사를 가지고 있지만, 모든 방법들이 특정 방법이나 [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis) 의 방법에 따라 달라진다. 그리고 이 논문은 같은 맥락에서 나타나는 단어들은 시멘틱 의미를 공유한다고 명시한다. 이 원리에 영향을 주는 다른 접근법들은 두 종류로 나눠진다: *count-based methods*(e.g.[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)), 와 *predictive methods* (e.g.[neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models)).

This distinction is elaborated in much more detail by
[Baroni et al.](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf),
but in a nutshell: Count-based methods compute the statistics of
how often some word co-occurs with its neighbor words in a large text corpus,
and then map these count-statistics down to a small, dense vector for each word.
Predictive models directly try to predict a word from its neighbors in terms of
learned small, dense *embedding vectors* (considered parameters of the
model).
이 차이는 [Baroni et al.](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf) 에 의해 더 자세하게 설명되어 있다. 간략하게 말해: Count-based methods 는 대량의 텍스트 말뭉치에서 특정 단어가 그 주변 단어들과 함께 얼마나 나타나는지에 대한 통계를 계산한다. 그다음 이 count-statistics 를 각각의 단어에 대해 작고 dense 벡터로 상세히 묘사한다. 예측 모델(Predictive models)은 학습된 작고, dense *embedding vectors*(고려된 모델의 파라미터들) 에 대해서 단어를 그 주변 단어들로부터 직접적으로 예측하려 시도한다.

Word2vec is a particularly computationally-efficient predictive model for
learning word embeddings from raw text. It comes in two flavors, the Continuous
Bag-of-Words model (CBOW) and the Skip-Gram model (Chapter 3.1 and 3.2 in [Mikolov et al.](http://arxiv.org/pdf/1301.3781.pdf)). Algorithmically, these
models are similar, except that CBOW predicts target words (e.g. 'mat') from
source context words ('the cat sits on the'), while the skip-gram does the
inverse and predicts source context-words from the target words. This inversion
might seem like an arbitrary choice, but statistically it has the effect that
CBOW smoothes over a lot of the distributional information (by treating an
entire context as one observation). For the most part, this turns out to be a
useful thing for smaller datasets. However, skip-gram treats each context-target
pair as a new observation, and this tends to do better when we have larger
datasets. We will focus on the skip-gram model in the rest of this tutorial.
Word2vec 는 가공되지 않은 텍스트로부터 학습한 단어 embeddings 에 대해 특히 계산적으로 효율적인 예측 모델이다. 이것은 두 가지 형태로 나타난다, Continuous Bag-of-Word(CBOW) 모델과 Skip-Gram 모델([Mikolov et al.](http://arxiv.org/pdf/1301.3781.pdf) 의 Chapter 3.1과 3.2). 알고리즘적으로, 이들 모델들은 CBOW 는 원본 컨텍스트 단어들('the cat sits on the') 로부터 타켓 단어들(e.g. 'mat') 을 예측하는 반면 skip-gram 은 타겟 단어들로부터 원본 컨텍스트 단어들을 역으로 예측한다는 점을 제외하고 유사하다. 이 관계(도치)는 임의적인 선택이라고 볼 수 있지만, 통계적으로 CBOW 는 많은 수의 분포상 정보(전체 컨텍스트를 하나의 발견으로 처리함으로써)를 바로잡는 효과를 가진다. 대부분의 경우, 이것(CBOW)은 작은 datasets 일 수록 유용한 것으로 밝혀졌다. 그러나 skip-gram 은 각 컨텍스트-타겟 쌍을 새로운 발견으로 처리하고, 이것은 대량의 datasets 을 가질 때 더 잘 동작하는 경향이 있다. 이 튜토리얼의 나머지는 skip-gram 모델에 초점을 맞춰 설명할 것이다.

## Scaling up with Noise-Contrastive Training
## Noise-Contrastive 학습법을 이용한 규모확장

Neural probabilistic language models are traditionally trained using the
[maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood) (ML)
principle  to maximize the probability of the next word \\(w_t\\) (for "target")
given the previous words \\(h\\) (for "history") in terms of a
[*softmax* function](https://en.wikipedia.org/wiki/Softmax_function),
신경 확률 언어 모델들(Neural probabilistic language models) 은 일반적으로 [*softmax* function](https://en.wikipedia.org/wiki/Softmax_function) 에서 주어진 이전 단어들\\(h'\\)(for "history")에 대해 다음 단어(\\(w_t\\)(for "target") 의 확률을 최대화하는 [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood) (ML) 원리를 이용하여 학습되어 진다.

$$
\begin{align}
P(w_t | h) &= \text{softmax}(\text{score}(w_t, h)) \\
           &= \frac{\exp \{ \text{score}(w_t, h) \} }
             {\sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} }.
\end{align}
$$

where \\(\text{score}(w\_t, h)\\) computes the compatibility of word \\(w\_t\\)
with the context \\(h\\) (a dot product is commonly used). We train this model
by maximizing its [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function) 
on the training set, i.e. by maximizing
\\(\text{score}(w\_t, h)\\) 은 컨텍스트와 함께 하는 단어의 호환성을 계산한다.(내적, dot product, 을 일반적으로 사용) 우리는 학습하는 set 에서 이것의 [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function) 를 최대화 함으로써 모델을 학습한다. 즉 최대화 하도록.

$$
\begin{align}
 J_\text{ML} &= \log P(w_t | h) \\
  &= \text{score}(w_t, h) -
     \log \left( \sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} \right)
\end{align}
$$

This yields a properly normalized probabilistic model for language modeling.
However this is very expensive, because we need to compute and normalize each
probability using the score for all other \\(V\\) words \\(w'\\) in the current
context \\(h\\), *at every training step*.
이것은 언어 모델링에 대해서 적절하게 정규화된 확률 모델을 만들어 낸다. 그러나 이 방법은 매 학습 스텝에서(*at every training step*) 현재 컨텍스트의 다른 모든\\(V\\) words \\(w'\\) 에 대한 범위를 활용하여 각각의 확률을 계산하고 정규화하는 것이 필요한 이유로 그 비용이 매우 비싸다.

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/softmax-nplm.png" alt>
</div>

On the other hand, for feature learning in word2vec we do not need a full
probabilistic model. The CBOW and skip-gram models are instead trained using a
binary classification objective ([logistic regression](https://en.wikipedia.org/wiki/Logistic_regression))
to discriminate the real target words \\(w_t\\) from \\(k\\) imaginary (noise) words \\(\tilde w\\), in the same context. We illustrate this below for a CBOW model. For skip-gram the direction is simply inverted.
반면, word2vec 의 feature 학습에 대하여 모든 것에 대한 확률적 모델(full probabilistic model) 을 필요로 하지 않는다. 대신 CBOW 나skip-gram 모델은 같은 컨텍스트 내에서 실제 타겟 단어들 \\(w_t\\) 을 가상(노이즈) 단어들 \\(\tilde w\\) 로부터 구별해 내기 위한 이진 분류 목적(objective)([logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)) 을 이용하여 학습되어 진다. CBOW 모델에 대한 것은 아래에 도식화하였다. skip-gram 에 대해서 그 방향이 간략히 반대로 되어 있다.

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/nce-nplm.png" alt>
</div>

Mathematically, the objective (for each example) is to maximize
수학적으로, 목적함수(각 예제에 대해) 는 이를 최대화 한다.

$$J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) +
  k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right]$$

where \\(Q_\theta(D=1 | w, h)\\) is the binary logistic regression probability
under the model of seeing the word \\(w\\) in the context \\(h\\) in the dataset
\\(D\\), calculated in terms of the learned embedding vectors \\(\theta\\). In
practice we approximate the expectation by drawing \\(k\\) contrastive words
from the noise distribution (i.e. we compute a
[Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration)).
\\(Q_\theta(D=1 | w, h)\\)학습된 embedding vectors \\(\theta\\) 에 대해 계산된 dataset\\(D\\) 내부 컨텍스트\\(h\\) 에서 보이는 단어(\\(w\\) 의 모델에 대한 이진 로지스틱 회귀 확률이다. 실제로 노이즈 분포(즉, [Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration) 를 게산한다) 로부터 대조 단어들을 찾아냄으로써 기대값에 대해 가늠했다.

This objective is maximized when the model assigns high probabilities
to the real words, and low probabilities to noise words. Technically, this is
called
[Negative Sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf),
and there is good mathematical motivation for using this loss function:
The updates it proposes approximate the updates of the softmax function in the
limit. But computationally it is especially appealing because computing the
loss function now scales only with the number of *noise words* that we
select (\\(k\\)), and not *all words* in the vocabulary (\\(V\\)). This makes it
much faster to train. We will actually make use of the very similar
[noise-contrastive estimation (NCE)](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)
loss, for which TensorFlow has a handy helper function `tf.nn.nce_loss()`.
이 목적함수는 모델이 실제 단어들에 높은 확률을 배정하고 노이즈 단어들에 낮은 확률을 배정할 때 최대화된다. 기술적으로, [Negative Sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 이라 불리며, 이 손실(loss) 함수의 사용에 대해 수학적으로 유리한 동기가 존재한다: 제시되는 업데이트들은 제한된 범위에서 softmax 함수의 업데이트들을 근사값을 계산한다. 하지만 손실 함수의 계산을 우리가 선택한 *noise words*(\\(k\\)) 의 수와 선택하지 않은 어휘(\\(V\\)) 모든 단어(*all words*) 만으로 변경한다는 점 때문에 계산적으로 특히 매력적이다. 이것은 학습을 더욱 빠르게 만든다. 우리는 [noise-contrastive estimation (NCE)](http://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf) 손실(loss) 와 매우 유사한, TensorFlow 가 가지고 있는 유용한 헬퍼 함수 `tf.nn.nce_loss()` 를 활용할 것이다.

Let's get an intuitive feel for how this would work in practice!
이제 실제로 어떻게 동작했는지에 대해 직관적으로 이해해보자.

## The Skip-gram Model
## Skip-gram 모델

As an example, let's consider the dataset
예제로, dataset 을 생각해보자

`the quick brown fox jumped over the lazy dog`

We first form a dataset of words and the contexts in which they appear. We
could define 'context' in any way that makes sense, and in fact people have
looked at syntactic contexts (i.e. the syntactic dependents of the current
target word, see e.g.
[Levy et al.](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)),
words-to-the-left of the target, words-to-the-right of the target, etc. For now,
let's stick to the vanilla definition and define 'context' as the window
of words to the left and to the right of a target word. Using a window
size of 1, we then have the dataset
우선 단어들의 dataset 과 그들이 존재하는 컨텍스트를 생성한다. 우리는 타당한 여러 방법으로 'context' 를 정의할 수 있고, 사실 사람들은 통사적 문맥(즉, 현재 타겟 단어의 통사적 의존성, 참고 [Levy et al.](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)), 타겟의 좌측, 우측 단어들 등을 고려하게 된다. 우선, 평범한 정의와 연관지어보고 타겟 단어의 좌측과 우측의 단어들의 윈도우로 'context' 를 정의해보자. 윈도우 크기를 1 을 이용하면, `(context, target)` 쌍의 dataset 을 가지게 된다.

`([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...`

of `(context, target)` pairs. Recall that skip-gram inverts contexts and
targets, and tries to predict each context word from its target word, so the
task becomes to predict 'the' and 'brown' from 'quick', 'quick' and 'fox' from
'brown', etc. Therefore our dataset becomes
skip-gram 은 컨텍스트와 타겟을 뒤집어 버리고, 이들 타겟 단어로부터 각 컨텍스트 단어 예측을 시도한다는 점을 상기하자. 그래서 문제는 'quick' 으로부터 'the' 와 'brown' 을, 'brown' 으로부터 'quick' 과 'fox' 를 예상하는 것이 된다. 따라서 우리의 dataset 은 `(input, output)` 쌍의 dataset이 된다.

`(quick, the), (quick, brown), (brown, quick), (brown, fox), ...`

of `(input, output)` pairs.  The objective function is defined over the entire
dataset, but we typically optimize this with
[stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
(SGD) using one example at a time (or a 'minibatch' of `batch_size` examples,
where typically `16 <= batch_size <= 512`). So let's look at one step of
this process.
목적함수는 전체 dataset 에 대해 정의되지만, 일반적으로 한번에 한 예를 이용한 [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)(SGD) 로 이 목적함수를 최적화 한다.(또는 일반적으로 `16 <= batch_size <= 512` 인 `batch_size` 예제들의 'minibatch') 그럼 이 과정의 한 단계를 살펴보자.

Let's imagine at training step \\(t\\) we observe the first training case above,
where the goal is to predict `the` from `quick`. We select `num_noise` number
of noisy (contrastive) examples by drawing from some noise distribution,
typically the unigram distribution, \\(P(w)\\). For simplicity let's say
`num_noise=1` and we select `sheep` as a noisy example. Next we compute the
loss for this pair of observed and noisy examples, i.e. the objective at time
step \\(t\\) becomes
`quick` 에서 `the` 를 예측하기 위한 목표에 대해 우리가 관찰한 위 첫 학습 케이스의 학습 단계\\(t\\) 를 상상해보자. 우리는 특정 노이즈 분포, 일반적으로 unigram 분포 \\(P(w)\\), 로부터 이끌어 낸 noisy(contrastive) 예제의 `num_noise` 수를 선택했다. 간단하게, noisy 예제에서 `num_noise=1` 이라하고, `sheep` 을 선택한다. 이어서 observed and noisy 예제들의 이 쌍에 대한 loss 를 계산한다, 즉 time step \\(t\\) 에서 목적함수는 아래와 같다.

$$J^{(t)}_\text{NEG} = \log Q_\theta(D=1 | \text{the, quick}) +
  \log(Q_\theta(D=0 | \text{sheep, quick}))$$

The goal is to make an update to the embedding parameters \\(\theta\\) to improve
(in this case, maximize) this objective function.  We do this by deriving the
gradient of the loss with respect to the embedding parameters \\(\theta\\), i.e.
\\(\frac{\partial}{\partial \theta} J_\text{NEG}\\) (luckily TensorFlow provides
easy helper functions for doing this!). We then perform an update to the
embeddings by taking a small step in the direction of the gradient. When this
process is repeated over the entire training set, this has the effect of
'moving' the embedding vectors around for each word until the model is
successful at discriminating real words from noise words.
목표는 이 목적 함수를 향상시키기 위한(여기서는 최대화) embedding parameters \\(\theta\\) 를 업데이트 시키는 것이다. 우리는 embedding parameters \\(\theta\\) 에 대해서 loss 의 gradient 를 미분함으로써 이를 수행했다, 즉 \\(\frac{\partial}{\partial \theta} J_\text{NEG}\\) (다행히 TensorFlow 는 이를 위해 쉬운 헬퍼 함수들을 제공한다!). 다음 gradient 의 방향으로 조금 진행하여 embeddings 의 업데이트를 수행한다. 전체 학습 set에 대해 이 과정을 반복할 때, 모델이 실제 단어들를 노이즈 단어들로부터 구별하는 것을 성공적으로 할 때까지 각 단어 주변 embedding 벡터들을 'moving' 하는 효과를 가진다.

We can visualize the learned vectors by projecting them down to 2 dimensions
using for instance something like the
[t-SNE dimensionality reduction technique](http://lvdmaaten.github.io/tsne/).
When we inspect these visualizations it becomes apparent that the vectors
capture some general, and in fact quite useful, semantic information about
words and their relationships to one another. It was very interesting when we
first discovered that certain directions in the induced vector space specialize
towards certain semantic relationships, e.g. *male-female*, *gender* and
even *country-capital* relationships between words, as illustrated in the figure
below (see also for example
[Mikolov et al., 2013](http://www.aclweb.org/anthology/N13-1090)).
예를 들어 [t-SNE dimensionality reduction technique](http://lvdmaaten.github.io/tsne/). 와 같은 것을 이용하여 학습된 벡터들을 2차원으로 투영(projecting)하여 학습 벡터들을 시각화 할 수 있다. 이들 시각화에 대해 조사하면, 벡터들이 단어들과 그들과 다른 나머지들과의 관계에 대한 일반적인, 사실 꽤 유용하고, semantic 정보를 담는다는 것을 분명히 할 수 있다. 유도된 벡터 공간에서의 특정 방향성은 특정 시멘틱 관계로 특징화되어 연결된다는 우리의 첫 발견은 매우 흥미롭니다, 즉 *male-femal*, *gender*, 그리고 심지어 *country-capital* 단어들 간의 관계, 아래 그림에 도식화하였다.(예제 참고, [Mikolov et al., 2013](http://www.aclweb.org/anthology/N13-1090))
 
<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/linear-relationships.png" alt>
</div>

This explains why these vectors are also useful as features for many canonical
NLP prediction tasks, such as part-of-speech tagging or named entity recognition
(see for example the original work by
[Collobert et al., 2011](http://arxiv.org/abs/1103.0398)
([pdf](http://arxiv.org/pdf/1103.0398.pdf)), or follow-up work by
[Turian et al., 2010](http://www.aclweb.org/anthology/P10-1040)).
이것은 문법적 테그(part-of-speech tagging) 나 개체명 인식과 같은 많은 고전 NLP 예측 문제들에 대해 이들 벡터들이 왜 유용한 features 인지 설명한다(원저작물 인 [Collobert et al., 2011](http://arxiv.org/abs/1103.0398)([pdf](http://arxiv.org/pdf/1103.0398.pdf)) 예제를 참고하거나 후속 연구인 [Turian et al., 2010](http://www.aclweb.org/anthology/P10-1040) 을 참고하자).

But for now, let's just use them to draw pretty pictures!
이제 이들을 이용해서 멋진 그림들을 그려보자!

## Building the Graph
## Graph 만들기

This is all about embeddings, so let's define our embedding matrix.
This is just a big random matrix to start.  We'll initialize the values to be
uniform in the unit cube.

```python
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
```

The noise-contrastive estimation loss is defined in terms of a logistic regression
model. For this, we need to define the weights and biases for each word in the
vocabulary (also called the `output weights` as opposed to the `input
embeddings`). So let's define that.

```python
nce_weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```

Now that we have the parameters in place, we can define our skip-gram model
graph. For simplicity, let's suppose we've already integerized our text corpus
with a vocabulary so that each word is represented as an integer (see
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
for the details). The skip-gram model takes two inputs. One is a batch full of
integers representing the source context words, the other is for the target
words. Let's create placeholder nodes for these inputs, so that we can feed in
data later.

```python
# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
```

Now what we need to do is look up the vector for each of the source words in
the batch.  TensorFlow has handy helpers that make this easy.

```python
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```

Ok, now that we have the embeddings for each word, we'd like to try to predict
the target word using the noise-contrastive training objective.

```python
# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                 num_sampled, vocabulary_size))
```

Now that we have a loss node, we need to add the nodes required to compute
gradients and update the parameters, etc. For this we will use stochastic
gradient descent, and TensorFlow has handy helpers to make this easy as well.

```python
# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
```

## Training the Model

Training the model is then as simple as using a `feed_dict` to push data into
the placeholders and calling
[`session.run`](../../api_docs/python/client.md#Session.run) with this new data
in a loop.

```python
for inputs, labels in generate_batch(...):
  feed_dict = {training_inputs: inputs, training_labels: labels}
  _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
```

See the full example code in
[tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py).

## Visualizing the Learned Embeddings

After training has finished we can visualize the learned embeddings using
t-SNE.

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/tsne.png" alt>
</div>

Et voila! As expected, words that are similar end up clustering nearby each
other. For a more heavyweight implementation of word2vec that showcases more of
the advanced features of TensorFlow, see the implementation in
[tensorflow/models/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py).

## Evaluating Embeddings: Analogical Reasoning

Embeddings are useful for a wide variety of prediction tasks in NLP. Short of
training a full-blown part-of-speech model or named-entity model, one simple way
to evaluate embeddings is to directly use them to predict syntactic and semantic
relationships like `king is to queen as father is to ?`. This is called
*analogical reasoning* and the task was introduced by
[Mikolov and colleagues](http://msr-waypoint.com/en-us/um/people/gzweig/Pubs/NAACL2013Regularities.pdf),
and the dataset can be downloaded from here:
https://word2vec.googlecode.com/svn/trunk/questions-words.txt.

To see how we do this evaluation, have a look at the `build_eval_graph()` and
`eval()` functions in
[tensorflow/models/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py).

The choice of hyperparameters can strongly influence the accuracy on this task.
To achieve state-of-the-art performance on this task requires training over a
very large dataset, carefully tuning the hyperparameters and making use of
tricks like subsampling the data, which is out of the scope of this tutorial.


## Optimizing the Implementation

Our vanilla implementation showcases the flexibility of TensorFlow. For
example, changing the training objective is as simple as swapping out the call
to `tf.nn.nce_loss()` for an off-the-shelf alternative such as
`tf.nn.sampled_softmax_loss()`. If you have a new idea for a loss function, you
can manually write an expression for the new objective in TensorFlow and let
the optimizer compute its derivatives. This flexibility is invaluable in the
exploratory phase of machine learning model development, where we are trying
out several different ideas and iterating quickly.

Once you have a model structure you're satisfied with, it may be worth
optimizing your implementation to run more efficiently (and cover more data in
less time).  For example, the naive code we used in this tutorial would suffer
compromised speed because we use Python for reading and feeding data items --
each of which require very little work on the TensorFlow back-end.  If you find
your model is seriously bottlenecked on input data, you may want to implement a
custom data reader for your problem, as described in
[New Data Formats](../../how_tos/new_data_formats/index.md).  For the case of Skip-Gram
modeling, we've actually already done this for you as an example in
[tensorflow/models/embedding/word2vec.py](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec.py).

If your model is no longer I/O bound but you want still more performance, you
can take things further by writing your own TensorFlow Ops, as described in
[Adding a New Op](../../how_tos/adding_an_op/index.md).  Again we've provided an
example of this for the Skip-Gram case
[tensorflow/models/embedding/word2vec_optimized.py](https://www.tensorflow.org/code/tensorflow/models/embedding/word2vec_optimized.py).
Feel free to benchmark these against each other to measure performance
improvements at each stage.

## Conclusion

In this tutorial we covered the word2vec model, a computationally efficient
model for learning word embeddings. We motivated why embeddings are useful,
discussed efficient training techniques and showed how to implement all of this
in TensorFlow. Overall, we hope that this has show-cased how TensorFlow affords
you the flexibility you need for early experimentation, and the control you
later need for bespoke optimized implementation.
