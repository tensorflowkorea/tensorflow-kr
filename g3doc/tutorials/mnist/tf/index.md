# TensorFlow 메커니즘 기초

코드: [tensorflow/examples/tutorials/mnist/](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/)

이 튜토리얼의 목표는 TensorFlow를 사용해 어떻게 트레이닝 하는지 그리고
전형적인 MNIST 데이터 셋을 사용해 손으로 쓴 숫자를 구별하는 간단한 feed-forward neural network를 평가하는지 보여주는 것이다. 
이 튜토리얼 대상 독자는 TensorFlow 사용에 관심이 있는 머신러닝 유경험자다. 

이 튜토리얼은 일반적인 머신러닝 교육에 적합하지 않다.

반드시 [TensorFlow 설치](../../../get_started/os_setup.md) 지시를 따랐는지 확인하라.

## 튜토리얼 파일

이 튜토리얼은 아래와 같은 파일들을 참조한다:

파일 | 목적
--- | ---
[`mnist.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist.py) | 이 코드는 완전히 연결된 MNIST 모델을 구축한다.
[`fully_connected_feed.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/fully_connected_feed.py) | 메인 코드는 feed dictionary를 사용해 다운로드 한 데이터 셋에 대해 구축된 MNIST모델을 트레이닝 한다. 

트레이닝을 시작하기 위해 직접 `fully_connected_feed.py` 파일을 간단히 실행해 보라:

```bash
python fully_connected_feed.py
```

## 데이터 준비

MNIST는 머신 러닝에서 고전적인 문제다. 이 문제는 그레이 스케일(greyscale)인 손으로 쓴 숫자 28x28 픽셀 이미지를 보고 
그 이미지가 표현하는 숫자가 0 부터 9 까지 숫자 중 어떤 것인지 판단하는 것이다.

![MNIST Digits](../../../images/mnist_digits.png "MNIST Digits")

더 많은 정보는 [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/) 또는
[Chris Olah's visualizations of MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) 참고하라.

### 다운로드

`run_training()` 메소드의 맨 위에는, `input_data.read_data_sets()`
함수가 당신의 트레이닝 폴더에 올바른 데이터가 다운되었는지 확인하고, 
`DataSet` 인스턴스의 딕셔너리에 반환하기 위해 그 데이터의 압축을 해제한다.

```python
data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
```

**주의**: `fake_data` flag 는 유닛 테스트의 목적으로 쓰이며 무시해도 이상이 없다.

데이터 셋 | 목적
--- | ---
`data_sets.train` | 기본 트레이닝을 위한 55000개의 이미지와 레이블.
`data_sets.validation` | 트레이닝 정확도를 반복해서 검증하기 위한 5000개의 이미지와 레이블.
`data_sets.test` | 트레이닝된 정확도를 마지막으로 테스트하기 위한 10000개의 이미지와 레이블.

데이터에 대한 더 많은 정보는 [Download](../../../tutorials/mnist/download/index.md) tutorial을 읽어 보세요.

### 입력과 플레이스 홀더(Placeholders)

`placeholder_inputs()` 함수는 두개의 [`tf.placeholder`](../../../api_docs/python/io_ops.md#placeholder) ops를 생성한다.
이 ops는 `batch_size` 를 포함해, 남은 그래프를 위한 입력 형태와 실제 트레이닝 example의 입력 형태를 정의한다.

```python
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
```

트레이닝 반복 루프 더 아랫부분에서, 전체 이미지와 레이블 데이터셋이 각 순서에서 `batch_size` 에 맞게 
나누어지고 이 플레이스 홀더 ops들과 맞게 매치된다. 그리고나서 `feed_dict` 변수를 사용해 `sess.run()` 함수에 전달된다.

## Build the Graph

데이터를 위한 플레이스 홀더를 생성한 후에, 3-스테이지 패턴(3-stage pattern): 
`inference()`, `loss()`, `training()` 을 따라서 `mnist.py` 파일로부터 그래프가 생성됩니다.

1.  `inference()` - 예측을 위해 network forward 실행에 필요한 수준의 그래프를 작성한다.

2.  `loss()` - inference 그래프에 loss를 생성하기 위해 필요한 ops를 더한다.

3.  `training()` - loss 그래프에 계산과 그라디언트(gradient)를 적용하기 위한 op를 더한다.

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../../images/mnist_subgraph.png">
</div>

### Inference

`inference()` 함수는 그래프를 작성하는데, 이 그래프는 
예측한 출력을 가지는 tensor를 반환하는데 필요한 정도까지 작성된다.

이것은 이미지 플레이스 홀더를 입력으로 취하고 그 위에 출력 logits를 지정한 10 노드 선형 층(ten node linear layer)을 동반하는 
[ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation을 가진 한 쌍의 완전 연결 층(fully connected layer)을 만든다.

각 층은 고유한 [`tf.name_scope`](../../../api_docs/python/framework.md#name_scope) 아래에서 생성된다.
이것은 해당 범위(scope) 안에서 생성된 것에게 접두어와 같은 기능을 한다.

```python
with tf.name_scope('hidden1'):
```

정의된 범위 내, weights와 biases의 층을 필요한 형태로 [`tf.Variable`](../../../api_docs/python/state_ops.md#Variable) 
인스턴스 안에서 생성해 사용한다:

```python
weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights')
biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')
```

예를 들어, 이것들이 `hidden1` 범위 내에서 생성될 때는 weights 변수에 부여된 고유한 이름은 
"`hidden1/weights`"다.

각 변수에게 initializer(초기화) ops가 생성자(construction)의 일부로서 주어져 있다.

보통의 경우에, weights는 [`tf.truncated_normal`](../../../api_docs/python/constant_op.md#truncated_normal)로 초기화 되고
2-D tensor의 형태가 된다. 첫 번째 dim(차원. dimension)은 weights가 연결해 나온 층의 유닛(units) 갯수이고 두 번째 dim은 
weights가 연결한 층의 유닛 갯수이다. 
(the first dim representing the number of units in the layer from which the
weights connect and the second dim representing the number of
units in the layer to which the weights connect.)
`hidden1`이라고 이름붙여진 첫 번째 레이어의 차원은 `[IMAGE_PIXELS, hidden1_units]` 다. 
왜냐하면 weights가 이미지 입력과 hidden1 layer를 연결하고 있기 때문이다.
`tf.truncated_normal` initializer는 주어진 평균과 표준 편차를 가지고 임의의 분포를 생성한다.

Then the biases are initialized with [`tf.zeros`](../../../api_docs/python/constant_op.md#zeros)
to ensure they start with all zero values, and their shape is simply the number
of units in the layer to which they connect.

The graph's three primary ops -- two [`tf.nn.relu`](../../../api_docs/python/nn.md#relu)
ops wrapping [`tf.matmul`](../../../api_docs/python/math_ops.md#matmul)
for the hidden layers and one extra `tf.matmul` for the logits -- are then
created, each in turn, with separate `tf.Variable` instances connected to each
of the input placeholders or the output tensors of the previous layer.

```python
hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
```

```python
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
```

```python
logits = tf.matmul(hidden2, weights) + biases
```

Finally, the `logits` tensor that will contain the output is returned.

### Loss

The `loss()` function further builds the graph by adding the required loss
ops.

First, the values from the `labels_placeholder` are converted to 64-bit integers. Then, a [`tf.nn.sparse_softmax_cross_entropy_with_logits`](../../../api_docs/python/nn.md#sparse_softmax_cross_entropy_with_logits) op is added to automatically produce 1-hot labels from the `labels_placeholder` and compare the output logits from the `inference()` function with those 1-hot labels.

```python
labels = tf.to_int64(labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits, labels, name='xentropy')
```

It then uses [`tf.reduce_mean`](../../../api_docs/python/math_ops.md#reduce_mean)
to average the cross entropy values across the batch dimension (the first
dimension) as the total loss.

```python
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
```

And the tensor that will then contain the loss value is returned.

> Note: Cross-entropy is an idea from information theory that allows us
> to describe how bad it is to believe the predictions of the neural network,
> given what is actually true. For more information, read the blog post Visual
> Information Theory (http://colah.github.io/posts/2015-09-Visual-Information/)

### Training

The `training()` function adds the operations needed to minimize the loss via
[Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent).

Firstly, it takes the loss tensor from the `loss()` function and hands it to a
[`tf.scalar_summary`](../../../api_docs/python/train.md#scalar_summary),
an op for generating summary values into the events file when used with a
`SummaryWriter` (see below).  In this case, it will emit the snapshot value of
the loss every time the summaries are written out.

```python
tf.scalar_summary(loss.op.name, loss)
```

Next, we instantiate a [`tf.train.GradientDescentOptimizer`](../../../api_docs/python/train.md#GradientDescentOptimizer)
responsible for applying gradients with the requested learning rate.

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
```

We then generate a single variable to contain a counter for the global
training step and the [`minimize()`](../../../api_docs/python/train.md#Optimizer.minimize)
op is used to both update the trainable weights in the system and increment the
global step.  This is, by convention, known as the `train_op` and is what must
be run by a TensorFlow session in order to induce one full step of training
(see below).

```python
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
```

The tensor containing the outputs of the training op is returned.

## Train the Model

일단 그래프가 작성되면, 반복해서 트레이닝 할 수 있고 반복 루프(loop)에서 실행할 수 있습니다. 
반복 루프(loop)는 `fully_connected_feed.py` 에 있는 유저 코드에 의해 컨트롤됩니다.

### The Graph

At the top of the `run_training()` function is a python `with` command that
indicates all of the built ops are to be associated with the default
global [`tf.Graph`](../../../api_docs/python/framework.md#Graph)
instance.

```python
with tf.Graph().as_default():
```

A `tf.Graph` is a collection of ops that may be executed together as a group.
Most TensorFlow uses will only need to rely on the single default graph.

More complicated uses with multiple graphs are possible, but beyond the scope of
this simple tutorial.

### The Session

Once all of the build preparation has been completed and all of the necessary
ops generated, a [`tf.Session`](../../../api_docs/python/client.md#Session)
is created for running the graph.

```python
sess = tf.Session()
```

Alternately, a `Session` may be generated into a `with` block for scoping:

```python
with tf.Session() as sess:
```

The empty parameter to session indicates that this code will attach to
(or create if not yet created) the default local session.

Immediately after creating the session, all of the `tf.Variable`
instances are initialized by calling [`sess.run()`](../../../api_docs/python/client.md#Session.run)
on their initialization op.

```python
init = tf.initialize_all_variables()
sess.run(init)
```

The [`sess.run()`](../../../api_docs/python/client.md#Session.run)
method will run the complete subset of the graph that
corresponds to the op(s) passed as parameters.  In this first call, the `init`
op is a [`tf.group`](../../../api_docs/python/control_flow_ops.md#group)
that contains only the initializers for the variables.  None of the rest of the
graph is run here; that happens in the training loop below.

### Train Loop

After initializing the variables with the session, training may begin.

The user code controls the training per step, and the simplest loop that
can do useful training is:

```python
for step in xrange(FLAGS.max_steps):
    sess.run(train_op)
```

However, this tutorial is slightly more complicated in that it must also slice
up the input data for each step to match the previously generated placeholders.

#### Feed the Graph

For each step, the code will generate a feed dictionary that will contain the
set of examples on which to train for the step, keyed by the placeholder
ops they represent.

In the `fill_feed_dict()` function, the given `DataSet` is queried for its next
`batch_size` set of images and labels, and tensors matching the placeholders are
filled containing the next images and labels.

```python
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                               FLAGS.fake_data)
```

A python dictionary object is then generated with the placeholders as keys and
the representative feed tensors as values.

```python
feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
}
```

This is passed into the `sess.run()` function's `feed_dict` parameter to provide
the input examples for this step of training.

#### Check the Status

The code specifies two values to fetch in its run call: `[train_op, loss]`.

```python
for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)
```

Because there are two values to fetch, `sess.run()` returns a tuple with two
items.  Each `Tensor` in the list of values to fetch corresponds to a numpy
array in the returned tuple, filled with the value of that tensor during this
step of training. Since `train_op` is an `Operation` with no output value, the
corresponding element in the returned tuple is `None` and, thus,
discarded. However, the value of the `loss` tensor may become NaN if the model
diverges during training, so we capture this value for logging.

Assuming that the training runs fine without NaNs, the training loop also
prints a simple status text every 100 steps to let the user know the state of
training.

```python
if step % 100 == 0:
    print 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)
```

#### Visualize the Status

In order to emit the events files used by [TensorBoard](../../../how_tos/summaries_and_tensorboard/index.md),
all of the summaries (in this case, only one) are collected into a single op
during the graph building phase.

```python
summary_op = tf.merge_all_summaries()
```

And then after the session is created, a [`tf.train.SummaryWriter`](../../../api_docs/python/train.md#SummaryWriter)
may be instantiated to write the events files, which
contain both the graph itself and the values of the summaries.

```python
summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
```

Lastly, the events file will be updated with new summary values every time the
`summary_op` is run and the output passed to the writer's `add_summary()`
function.

```python
summary_str = sess.run(summary_op, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)
```

When the events files are written, TensorBoard may be run against the training
folder to display the values from the summaries.

![MNIST TensorBoard](../../../images/mnist_tensorboard.png "MNIST TensorBoard")

**NOTE**: For more info about how to build and run Tensorboard, please see the accompanying tutorial [Tensorboard: Visualizing Your Training](../../../how_tos/summaries_and_tensorboard/index.md).

#### Save a Checkpoint

In order to emit a checkpoint file that may be used to later restore a model
for further training or evaluation, we instantiate a
[`tf.train.Saver`](../../../api_docs/python/state_ops.md#Saver).

```python
saver = tf.train.Saver()
```

In the training loop, the [`saver.save()`](../../../api_docs/python/state_ops.md#Saver.save)
method will periodically be called to write a checkpoint file to the training
directory with the current values of all the trainable variables.

```python
saver.save(sess, FLAGS.train_dir, global_step=step)
```

At some later point in the future, training might be resumed by using the
[`saver.restore()`](../../../api_docs/python/state_ops.md#Saver.restore)
method to reload the model parameters.

```python
saver.restore(sess, FLAGS.train_dir)
```

## Evaluate the Model

Every thousand steps, the code will attempt to evaluate the model against both
the training and test datasets.  The `do_eval()` function is called thrice, for
the training, validation, and test datasets.

```python
print 'Training Data Eval:'
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.train)
print 'Validation Data Eval:'
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.validation)
print 'Test Data Eval:'
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.test)
```

> Note that more complicated usage would usually sequester the `data_sets.test`
> to only be checked after significant amounts of hyperparameter tuning.  For
> the sake of a simple little MNIST problem, however, we evaluate against all of
> the data.

### Build the Eval Graph

Before entering the training loop, the Eval op should have been built
by calling the `evaluation()` function from `mnist.py` with the same
logits/labels parameters as the `loss()` function.

```python
eval_correct = mnist.evaluation(logits, labels_placeholder)
```

The `evaluation()` function simply generates a [`tf.nn.in_top_k`](../../../api_docs/python/nn.md#in_top_k)
op that can automatically score each model output as correct if the true label
can be found in the K most-likely predictions.  In this case, we set the value
of K to 1 to only consider a prediction correct if it is for the true label.

```python
eval_correct = tf.nn.in_top_k(logits, labels, 1)
```

### Eval Output

One can then create a loop for filling a `feed_dict` and calling `sess.run()`
against the `eval_correct` op to evaluate the model on the given dataset.

```python
for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
```

The `true_count` variable simply accumulates all of the predictions that the
`in_top_k` op has determined to be correct.  From there, the precision may be
calculated from simply dividing by the total number of examples.

```python
precision = true_count / num_examples
print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
      (num_examples, true_count, precision))
```
