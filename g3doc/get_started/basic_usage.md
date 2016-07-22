# Basic Usage

TensorFlow 를 사용하기 위해서는 TensorFlow 가 어떻게 동작하는지 이해할 필요가 있다.

* `Graphs`는 컴퓨터 연산을 나타냅니다.
* `Sessions`이라는 컨텍스트 안에서 그래프는 실행 됩니다.
* `Tensors`는 데이터를 나타냅니다.
* `Variables`은 상태를 유지합니다.
* `Feeds & Fetches`은 임의의 작업의 데이터 입출력에 사용 됩니다.

## Overview

TensorFlow 의 프로그램 시스템은 `Graphs`를 통해서 컴퓨터 연산을 나타낸다.  `Graph` 안에서 노드는
*ops* (작은 단위의 동작) 불리우며, 한개의 OP는 제로 또는 하나 이상의 `Tensors` 가지게 된다.
OP는 몇 가지의 연산을 하기 되고, 제로 또는 하나 이상의 `Tensors` 만들어 냅니다.
`Tensor` 는 다차원 배열을 타입으로 가지게 된다.
예를 들어, 이미지를 다음과 같이 `[batch, height, width, channels]`4차원 배열로 나타낼 수 있다.

TensorFlow의 graph는 컴퓨터 연산의 *description* 설명이다. 어떤것을 연산하든, graph는
반드시 `Session` 안에서 실행되어야 한다. `Session`은 graph 작은 단위 연산들을 CPUs 또는 GPUs
불리우는 `Devices`에 배치시키고, 작은 단위 연산(ops)들이 작동할수 있는 기능을 제공한다.
이런 기능은 단위 연산(ops)에 의해서 tensors 들이 만들어진다. 파이션에서는 `ndarray` 오프젝트로
C 와 C++ 에서는 `TensorFlow::Tensor` 오프젝트를 만들어내게 된다.

## The computation graph

TensorFlow 프로그램들은 대개 graph를 조립하는 '구성 단계'와 session을 이용해 graph 안에
작은 단위의 연산(ops)을 실행시키는 '실행 단계'로 구성돼 있다.

예를 들어, 일반적으로 '구성 단계'에선 neural network를 대표하고 훈련시키기 위한 graph를 만들고,
'실행 단계'에선 트레이닝할 작은 단위의 연산(ops) 세트를 session을 이용해 반복 실행 시킨다.

TensorFlow는 C, C++, Python에서 사용할 수 있다. 현재, Python 라이브러리에서 C/C++에서 제공하지 않는
많은 유용한 함수들을 제공하고 있어 Python을 사용하는 것이 graph를 조립하는데 더 편할 것이다.

session 라이브러리는 세 언어에서 동등한 기능을 사용할 수 있다.

### Building the graph
graph를 만드는 것은 `Constant`와 같이 어떠한 input도 필요하지 않는 단위의 동작(ops)으로 시작한다.
Python 라이브러리에서 단위 연산(ops) 생성자는 구성된 단위 연산(ops)의 결과(output)를 대기하는
객체를 반환한다. 그리고 이 객체들은 다른 단위 연산(ops) 생성자의 input으로 전달할 수 있다.

Python 라이브러리로 사용하는 TensorFlow는 단위 연산(ops) 생성자가 노드를 추가한
*graph* 를 가지고 된다. *graph* 는 많은 어플리케이션용으로 충분하다.
[Graph class](../api_docs/python/framework.md#Graph) documentation에서 어떻게 많은 graph를
명시적으로 관리할 수 있는지 알 수 있다.

```python
import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)
```

예시 graph는 3개의 노드(`constant()` ops 2개와 `matmul()` ops 한개)를 가지고 있다.
실제 매트릭스들을 곱하고 곱셈한 연산의 결과를 얻기 위해선, session에서 graph를 실행해야 한다.

### Launching the graph in a session

아래와 같은 구성으로 동작하게 된다. 'graph' 동작 시키기 위해서는 `Session` 을 만든다.
예시 graph 에서는 변수 없이 session 이 동작하게 된다.

모든 session API 는 [Session class](../api_docs/python/client.md#session-management)
에서 볼수 있다.

```python
# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()
```

Sessions 은 자원을 해체하기 위해서 close()을 사용해야 한다. 또한 `Session`을 "with" 블럭 안에서
사용할 수 있다. `with` 블럭이 끝날때 `Session`은 자동적으로 자원을 해체하고 close 된다.

```python
with tf.Session() as sess:
  result = sess.run([product])
  print(result)
```

TensorFlow는 graph 에 정의된 단위 연산들이 컴퓨터 자원(CPU or GPU)을 분배해서 사용 할수 있게
구현되어 있다.
만약에 CPU or GPU 를 명시적으로 지정하지 않는다면, TensorFlow 는 당신이 가지고 있는 첫번째 GPU를
사용하게 될것이다. 이것은 보다 많은 연산처리를 가능하게 한다.

만약에 당신의 컴퓨터에 하나 이상의 GPU가 있다면, 당신은 명시적으로 GPU를 지정해서 사용할수 있다.
`with...Device` 라는 문법을 사용해서 연산에 사용될 CPU or GPU를 지정할 수 있다.

```python
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
```

CPU or GPU 는 문자열로 지정 한다. 현재 제공하는 CPU or GPU는 아래와 같다.

*  `"/cpu:0"`: 컴퓨터의 CPU.
*  `"/gpu:0"`: 컴퓨터의 1번째 GPU.
*  `"/gpu:1"`: 컴퓨터의 2번쨰 GPU.

GPU 와 TensorFlow 보다 많은 정보는 [Using GPUs](../how_tos/using_gpu/index.md) 보면 된다.

### Launching the graph in a distributed session

TensorFlow 클러스터 만들기, TensorFlow 는 클러스터 안의 여러 머신에서 동작 시킬수 있다.
너의 클라이언의 Session을 인스턴스화 시키고, 클러스터 안의 머신 네트워크에 보내면 된다.

```python
with tf.Session("grpc://example.org:2222") as sess:
  # Calls to sess.run(...) will be executed on the cluster.
  ...
```
해당 머신은 현재 Session의 마스터가 된다. 마스터는 클러스터(workers) 안의 여러 머신들과 graph 를
교차해서 분배하게 된다. 이런 분배는 머신의 사용 가능한 컴퓨팅 자원을 고려해서 이러어 진다.

"with tf.device():" 문법을 이용해서 특정 머신에게 graph의 특정 연산을 지정해 줄 수 있다.

```python
with tf.device("/job:ps/task:0"):
  weights = tf.Variable(...)
  biases = tf.Variable(...)
```
Session 과 클러스터 의 분산처리에 대한 더 많은 정보는 [Distributed TensorFlow How To](../how_tos/distributed/)
볼 수 있다.

## Interactive Usage

The Python examples in the documentation launch the graph with a
[`Session`](../api_docs/python/client.md#Session) and use the
[`Session.run()`](../api_docs/python/client.md#Session.run) method to execute
operations.

For ease of use in interactive Python environments, such as
[IPython](http://ipython.org) you can instead use the
[`InteractiveSession`](../api_docs/python/client.md#InteractiveSession) class,
and the [`Tensor.eval()`](../api_docs/python/framework.md#Tensor.eval) and
[`Operation.run()`](../api_docs/python/framework.md#Operation.run) methods.  This
avoids having to keep a variable holding the session.

```python
# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2. -1.]

# Close the Session when we're done.
sess.close()
```

## Tensors

TensorFlow programs use a tensor data structure to represent all data -- only
tensors are passed between operations in the computation graph. You can think
of a TensorFlow tensor as an n-dimensional array or list. A tensor has a
static type, a rank, and a shape.  To learn more about how TensorFlow handles
these concepts, see the [Rank, Shape, and Type](../resources/dims_types.md)
reference.

## Variables

Variables maintain state across executions of the graph. The following example
shows a variable serving as a simple counter.  See
[Variables](../how_tos/variables/index.md) for more details.

```python
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.initialize_all_variables()

# Launch the graph and run the ops.
with tf.Session() as sess:
  # Run the 'init' op
  sess.run(init_op)
  # Print the initial value of 'state'
  print(sess.run(state))
  # Run the op that updates 'state' and print 'state'.
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))

# output:

# 0
# 1
# 2
# 3
```

The `assign()` operation in this code is a part of the expression graph just
like the `add()` operation, so it does not actually perform the assignment
until `run()` executes the expression.

You typically represent the parameters of a statistical model as a set of
Variables. For example, you would store the weights for a neural network as a
tensor in a Variable. During training you update this tensor by running a
training graph repeatedly.

## Fetches

To fetch the outputs of operations, execute the graph with a `run()` call on
the `Session` object and pass in the tensors to retrieve. In the previous
example we fetched the single node `state`, but you can also fetch multiple
tensors:

```python
input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)

# output:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]
```

All the ops needed to produce the values of the requested tensors are run once
(not once per requested tensor).

## Feeds

The examples above introduce tensors into the computation graph by storing them
in `Constants` and `Variables`. TensorFlow also provides a feed mechanism for
patching a tensor directly into any operation in the graph.

A feed temporarily replaces the output of an operation with a tensor value.
You supply feed data as an argument to a `run()` call. The feed is only used for
the run call to which it is passed. The most common use case involves
designating specific operations to be "feed" operations by using
tf.placeholder() to create them:

```python

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# output:
# [array([ 14.], dtype=float32)]
```

A `placeholder()` operation generates an error if you do not supply a feed for
it. See the
[MNIST fully-connected feed tutorial](../tutorials/mnist/tf/index.md)
([source code](https://www.tensorflow.org/code/tensorflow/g3doc/tutorials/mnist/fully_connected_feed.py))
for a larger-scale example of feeds.
