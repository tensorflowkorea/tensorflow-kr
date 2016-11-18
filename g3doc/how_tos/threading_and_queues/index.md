# 쓰레딩(Threading)과 큐(Queues)

Queues are a powerful mechanism for asynchronous computation using TensorFlow.
큐는 TensorFlow 에서 사용하는 비동기 계산에 대한 강력한 메커니즘이다.

Like everything in TensorFlow, a queue is a node in a TensorFlow graph. It's a
stateful node, like a variable: other nodes can modify its content. In
particular, nodes can enqueue new items in to the queue, or dequeue existing
items from the queue.
TensorFlow의 다른 모든 것들 처럼, 하나의 큐는 TensorFlow 그래프의 하나의 노드입니다. 이것은 변수(variable)와 비슷한, 상태저장 노드이다: 다른 노드들은 그 저장물(콘텐츠)의 수정이 가능하다. 특히, 노드들은 큐에 새로운 아이템들을 추가할 수 있거나 큐에 존재하는 아이템들을 해제할 수 있다.

To get a feel for queues, let's consider a simple example. We will create a
"first in, first out" queue (`FIFOQueue`) and fill it with zeros.
Then we'll construct a graph
that takes an item off the queue, adds one to that item, and puts it back on the
end of the queue. Slowly, the numbers on the queue increase.
큐에 대한 감을 잡기 위해, 간단한 예제를 고려해보자. 우리는 "first in, first out" 큐(`FIFOQueus`) 를 만들어볼 것이고, 이를 0 값으로 채울 것이다. 다음은 큐에서 아이템을 제거, 아이템을 추가, 그리고 큐 끝에 이를 다시 넣는 그래프를 만들것이다. 그럼 천천히, 큐의 숫자들이 증가한다.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/IncremeterFifoQueue.gif">
</div>

`Enqueue`, `EnqueueMany`, and `Dequeue` are special nodes. They take a pointer
to the queue instead of a normal value, allowing them to change it. We recommend
you think of these as being like methods of the queue. In fact, in the Python
API, they are methods of the queue object (e.g. `q.enqueue(...)`).
`Enqueue`, `EnqueueMany`, 그리고 `Dequeue` 는 특별 노드들이다. 이들은 평범한 값 대신 큐에 대한 포인터를 가지며, 이들 포인터를 변경할 수 있다. 우리는 당신이 큐의 방법과 같은 것들에 대해 생각해보길 권한다. 사실, Python API 에서, 이들은 큐 객체에 대한 방법들이다(`q.enqueue(...)`).

**N.B.** Queue methods (such as `q.enqueue(...)`) *must* run on the same device
as the queue. Incompatible device placement directives will be ignored when
creating these operations.
**주의** 큐 방법들(`q.enqueue(...)`와 같은) 큐에 대해 *반드시* 같은 장치에 대해 실행해야 한다. 이들 연산이 생성될 때 호환성이 없는 장치 배치 지시들(Incompatible device placement directives) 은 무시된다.

Now that you have a bit of a feel for queues, let's dive into the details...
이제 큐에 대한 감을 조금 가졌을 것이고, 자세한 부분으로 들어가보자...

## Queue Use Overview
## 큐 사용 개요

Queues, such as `FIFOQueue` and `RandomShuffleQueue`, are important TensorFlow
objects for computing tensors asynchronously in a graph.
큐, `FIFOQueue` 와 `RandomShffleQueue` 와 같은, 들은 그래프에서 비동기로 tensor 들을 계산하기 위한 중요한 TensorFlow 객체이다.

For example, a typical input architecture is to use a `RandomShuffleQueue` to
prepare inputs for training a model:
예를 들어, 대표적인 입력 아키텍쳐는 모델 학습을 위한 입력의 준비를 위해 `RandomShuffleQueue`를 사용해야 한다:

* Multiple threads prepare training examples and push them in the queue.
* A training thread executes a training op that dequeues mini-batches from the
  queue
* 대수의 쓰레드는 학습 예제들을 대비하고 이들을 큐에 넣는다.
* 학습하는 쓰레드는 큐에서 mini-batches 를 빼는 학습 연산을 실행한다.

This architecture has many benefits, as highlighted in the
[Reading data how to](../reading_data), which also gives an overview of
functions that simplify the construction of input pipelines.
이 아키텍쳐는, 입력 파이프라인들의 구성을 간략화하는 함수들의 개요를 알려주는 [Reading data how to](../reading_data) 에서 강조되었던 것처럼, 많은 이점들을 가진다.

The TensorFlow `Session` object is multithreaded, so multiple threads can
easily use the same session and run ops in parallel.  However, it is not always
easy to implement a Python program that drives threads as described above.  All
threads must be able to stop together, exceptions must be caught and
reported, and queues must be properly closed when stopping.
TensorFlow `Session` 객체는 멀티 쓰레드화 되어있다. 그래서 다수 쓰레드들은 같은 세션을 쉽게 사용할 수 있고 병렬로 연산들을 실행할 수 있다. 그러나, 위에서 묘사된 것처럼 쓰레드들을 다루는 Python 프로그램을 구현하는 것이 항상 쉽지만은 않다. 모든 쓰레드들은 함께 멈춰질 수 있어야 하며, 예외처리들은 처리되어야 하고 알려져야 한다. 그리고 큐는 멈춰야 할 때 적절하게 종료되어야 한다.

TensorFlow provides two classes to help:
[tf.Coordinator](../../api_docs/python/train.md#Coordinator) and
[tf.QueueRunner](../../api_docs/python/train.md#QueueRunner). These two classes
are designed to be used together. The `Coordinator` class helps multiple threads
stop together and report exceptions to a program that waits for them to stop.
The `QueueRunner` class is used to create a number of threads cooperating to
enqueue tensors in the same queue.
TensorFlow 는 도움을 주는 두 클래스들을 제공한다:
[tf.Coordinator](../../api_docs/python/train.md#Coordinator) 와 [tf.QueueRunner](../../api_docs/python/train.md#QueueRunner). 이들 두 클래스들은 함께 사용되기 위해 디자인되었다. `Coordinator` 클래스는 멀티 쓰레드들이 함께 정지되도록 돕고 예외처리들을 그들이 정지되기 위해 대기하는 프로그램에 알린다. `QueueRunner` 클래스는 같은 큐 안의 tensors 를 추가하기 위해 협력하는 많은 쓰레드들을 생성한다.

## Coordinator
## 조정자(Coordinator)

The Coordinator class helps multiple threads stop together.
조정자 클래스는 멀티 쓰레드들이 같이 정지되도록 한다.

Its key methods are:
이것의 핵심 방법들은 아래와 같다:

* `should_stop()`: returns True if the threads should stop.
* `request_stop(<exception>)`: requests that threads should stop.
* `join(<list of threads>)`: waits until the specified threads have stopped.
* `should_stop()`: 쓰레드들이 정지되어야 한다면 True 값을 반환한다.
* `request_stop(<exception>)`: 쓰레드들이 정지되어야 함을 요청한다.
* `join(<list of threads>): 특정 쓰레드들이 멈출 때 까지 대기한다.

You first create a `Coordinator` object, and then create a number of threads
that use the coordinator.  The threads typically run loops that stop when
`should_stop()` returns `True`.
당신은 우선 `Coordinator` 객체를 생성하고, 다음으로 coordinator 로 사용하는 쓰레드들을 생성한다. 일반적으로 쓰레드들은 `should_stop()` 이 `True` 를 반환할 때 멈추는 루프를 실행한다.

Any thread can decide that the computation should stop.  It only has to call
`request_stop()` and the other threads will stop as `should_stop()` will then
return `True`.
어떤 쓰레드이는 계산을 멈춰야 하는 것을 결정할 수 있다. 이것은 `request_stop()` 함수를 불르는 것이고 다른 쓰레드들은 `should_stop()` 함수가 `True` 값을 반환한 다음 정지된다.

```python
# Thread body: loop until the coordinator indicates a stop was requested.
# If some condition becomes true, ask the coordinator to stop.
# 쓰레드 : coordinator 가 정지가 요청됨을 알릴 때까지 반복
# 어떤 조건이 true 가 이면, coordinator 가 멈출 것을 요청
def MyLoop(coord):
  while not coord.should_stop():
    ...do something...
    if ...some condition...:
      coord.request_stop()

# Main code: create a coordinator.
# Main code: coordinator 생성
coord = Coordinator()

# Create 10 threads that run 'MyLoop()'
# 'MyLoop()' 를 실행하는 10개의 쓰레드를 생성
threads = [threading.Thread(target=MyLoop, args=(coord,)) for i in xrange(10)]

# Start the threads and wait for all of them to stop.
# 쓰레드들을 시작하고 그들 모두의 정지를 대기
for t in threads: t.start()
coord.join(threads)
```

Obviously, the coordinator can manage threads doing very different things.
They don't have to be all the same as in the example above.  The coordinator
also has support to capture and report exceptions.  See the [Coordinator class](../../api_docs/python/train.md#Coordinator) documentation for more details.
분명히, coordinator 는 다양한 처리를 하는 쓰레드들을 관리할 수 있다. 그들은 위 예제에서와 같이 모두 같을 수는 없다. 또한 coordinator 는 예외처리들을 감지하고 알린다. 자세한 것은 [Coordinator class](../../api_docs/python/train.md#Coordinator) 문서를 살펴보자.

## QueueRunner
## QueueRunner

The `QueueRunner` class creates a number of threads that repeatedly run an
enqueue op.  These threads can use a coordinator to stop together.  In
addition, a queue runner runs a *closer thread* that automatically closes the
queue if an exception is reported to the coordinator.
`QueueRunner` 클래스는 enqueue 연산을 반복적으로 실행하는 쓰레드들을 생성한다. 이들 쓰레드들은 coordinator 를 이용해 동시에 정지하도록 할 수 있다. 추가로, queue runner 는 예외처리가 coordinator 에 보고되었다면 자동적으로 queue 를 종료하는 *closer thread* 를 실행한다.

You can use a queue runner to implement the architecture described above.
당신은 위에 설명된 아키텍처 구현을 위해 queue runner 를 사용할 수 있다.

First build a graph that uses a `Queue` for input examples.  Add ops that
process examples and enqueue them in the queue.  Add training ops that start by
dequeueing from the queue.
우선 입력 예제들에 대한 `Queue`를 사용하는 그래프를 만든다. 예제들을 처리하고 큐에 이들을 추가하는 연산을 추가한다. 큐에서 해제하는 것을 시작하는 학습 연산들을 추가한다.

```python
example = ...ops to create one example...
example = ...예제를 생성하는 연산들...
# Create a queue, and an op that enqueues examples one at a time in the queue.
# 큐와 큐에 차례로 예제들을 추가하는 연산을 생성한다.
queue = tf.RandomShuffleQueue(...)
enqueue_op = queue.enqueue(example)
# Create a training graph that starts by dequeuing a batch of examples.
# 예제들을 큐에서 해제하도록 하는 학습 그래프를 생성한다.
inputs = queue.dequeue_many(batch_size)
train_op = ...use 'inputs' to build the training part of the graph...
train_op = ...그래프의 학습 부분을 만들기 위한 'inputs' 을 사용
```

In the Python training program, create a `QueueRunner` that will run a few
threads to process and enqueue examples.  Create a `Coordinator` and ask the
queue runner to start its threads with the coordinator.  Write a training loop
that also uses the coordinator.
Python 학습 프로그램에서, 예제들을 처리하고 큐에 추가하기 위한 쓰레드들을 실행하는 `QueueRunner` 를 생성한다. `Coordinator` 를 생성하고 queue runner 가 coordinator 와 함께 이들의 쓰레드들을 실행하는 것을 요청한다. coordinator 를 사용하는 학습 루프를 적어보자.

```
# Create a queue runner that will run 4 threads in parallel to enqueue
# examples.
# 예제들을 큐에 병렬로 추가하기 위한 4 쓰레드를 실행하는 queue runner 를 생성한다.
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

# Launch the graph.
# 그래프 실행
sess = tf.Session()
# Create a coordinator, launch the queue runner threads.
# queue runner 쓰레드들을 실행하는 coordinator 생성
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
# Run the training loop, controlling termination with the coordinator.
# coordinator 와 함께 종료를 제어하는 학습 루프를 실행
for step in xrange(1000000):
    if coord.should_stop():
        break
    sess.run(train_op)
# When done, ask the threads to stop.
# 완료 후, 쓰레드에 정지를 요청
coord.request_stop()
# And wait for them to actually do it.
# 실제 정지하기를 대기
coord.join(threads)
```

## Handling Exceptions
## 예외처리 다루기

Threads started by queue runners do more than just run the enqueue ops.  They
also catch and handle exceptions generated by queues, including
`OutOfRangeError` which is used to report that a queue was closed.
queue runner 에 의해 시작된 쓰레드들은 그냥 연산들을 큐에 추가하도록 실행하는 것보다 더 많이 추가한다. 또한 이들은 큐가 닫혔음을 알리기 위해 사용되는 `OutOfRangeError` 를 포함하여, 큐에 의해 생성된 예외처리들을 포착하고 처리한다.

A training program that uses a coordinator must similarly catch and report
exceptions in its main loop.
coordinator 를 사용하는 학습 프로그램은 마찬가지로 이들의 메인 루프에서 예외처리들을 포착하고 알려야 한다.

Here is an improved version of the training loop above.
아래는 위 학습 루프에 대한 향상된 버전이다.

```python
try:
    for step in xrange(1000000):
        if coord.should_stop():
            break
        sess.run(train_op)
except Exception, e:
    # Report exceptions to the coordinator.
    coord.request_stop(e)
finally:
    # Terminate as usual.  It is innocuous to request stop twice.
    coord.request_stop()
    coord.join(threads)
```
