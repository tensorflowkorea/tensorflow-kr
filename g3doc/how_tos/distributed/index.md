# 분산환경 텐서플로우

이 문서는 텐서플로우 서버 클러스터를 생성하고, 클러스터 상에서 분산 처리를 수행하는 방법에 대하여 설명한다.  이 문서의 독자들은 텐서플로우를 이용한 [기본적인 프로그래밍 개념](../../get_started/basic_usage)은 숙지가 되어있다고 가정한다 .

## 안녕 분산환경 텐서플로우!

 텐서플로우 클러스터의 간단한 동작을 살펴보기 위해서는 아래 예제를 실행해보아라. 

```shell
# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

[`tf.train.Server.create_local_server()`](../../api_docs/python/train.md#Server.create_local_server) 메소드는  in-process 서버 형태로 단일 프로세스 클러스터를 생성한다.

## 클러스터 생성하기

텐서플로우에서 "클러스터"란 텐서플로우 그래프 상에서의 분산 연산의 일부로서 "작업(Task)"의 집합을 의미한다. 각각의 작업은 텐서플로우의 "서버"에 연관되어 있으며, 각 서버는 세션을 생성할 수 있는 "마스터"와 그래프상에서 연산을 수행하는 "워커"로 구성된다.   각 클러스터는 복 수개의 "업무(Jobs)"로 구성되어 있으며, 각각의 업무는 복 수개의 작업으로 이루어져 있다.

클러스터를 생성하기 위해서는, 작업 하나당 하나의 텐서플로우 서버를 실행해야한다. 각 작업은 보통 서로 다른 머신에서 실행되지만, 하나의 머신에서 여러개의 작업을 실행하는 것도 가능하다.(예를 들어 복 수개의 GPU를 사용하는 경우). 각 작업마다 아래의 절차를 따라서 진행된다.

1.  클러스터에 할당된 작업을 설명하는 **`tf.train.ClusterSpec`  을 생성**하라. 이 것은 각 작업마다 동일해야 한다.

2.  `tf.train.ClusterSpec` 를 생성자 인자로 넘겨주어 **`tf.train.Server` 를 생성**하고, 로컬 작업을 업무 이름과 작업 인덱스로 식별하라. 

     ​


### 클러스터를 설명하는 `tf.train.ClusterSpec` 생성하기

클러스터 명세 딕셔너리는 업무 이름과 네트워크 주소를 매핑한다. 아래와 같이 딕셔너리를 `tf.train.ClusterSpec` 의 생성자의 인자로 넘겨 주어라.

<table>
  <tr><th><code>tf.train.ClusterSpec</code> 생성자</th><th>사용 가능 작업</th>
  <tr>
    <td><pre>
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
</pre></td>
<td><code>/job:local/task:0<br/>/job:local/task:1</code></td>
  </tr>
  <tr>
    <td><pre>
tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222", 
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
</pre></td><td><code>/job:worker/task:0</code><br/><code>/job:worker/task:1</code><br/><code>/job:worker/task:2</code><br/><code>/job:ps/task:0</code><br/><code>/job:ps/task:1</code></td>
  </tr>
</table>

### 각 작업 마다 `tf.train.Server` 객체 생성하기

[`tf.train.Server`](../../api_docs/python/train.md#Server)  객체는 여러개의 로컬 디바이스 정보와, 각 작업과 디바이스를 연결해주는 정보인 `tf.train.ClusterSpec` , 분산 연산 수행에 이용되는 ["session target"](../../api_docs/python/client.md#Session) 을 포함하고 있다. 

각각의 서버는 특정한 이름이 부여된 업무의 멤버이며, 해당 업무에서의 작업 인덱스를 가지고 있다.  서버는 클러스터내에 있는 다른 서버와 통신이 가능하다.

예를 들어,  `localhost:2222` `localhost:2223` 두 개의 서버를 가진 클러스터를 구동하려면 밑의 두 코드를 로컬머신의 다른 두 개의 프로세스에서 실행하면 된다.

```python
# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)
```
```python
# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```

**Note:**  거대한 클러스터를 생성하기 위해서 클러스터 명세를 일일히 수동으로 작성하는 것은 지루한 일일 수 있다. 우리는 이러한 작업을 프로그래밍으로 구동하기 위한 도구를 고민중이다. 예를 들어, [Kubernetes](http://kubernetes.io) 와 같은 클러스터 매니저를 사용하는 방법이다. 만약 우리가 지원하기를 원하는 클러스터 매니저가 있다면, [GitHub issue](https://github.com/tensorflow/tensorflow/issues) 에 제보해주길 바란다.

## 모델 내 에서 디바이스 명시하기

연산을 CPU 혹은 GPU에서 수행할지 선택할 때와 마찬가지로, 특정 연산에 연산을 수행할 디바이스를 명시할 때는     [`tf.device()`](../../api_docs/python/framework.md#device) 를 사용하면 된다. 

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```

위의 예제에서, 변수들은 `ps` 업무(job)에서 생성이 되고, 연산이 집중적으로 일어나는 모델은 `worker` 에서 생성이 된다. 텐서플로우는 각각의 업무간에 적절하게 데이터를 이동시켜준다.(정방향 연산시에는 `ps` 에서 `worker` 로 gradients 전파시에는 `worker` 에서 `ps`로 전달한다.)

## 훈련 복제

일반적으로 ''데이터 병렬화(data parallelism)'' 로 명명되는 훈련 방식은  `worker`업무의 여러개의 작업이 하나의 모델에 대하여, 데이터의 각기 다른 일부를 이용하여 `ps` 에서 생성된 공유변수를 병렬적으로 업데이트 시킨다. 모든 작업들은 각각 다른 머신에서 동작한다. 텐서플로우에서 이러한 훈련 방식을 구현하는 방법은 여러가지가 있는데, 우리는 복제된 모델을 간단하게 명시할 수 있도록 도와주는 라이브러리를 구축하였다. 시도 가능한 방법은 아래와 같다:

* **그래프내 복제(In-graph replication).** 이 방법에서 클라이언트는 한 세트의 변수( `/job:ps`에 연관된 `tf.Variable` )가 포함 된 `tf.Graph` 하나를 구축하고, `/job:worker` 에 소속된 서로 다른 작업에 각각 연관된 여러개의 연산 집중 모델을 복제하여 구축한다.

* **그래프간 복제(Between-graph replication).** In this approach, there is a separate client
  for each `/job:worker` task, typically in the same process as the worker
  task. Each client builds a similar graph containing the parameters (pinned to
  `/job:ps` as before using
  [`tf.train.replica_device_setter()`](../../api_docs/python/train.md#replica_device_setter)
  to map them deterministically to the same tasks); and a single copy of the
  compute-intensive part of the model, pinned to the local task in
  `/job:worker`.

* **Asynchronous training.** In this approach, each replica of the graph has an
  independent training loop that executes without coordination. It is compatible
  with both forms of replication above.

* **Synchronous training.** In this approach, all of the replicas read the same
  values for the current parameters, compute gradients in parallel, and then
  apply them together. It is compatible with in-graph replication (e.g. using
  gradient averaging as in the
  [CIFAR-10 multi-GPU trainer](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py)),
  and between-graph replication (e.g. using the
  `tf.train.SyncReplicasOptimizer`).

### Putting it all together: example trainer program

The following code shows the skeleton of a distributed trainer program,
implementing **between-graph replication** and **asynchronous training**. It
includes the code for the parameter server and worker tasks.

```python
import tensorflow as tf

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  
  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      loss = ...
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      step = 0
      while not sv.should_stop() and step < 1000000:
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        _, step = sess.run([train_op, global_step])

    # Ask for all the services to stop.
    sv.stop()

if __name__ == "__main__":
  tf.app.run()
```

To start the trainer with two parameter servers and two workers, use the
following command line (assuming the script is called `trainer.py`):

```shell
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1
```

## Glossary

<dl>
  <dt>Client</dt>
  <dd>
    A client is typically a program that builds a TensorFlow graph and
    constructs a `tensorflow::Session` to interact with a cluster. Clients are
    typically written in Python or C++. A single client process can directly
    interact with multiple TensorFlow servers (see "Replicated training" above),
    and a single server can serve multiple clients.
  </dd>
  <dt>Cluster</dt>
  <dd>
    A TensorFlow cluster comprises a one or more "jobs", each divided into lists
    of one or more "tasks". A cluster is typically dedicated to a particular
    high-level objective, such as training a neural network, using many machines
    in parallel. A cluster is defined by a `tf.train.ClusterSpec` object.
  </dd>
  <dt>Job</dt>
  <dd>
    A job comprises a list of "tasks", which typically serve a common
    purpose. For example, a job named `ps` (for "parameter server") typically
    hosts nodes that store and update variables; while a job named `worker`
    typically hosts stateless nodes that perform compute-intensive tasks.
    The tasks in a job typically run on different machines. The set of job roles
    is flexible: for example, a `worker` may maintain some state.
  </dd>
  <dt>Master service</dt>
  <dd>
    An RPC service that provides remote access to a set of distributed devices,
    and acts as a session target. The master service implements the
    <code>tensorflow::Session</code> interface, and is responsible for
    coordinating work across one or more "worker services". All TensorFlow
    servers implement the master service.
  </dd>
  <dt>Task</dt>
  <dd>
    A task corresponds to a specific TensorFlow server, and typically
    corresponds to a single process. A task belongs to a particular "job" and is
    identified by its index within that job's list of tasks.
  </dd>
  <dt>TensorFlow server</dt>
  <dd>
    A process running a <code>tf.train.Server</code> instance, which is a
    member of a cluster, and exports a "master service" and "worker service".
  </dd>
  <dt>Worker service</dt>
  <dd>
    An RPC service that executes parts of a TensorFlow graph using its local
    devices. A worker service implements <a href=
    "https://www.tensorflow.org/code/tensorflow/core/protobuf/worker_service.proto"
    ><code>worker_service.proto</code></a>. All TensorFlow servers implement the
    worker service.
  </dd>
</dl>
