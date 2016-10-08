# 변수: 생성, 초기화, 저장, 복구

모델을 학습 시킬 때, [변수(Variables)](../../api_docs/python/state_ops.md)
를 매개 변수 업데이트와 유지를 위해 사용합니다. 변수는 텐서를 포함하는 인-메모리 버퍼입니다.
변수는 반드시 명시적으로 초기화해야 합니다. 그리고 학습 중 혹은 학습 후에 디스크에 저장할 수 있습니다.
나중에 저장된 값들을 모델 실행이나 분석을 위해 복원 가능합니다.

이 문서는 다음 TensorFlow 클래스를 참조합니다.
링크를 따라가 API에 대한 자세한 설명이 있는 문서를 살펴보세요.

* [`tf.Variable`](../../api_docs/python/state_ops.md#Variable) 클래스.
* [`tf.train.Saver`](../../api_docs/python/state_ops.md#Saver) 클래스.


## 생성

[변수(Variable)](../../api_docs/python/state_ops.md) 를 생성할 때 `Variable()` 생성자의 초기 값으로 `Tensor` 를 전달받게 됩니다.  
TensorFlow는 [상수(constants) 또는 임의(random)의 값](../../api_docs/python/constant_op.md) 으로 초기화 하는 다양한 명령어를 제공합니다.

이 모든 명령어는 `Tensor` 들의 형태(shape)을 지정해줘야 합니다. 이 형태는 자동적으로 변수(Variable)의 형태가 됩니다. 변수는 대부분 고정된 형태를 가집니다, 그렇지만 TensorFlow에서는
변수의 형태를 수정하기 위한(reshape) 고급 매커니즘을 제공합니다.

```python
# 두 변수를 생성.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
```

`tf.Variable()`호출은 그래프에 여러 오퍼레이션을 추가합니다:

* `variable` 오퍼레이션은 그 변수의 값을 가지고 있습니다.
* `tf.assign` 오퍼레이션은 초기값으로 변수를 설정합니다.
* 초기값을 위한 오퍼레이션으로는 위에 있는 예제의 `biases` 변수의 `zeros` 등도 그래프에 추가 됩니다.

`tf.Variable()`에서 반환되는 값은 Python 클래스의 인스턴스인 `tf.Variable` 입니다.

### 디바이스에 변수 배치하기

변수는 생성될때 [`with tf.device(...):`](../../api_docs/python/framework.md#device) 블록을 사용하여 특수한 디바이스에 배치할 수 있습니다.

```python
# 변수를 CPU에 배치
with tf.device("/cpu:0"):
  v = tf.Variable(...)

# 변수를 GPU에 배치.
with tf.device("/gpu:0"):
  v = tf.Variable(...)

# 변수를 특정 서버 테스크(task)에 배치.
with tf.device("/job:ps/task:7"):
  v = tf.Variable(...)
```

**주의** 이 오퍼레이션은 변수를 변경합니다.
[`v.assign()`](../../api_docs/python/state.md#Variable.assign) 와
[`tf.train.Optimizer`](../../api_docs/python/train.md#Optimizer)등
에서의 매개변수 변경 작업은 *반드시*  배치된 변수와 같은 디바이스에서 실행해야 합니다.
호환되지 않는 디바이스 배치 지시는 오퍼레이션을 생성할때 무시될것 입니다.

디바이스 배치는 복제된 설정에서 실행할 때 특별히 중요합니다.
[`tf.train.replica_device_setter()`](../../api_docs/python/train.md#replica_device_setter)
에서 복제된 모델의 디바이스에서 간단하게 설정할 수 있는
디바이스 관련 함수에 대한 자세한 내용을 확인하세요.

## 초기화

변수 초기화는 다른 오퍼레이션이 모델을 실행하기 전에 명시적으로 실행해야만 합니다.
가장 쉬운 방법은 모든 변수를 초기화 하는 오퍼레이션을
모델을 사용하기 전에 실행하는것입니다.

다른 방법으로는 체크포인트 파일에서 변수 값을 복원할 수 있습니다.

`tf.initialize_all_variables()`를 오퍼레이션에 추가하여 변수 초기화를 진행할 수 있습니다.
완전히 모델을 구축한 후 세션안에서 이 오퍼레이션을 실행할 수 있습니다.  

```python
# 두 변수를 생성
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# 변수 초기화 오퍼레이션을 초기화
init_op = tf.initialize_all_variables()

# 나중에 모델을 실행할때
with tf.Session() as sess:
  # 초기화 오퍼레이션을 실행
  sess.run(init_op)
  ...
  # 모델 사용
  ...
```

### 다른 변수값을 참조하여 초기화 하기

다른 변수로 부터 값을 참조하여 변수를 초기화 해야할 때가 있습니다.  
`tf.initialize_all_variables()`에 의해 추가된 오퍼레이션으로
모든 변수를 초기화 할때 조심해야합니다.

다른 변수의 값으로 부터 새로운 변수를 초기화 하려 할때
다른 변수의 `initialized_value()` 속성을 사용할 수 있습니다.
초기화된 값을 직접적으로 새로운 변수의 초기값으로 사용할 수 있습니다.
또는 다른 텐서를 새로운 변수의 계산된 값으로 사용할 수 있습니다.

```python
# 랜덤 값으로 새로운 변수 초기화
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
# weights와 같은 값으로 다른 변수 초기화
w2 = tf.Variable(weights.initialized_value(), name="w2")
# weights의 2배 값으로 다른 변수 초기화
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")
```

### 커스텀 초기화

`tf.initialize_all_variables()`의 편의 기능은 모델에서 모든 변수를 초기화 하는 오퍼레이션을 추가합니다.  
또한 명시적으로 변수를 초기화 할 리스트를 전달할 수 있습니다.
[Variables Documentation](../../api_docs/python/state_ops.md)
에서 변수가 초기화 되었는지 확인하는 추가적인 옵션을 확인하세요.

## 저장과 복구

모델을 저장하고 복구하기위한 가장 쉬운 방법은 `tf.train.Saver` 오브젝트를 이용하는 것입니다.
생성자는 모든 그래프 또는 그래프 안에 지정된 변수 리스트에 `save` and `restore` 오퍼레이션을 추가합니다.
`Saver` 오브젝트는 이 오퍼레이션을 실행하거나 지정된 경로에 체크포인트 파일을 읽고 쓰는 메소드를 제공합니다.

### 체크 포인트 파일

변수는 바이너리 파일에 저장됩니다.  
바이너리 파일은 크게 변수 이름과 텐서 값의 매칭이 포함되어 있습니다.

`Saver` 오브젝트를 생성할때, 체크포인트 파일의 이름을 정할 수 있습니다.
기본적(default)으로 각 변수의 [`Variable.name`](../../api_docs/python/state_ops.md#Variable.name)
속성을 사용합니다.

체크포인트 안에 어떤 변수가 있는지 알기 위해선,
[`inspect_checkpoint`](https://www.tensorflow.org/code/tensorflow/python/tools/inspect_checkpoint.py)
라이브러리를 사용할 수 있고 특별한 경우 `print_tensors_in_checkpoint_file` 함수를 이용할수 있습니다.

### 변수 저장

`tf.train.Saver()` 로 `Saver` 를 생성하여 모든 변수를 관리할 수 있습니다.

```python
# 몇개의 변수를 생성
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 변수 초기화를 위한 오퍼레이션 추가
init_op = tf.initialize_all_variables()

# 모든 변수의 저장과 복구를 위한 오퍼레이션 추가
saver = tf.train.Saver()

# 모델 실행, 변수 초기화, 몇 가지의 작업 실행, 디스크에 변수 저장
with tf.Session() as sess:
  sess.run(init_op)
  # 모델을 사용하여 작업
  ...
  # 디스크에 변수를 저장
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)
```

### 변수 복구

똑같은 `Saver` 오브젝트를 변수를 복구하는데 사용할 수 있습니다.
변수를 파일에서 복구 할때 변수를 초기화 할 필요 없습니다.

```python
# 몇개의 변수를 생성
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 변수 초기화를 위한 오퍼레이션 추가
saver = tf.train.Saver()

# 모델 실행, 변수 초기화, 몇 가지의 작업 실행, 디스크에 변수 저장
with tf.Session() as sess:
  # 디스크에서 변수를 복구
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # 모델을 사용하여 작업
  ...
```

### 저장 및 복구할 변수들의 선택

만약 `tf.train.Saver()`에 아무런 인자도 전달하지 않는다면,
saver는 그래프 안의 모든 변수를 처리합니다.
그들은 각각 변수가 생성될때 부여된 이름으로 저장됩니다.

체크포인트 파일의 변수에 명시적으로 이름을 지정하는 것이 유용할 수 있습니다.
예를 들어, `"weights"` 라는 변수로 모델을 트레이닝 하고 이 값을 `"params"` 라는 새로운 변수로
복구할 수 있습니다.

또한, 모델에 사용된 일부의 변수를 저장하고 복원하는데 유용합니다.
예를 들어, 5단 레이어를 가지는 뉴럴 넷을 학습시키고 나중에 6단 레이어를 가지는 새로운 모델을
학습시키고 싶을때, 5단 레이어로 부터 훈련된 값을
새로운 모델(6단 레이어)의 첫 5단 레이어에 복원할 수 있습니다.

지정된 이름과 변수를 파이썬의 dictionary 타입으로
`tf.train.Saver()` 생성자에 전달하여 쉽게 저장할수 있습니다.
이때 dictionary의 키는 사용하는 이름이며, 값은 관리 해야될 변수 입니다.

주의:

*  모델 변수의 일부분을 저장하고 복원해야하는 경우 당신이 원하는만큼
   saver 오브젝트를 만들 수 있습니다. 같은 변수가 여러개의 saver 오브젝트의 리스트에 있을 수 있습니다. 이 값은 saver의 `restore()` 메소드를 실행하여야만 변경 가능합니다.

* 만약 모델 변수의 일부분을 세션의 시작에서 복원 한다면 다른 변수들을 초기화 하는
  오퍼레이션을 실행하여야 합니다. [`tf.initialize_variables()`](../../api_docs/python/state_ops.md#initialize_variables)
  에서 자세한 사항을 확인하세요

```python
# 몇개의 변수 생성
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# "my_v2"라는 이름을 이용하여 'v2'를 저장하고 복구 하는 오퍼레이션 추가
saver = tf.train.Saver({"my_v2": v2})
# 그 뒤 saver 오브젝트를 평상시 사용하던대로 사용
...
```
