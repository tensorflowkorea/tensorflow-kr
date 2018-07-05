<!-- 160708 @mikibear : 최종 배포시에 md로 링크된 것을 html로 변경할 것입니다. 혹시 발견하신다면, 변경해주시면 감사하겠습니다. -->

# tf.contrib.learn 시작하기
(v1.0)

텐서플로우의 고수준 머신러닝 API(tf.contrib.learn)는 다양한 머신러닝 모델을 쉽게 설정하고, 훈련하고, 평가할 수 있도록 해줍니다. 이 튜토리얼에서는 tf.contrib.learn 을 사용하여 [신경망](https://en.wikipedia.org/wiki/Artificial_neural_network) 분류기를 만들고, [Iris 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)에 있는 꽃받침과 꽃잎의 정보를 이용하여 꽃의 종류를 예측할 수 있도록 분류기를 훈련시킬 것입니다. 코드는 다음의 다섯 단계로 수행됩니다:

1. Iris 훈련/테스트 데이터를 담은 CSV 파일을 텐서플로우 `Dataset`으로 불러옵니다.
2. [신경망 분류기](../../api_docs/python/contrib.learn.md#DNNClassifier)를 만듭니다.
3. 훈련 데이터를 이용하여 모델을 훈련 시킵니다.
4. 모델의 정확도를 평가합니다.
5. 새로운 표본을 분류합니다.

참고: 이 튜토리얼을 시작하기 전에 [텐서플로우를 설치](../../get_started/os_setup.md#download-and-setup)해야 합니다.

## 시작하기

다음은 이 신경망 분류기의 전체 코드입니다:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# 데이터셋
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 데이터셋을 불러옵니다.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

# 모든 특성이 실수값을 가지고 있다고 지정합니다
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 10, 20, 10개의 유닛을 가진 3층 DNN를 만듭니다
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# 모델을 학습시킵니다.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

# 정확도를 평가합니다.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('정확도: {0:f}'.format(accuracy_score))

# 새로운 두 개의 꽃 표본을 분류합니다.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print ('예측: {}'.format(str(y)))
```

다음에서 이 코드를 자세하게 살펴 보겠습니다.

## Iris CSV 데이터를 텐서플로우로 불러오기

[Iris 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)은 Iris 품종인 *Iris setosa*, *Iris virginica*, *Iris versicolor*가 각각 50개씩의 표본으로 구성된 150개의 행을 가진 데이터입니다.

![Petal geometry compared for three iris species: Iris setosa, Iris virginica,
and Iris versicolor](../../images/iris_three_species.jpg) **왼쪽에서 오른쪽으로,
[*Iris setosa*](https://commons.wikimedia.org/w/index.php?curid=170298) ([Radomil](https://commons.wikimedia.org/wiki/User:Radomil), CC BY-SA 3.0),
[*Iris versicolor*](https://commons.wikimedia.org/w/index.php?curid=248095) ([Dlanglois](https://commons.wikimedia.org/wiki/User:Dlanglois), CC BY-SA 3.0),
and [*Iris virginica*](https://www.flickr.com/photos/33397993@N05/3352169862)
([Frank Mayfield](https://www.flickr.com/photos/33397993@N05), CC BY-SA
2.0).**

각각의 행은 각 꽃의 표본에 대한 다음의 정보를 담고 있습니다 : [꽃받침](https://en.wikipedia.org/wiki/Sepal) 길이, 꽃받침 너비, [꽃잎](https://en.wikipedia.org/wiki/Petal) 길이, 꽃잎 너비, 그리고 꽃의 종류. 꽃의 종류는 정수로 표현되어 있으며, 0은 *Iris setosa*, 1은 *Iris versicolor*, 그리고 2는 *Iris virginica*를 나타냅니다.

꽃받침 길이 | 꽃받침 너비 | 꽃잎 길이 | 꽃잎 너비 | 종
:----------- | :---------- | :----------- | :---------- | :------
5.1          | 3.5         | 1.4          | 0.2         | 0
4.9          | 3.0         | 1.4          | 0.2         | 0
4.7          | 3.2         | 1.3          | 0.2         | 0
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
7.0          | 3.2         | 4.7          | 1.4         | 1
6.4          | 3.2         | 4.5          | 1.5         | 1
6.9          | 3.1         | 4.9          | 1.5         | 1
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
6.5          | 3.0         | 5.2          | 2.0         | 2
6.2          | 3.4         | 5.4          | 2.3         | 2
5.9          | 3.0         | 5.1          | 1.8         | 2

이 튜토리얼을 위해서 Iris data를 랜덤하게 섞은 후에, 두 개의 CSV 파일로 나누어 놓았습니다:

*   120개의 표본을 갖는 훈련 데이터([iris_training.csv](http://download.tensorflow.org/data/iris_training.csv))
*   30개의 표본을 갖는 테스트 데이터([iris_test.csv](http://download.tensorflow.org/data/iris_test.csv))

이 두 CSV 파일을 파이썬 코드와 같은 디렉토리에 놓습니다.

시작하려면, 먼저 텐서플로우와 numpy를 임포트합니다:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
```

그 다음, `learn.datasets.base`에 있는 [`load_csv_with_header()`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/datasets/base.py) 함수를 이용하여 훈련 셋과 테스트 셋을 `Dataset`으로 불러옵니다. `load_csv_with_header()` 함수는 세 개의 인자를 요구합니다:

*   `filename`, CSV 파일의 경로
*   `target_dtype`, 데이터셋에 있는 타깃 값의 [`numpy` 데이터형](http://docs.scipy.org/doc/numpy/user/basics.types.html)
*   `features_dtype`, 데이터셋에 있는 특성 값의 [`numpy` 데이터형](http://docs.scipy.org/doc/numpy/user/basics.types.html)

여기에서 타깃 값(모델을 훈련시켜 예측하려고 하는 값)은 0&ndash;2의 정수로 구성된 꽃의 종입니다. 따라서, 적절한 `numpy` 데이터형은 `np.int`입니다:

```python
# 데이터셋
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 데이터셋을 불러옵니다.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
```

tf.contrib.learn의 `Dataset`은 파이썬의 [네임드 튜플](https://docs.python.org/2/library/collections.html#collections.namedtuple)이며, `data`와 `target` 필드를*(역주 : namedtuple의 field_name을 말합니다)* 이용해 특성 데이터와 타깃 값에 접근할 수 있습니다. `training_set.data`와 `training_set.target`은 각각 훈련 셋의 특성 데이터와 타깃 값을 가지고 있고, `test_set.data`와 `test_set.target`은 테스트 셋의 특성 데이터와 타깃 값을 가지고 있습니다.

나중에 ["Iris 훈련 데이터로 DNNClassifier 훈련시키기"](#fit-dnnclassifier)에서, `training_set.data`와 `training_set.target`을 이용하여 모델을 훈련시키고, ["모델 정확도 평가하기"](#evaluate-accuracy)에서는 `test_set.data`와 `test_set.target`를 이용할 것입니다. 그전에 먼저, 다음 섹션에서 모델을 구성해 보겠습니다.

## 딥 신경망 분류기 만들기

tf.contrib.learn은 데이터를 가지고 훈련과 평가를 위해 곧장 사용할 수 있는 [`Estimator`](../../api_docs/python/contrib.learn.md#estimators)라 불리는 여러 가지의 미리 정의된 모델을 제공합니다. 여기에서는 Iris data를 학습시키기 위해 딥 신경망 모델을 구성하겠습니다. tf.contrib.learn을 이용하면, [`DNNClassifier`](../../api_docs/python/contrib.learn.md#DNNClassifier)의 객체를 몇 줄의 코드로 간단히 만들 수 있습니다:

```python
# 모든 특성이 실수값을 가지고 있다고 지정합니다
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 10, 20, 10개의 유닛을 가진 3층 DNN를 만듭니다
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")
```

위 코드는 먼저 데이터 셋에 있는 특성의 데이터 타입을 지정하기 위해 모델의 특성 열(feature_columns)을 정의합니다. 모든 특성이 연속적인 값이므로 `tf.contrib.layers.real_valued_column`가 특성 열을 구성하는 데 사용하기 적합한 함수입니다. 이 데이터 셋에는 네 개의 특성(꽃받침 너비, 꽃받침 길이 꽃잎 너비, 꽃잎 길이)이 있으므로 모두 포함시키기 위해 `dimensions`을 `4`로 지정합니다.

그런 다음 아래 인자를 사용하여 `DNNClassifier` 모델을 만듭니다:

*   `feature_columns=feature_columns`. 위에서 만든 특성 열을 지정합니다.
*   `hidden_units=[10, 20, 10]`. 각각 10, 20, 10 개의 뉴런을 가진 세 개의 [히든 레이어](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw).
*   `n_classes=3`. 세 가지 붓꽃의 품종을 나타내는 타깃 클래스.
*   `model_dir=/tmp/iris_model`. 모델이 학습하는 동안 체크 포인트를 저장할 디렉토리. 텐서플로우의 로깅과 모니터링에 대해서는 [Logging and Monitoring Basics with tf.contrib.learn](../monitors/index.md)을 참고하세요.

## Iris 훈련 데이터로 DNNClassifier 훈련 시키기 {#fit-dnnclassifier}

이제 DNN `classifier` 모델을 설정했으니, [`fit`](../../api_docs/python/contrib.learn.md#BaseEstimator.fit) 메소드를 이용하여 Iris 훈련 데이터에 학습시킬 수 있습니다. 특성 데이터(`training_set.data`)와 타깃 값(`training_set.target`), 그리고 훈련할 반복 횟수(여기서는 2000)를 인자로 넘겨줍니다:

```python
# 모델 학습
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)
```

`classifier`는 모델의 상태를 저장하고 있습니다. 따라서 원한다면 모델을 반복하여 학습시킬 수 있습니다. 예를 들어서, 위의 한 줄은 다음의 두 줄과 완벽하게 같습니다:

```python
classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
```

하지만 만약 학습되는 동안에 모델을 추적하고 싶다면, 대신 텐서플로우의 [`monitor`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/monitors.py)를 사용하여 로그를 남기는 게 낫습니다. 이에 대한 내용은 [&ldquo;Logging and Monitoring
Basics with tf.contrib.learn&rdquo;](../monitors/index.md)를 참고하세요.

## 모델 정확도 평가하기 {#evaluate-accuracy}

Iris 테스트 데이터를 사용해 `DNNClassifier` 모델을 학습시켰습니다. 이제, [`evaluate`](../../api_docs/python/contrib.learn.md#BaseEstimator.evaluate) 메소드를 이용하여 Iris 테스트 데이터로 모델의 정확도를 확인해 보겠습니다. `evaluate`는 `fit`과 같이 특성 데이터와 타깃 값을 인자로 받고, 평가 결과를 하나의 딕셔너리로 반환합니다. 다음의 코드는 Iris 테스트 데이터&mdash;`test_set.data`와 `test_set.target`&mdash;를 전달하고, 결과 값에서 `accuracy`를 출력합니다:

```python
accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
```

전체 스크립트를 실행하고 정확도 결과를 확인합니다:

```
Accuracy: 0.966667
```

결과 값이 조금 다를 수 있지만 90%보다는 높게 나올 것입니다. 비교적 적은 데이터셋 치고는 나쁘지 않습니다!

## 새로운 표본 분류하기

새로운 표본을 분류하기 위해 모델의 `predict()` 메소드를 이용합니다. 예를 들어, 다음의 새로운 꽃의 표본이 두 개가 있습니다:

꽃받침 길이 | 꽃받침 너비 | 꽃잎 길이 | 꽃잎 너비
:----------- | :---------- | :----------- | :----------
6.4          | 3.2         | 4.5          | 1.5
5.8          | 3.1         | 5.0          | 1.7        

다음 코드에서 이 데이터의 종을 예측할 수 있습니다:

```python
# 새로운 두 꽃의 표본을 분류합니다.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
```

`predict()` 메소드는 각 표본의 예측 결과를 표본 마다 하나씩 할당하여 배열로 반환합니다:

```python
Prediction: [1 2]
```

이 모델은 첫 번째 표본을 *Iris versicolor*, 두 번째 표본을 *Iris virginica*로 예측하였습니다.

## 추가적인 자료

*   tf.contrib.learn에 대해 추가적인 참고 자료를 원한다면, 공식적인 [API docs](../../api_docs/python/contrib.learn.md)를 살펴보세요.

*   tf.contrib.learn을 이용해 선형 모델을 생성하는 방법을 배우려면 [Large-scale Linear Models with TensorFlow](../linear/overview.md)를 살펴보세요.

*   tf.contrib.learn API를 이용하여 자신만의 Estimator 클래스를 만들고 싶다면 [Building
    Machine Learning Estimator in
    TensorFlow](http://terrytangyuan.github.io/2016/07/08/understand-and-build-tensorflow-estimator/)를 살펴보세요.

*   브라우저에서의 신경망 모델링과 시각화를 체험해 보려면 [Deep Playground](http://playground.tensorflow.org/)를 살펴보세요.

*   신경망에 대한 좀 더 심화된 튜토리얼을 원한다면 [Convolutional Neural Networks](../deep_cnn/)와 [Recurrent Neural Networks](../recurrent/)를 살펴보세요.
