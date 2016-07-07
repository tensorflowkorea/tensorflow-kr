## tf.contrib.learn 시작하기

텐서플로우의 고수준 머신러닝 API(tf.contrib.learn)는 다양한 머신러닝 모델을 쉽게 설정하고, 훈련하고, 평가할 수 있도록 해줍니다. 이 간단한 튜토리얼에서는  [신경망](https://en.wikipedia.org/wiki/Artificial_neural_network) 분류기를 만들고, [피셔의 Iris 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)으로 꽃받침과 꽃잎의 정보를 이용하여 꽃의 종류를 예측할 수 있도록 분류기를 훈련시킬 것입니다. 당신은 다음의 다섯 단계를 수행할 것입니다:

1. Iris 훈련/테스트 데이터를 담은 CSV 파일을 텐서플로우 Dataset으로 불러옵니다
2. [신경망 분류기](../../api_docs/python/contrib.learn.html#DNNClassifier)를 만듭니다
3. 훈련 데이터를 이용하여 모델을 피팅합니다
4. 모델의 정확도를 평가합니다
5. 새로운 표본을 분류합니다

## 시작하기

이 튜토리얼을 시작하기 전에  [당신의 머신에는 텐서플로우가 설치되어 있어야합니다](../../get_started/os_setup.html#download-and-setup).

다음은 우리의 신경망의 전체 코드입니다 :

```python
import tensorflow as tf
import numpy as np

# 데이터셋
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# 데이터셋을 불러옵니다.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target

# 10-20-10의 layer 구조를 갖는 3 layer DNN를 만듭니다
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

# 모델을 피팅합니다.
classifier.fit(x=x_train, y=y_train, steps=200)

# 정확도를 평가합니다.
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# 새로운 두 꽃의 표본을 분류합니다.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))
```

다음의 문단들은 이 코드를 자세하게 살펴볼 것입니다.

## Iris 데이터 CSV 파일을 텐서플로우로 불러오기

[Iris 데이터셋](https://en.wikipedia.org/wiki/Iris_flower_data_set)은 Iris 종 내에서 서로 연관된 *Iris setosa*, *Iris virginica*, and *Iris versicolor* 세 종이 각각 50개씩의 표본으로 구성되어 150개 행의 데이터로 되어있습니다. 각각의 행은 각 꽃 표본에 대한 다음의 정보를 담고 있습니다 : [꽃받침](https://en.wikipedia.org/wiki/Sepal) 길이, 꽃받침 너비, [꽃잎](https://en.wikipedia.org/wiki/Petal) 길이, 꽃잎 너비, 그리고 꽃의 종류. 꽃의 종류는 정수로 표현되어 있으며, 0은 *Iris setosa*, 1은 *Iris versicolor*, 그리고 2는 *Iris virginica*를 나타냅니다.

꽃받침 길이 | 꽃받침 너비 | 꽃받침 길이 | 꽃받침 너비 | 종
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

<!-- 유의사항 : 이 문단의 나머지에서는 CSV 파일이 튜토리얼 예제 파일과 같은 경로에 있다고 가정합니다 : 만약 그렇지 않다면, 링크와 코드를 갱신하십시오. -->
이 튜토리얼을 위해서는 Iris data는 임의적으로 섞인 후에, 두 개의 따로 떨어진 CSV 파일로 나누어져야 합니다. 이는 120개의 표본을 갖는 훈련 데이터(([iris_training.csv](http://download.tensorflow.org/data/iris_training.csv)))와 30개의 표본을 갖는 테스트 데이터(([iris_test.csv](http://download.tensorflow.org/data/iris_test.csv)))입니다.

To get started, first import TensorFlow and numpy:

```python
import tensorflow as tf
import numpy as np
```

Next, load the training and test sets into `Dataset`s using the [`load_csv()`]
(https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/datasets/base.py)  method in `learn.datasets.base`. The
`load_csv()` method has two required arguments:

*   `filename`, which takes the filepath to the CSV file, and 
*   `target_dtype`, which takes the [`numpy` datatype](http://docs.scipy.org/doc/numpy/user/basics.types.html) of the dataset's target value.

Here, the target (the value you're training the model to predict) is flower
species, which is an integer from 0&ndash;2, so the appropriate `numpy`
datatype is `np.int`:

```python
# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)
```

Next, assign variables to the feature data and target values: `x_train` for
training-set feature data, `x_test` for test-set feature data, `y_train` for
training-set target values, and `y_test` for test-set target values. `Dataset`s
in tf.contrib.learn are [named tuples](https://docs.python.org/2/library/collections.h
tml#collections.namedtuple), and you can access feature data and target values
via the `data` and `target` fields, respectively:

```python
x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
```

Later on, in "Fit the DNNClassifier to the Iris Training Data," you'll use
`x_train` and `y_train` to  train your model, and in "Evaluate Model
Accuracy", you'll use `x_test` and `y_test`. But first, you'll construct your
model in the next section.

## Construct a Deep Neural Network Classifier

tf.contrib.learn offers a variety of predefined models, called [`Estimator`s
](../../api_docs/python/contrib.learn.html#estimators),  which you can use "out
of the box" to run training and evaluation operations on your data.  Here,
you'll configure a Deep Neural Network Classifier model to fit the Iris data.
Using tf.contrib.learn, you can instantiate your
[`DNNClassifier`](../../api_docs/python/contrib.learn.html#DNNClassifier) with
just one line of code:

```python
# Build 3 layer DNN with 10, 20, 10 units respectively. 
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
```

The code above creates a `DNNClassifier` model with three [hidden layers](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw), 
containing 10, 20, and 10 neurons, respectively (`hidden_units=[10, 20, 10]`), and three target
classes (`n_classes=3`).


## Fit the DNNClassifier to the Iris Training Data

Now that you've configured your DNN `classifier` model, you can fit it to the Iris training data
using the [`fit`](../../api_docs/python/contrib.learn.html#BaseEstimator.fit) 
method. Pass as arguments your feature data (`x_train`), target values
(`y_train`), and the number of steps to train (here, 200):

```python
# Fit model
classifier.fit(x=x_train, y=y_train, steps=200)
```

<!-- Style the below (up to the next section) as an aside (note?) -->

<!-- Pretty sure the following is correct, but maybe a SWE could verify? -->
The state of the model is preserved in the `classifier`, which means you can train iteratively if
you like. For example, the above is equivalent to the following:

```python
classifier.fit(x=x_train, y=y_train, steps=100)
classifier.fit(x=x_train, y=y_train, steps=100)
```

<!-- TODO: When tutorial exists for monitoring, link to it here -->
However, if you're looking to track the model while it trains, you'll likely
want to instead use a TensorFlow [`monitor`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/monitors.py)
to perform logging operations.

## Evaluate Model Accuracy

You've fit your `DNNClassifier` model on the Iris training data; now, you can
check its accuracy on the Iris test data using the [`evaluate`
](../../api_docs/python/contrib.learn.html#BaseEstimator.evaluate) method.
Like `fit`, `evaluate` takes feature data and target values as
arguments, and returns a `dict` with the evaluation results. The following
code passes the Iris test data&mdash;`x_test` and `y_test`&mdash;to `evaluate`
and prints the `accuracy` from the results:

```python
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
```

Run the full script, and check the accuracy results. You should get:

```
Accuracy: 0.933333
```

Not bad for a relatively small data set!

## Classify New Samples

Use the estimator's `predict()` method to classify new samples. For example,
say you have these two new flower samples:

Sepal Length | Sepal Width | Petal Length | Petal Width
:----------- | :---------- | :----------- | :----------
6.4          | 3.2         | 4.5          | 1.5
5.8          | 3.1         | 5.0          | 1.7        

You can predict their species with the following code:

```python
# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))
```

The `predict()` method returns an array of predictions, one for each sample:

```python
Prediction: [1 2]
```

The model thus predicts that the first sample is *Iris versicolor*, and the
second sample is *Iris virginica*.

## Additional Resources

* For further reference materials on tf.contrib.learn, see the official
[API docs](../../api_docs/python/contrib.learn.md).

<!-- David, will the below be live when this tutorial is released? -->
* To learn more about using tf.contrib.learn to create linear models, see 
[Large-scale Linear Models with TensorFlow](../linear/).

* To experiment with neural network modeling and visualization in the browser,
check out [Deep Playground](http://playground.tensorflow.org/).

* For more advanced tutorials on neural networks, see [Convolutional Neural
Networks](../deep_cnn/) and [Recurrent Neural Networks](../recurrent/).
