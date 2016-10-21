# TensorFlow Wide & Deep Learning Tutorial
# TensorFlow Wide & Deep Learning 튜토리얼

In the previous [TensorFlow Linear Model Tutorial](../wide/),
이전 튜토리얼[선형모델(Linear Model) 튜토리얼] 에서,
we trained a logistic regression model to predict the probability that the
individual has an annual income of over 50,000 dollars using the [Census Income
Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income).
연소득이 $50,000 이상인 사람을 예측하기 위해 로지스틱 회귀분석 모델(logistic regression model) 을
[Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income) 를 이용하여 학습시켰다.
TensorFlow is great for training deep neural networks too,
and you might be thinking which one
you should choose—Well, why not both? Would it be possible to combine the
strengths of both in one model?
TensorFlow 는 deep neural networks 학습에 탁월하고,
어느 것을 잘 선택해야 하는가에 대한 것을 생각할 것이다.
이들을 동시에 사용하는 것은 안될까? 하나의 모델에 둘의 strength를 조합하는 것은 가능할까? 

In this tutorial, we'll introduce how to use the TF.Learn API to jointly train a
wide linear model and a deep feed-forward neural network. This approach combines
the strengths of memorization and generalization. It's useful for generic
large-scale regression and classification problems with sparse input features
(e.g., categorical features with a large number of possible feature values). If
you're interested in learning more about how Wide & Deep Learning works, please
check out our [research paper](http://arxiv.org/abs/1606.07792).
이 튜토리얼에서, TF 를 사용하는 법에 대해 소개할 것이다.
wide 선형 모델과 deep feed-forward 신경망을 함께 학습하기 위한 API 를 배우자.
이 접근법은 기억과 일반화의 강점을 조합한 것이다. 이 방법은 sparse 입력 특징
(즉, 많은 수의 가능한 특징값을 가진 분류적 특징) 의 일반적인 큰 규모의 회귀법과
분류법 문제에 유용하다.
Wide & Deep 학습의 학습 동작에 대해 관심이 있다면, 우리 논문[research paper]
(http://arxiv.org/abs/1606.07792) 를 참고하기 바란다.

![Wide & Deep Spectrum of Models]
(../../images/wide_n_deep.svg "Wide & Deep")

The figure above shows a comparison of a wide model (logistic regression with
sparse features and transformations), a deep model (feed-forward neural network
with an embedding layer and several hidden layers), and a Wide & Deep model
(joint training of both). At a high level, there are only 3 steps to configure a
wide, deep, or Wide & Deep model using the TF.Learn API:
위 그림은 wide 모델(sparse 특징과 변환의 로지스틱 회귀분석), deep 모델(embedding layer와
여러 hidden layer들의 feed-forward 신경망) 을 비교하여 보여준다.
가장 높은 층에서는 TF를 이용한 wide, deep, wide & deep 모델의 3 단계만 존재한다.API 를 배우자:

1.  Select features for the wide part: Choose the sparse base columns and
    crossed columns you want to use.
1.  Select features for the deep part: Choose the continuous columns, the
    embedding dimension for each categorical column, and the hidden layer sizes.
1.  Put them all together in a Wide & Deep model
    (`DNNLinearCombinedClassifier`).
1. wide 부분에 대한 설정 선택 : 원하는 sparse base 열과 crossed 열들을 선택한다
1. deep 부분에 대한 설정 선택 : 연속된 열, 각 분류 열의 embedding dimension,
   그리고 hidden layer 크기를 선택한다
1. 이들을 Wide & Deep 모델에 적용한다(`DNNLinearCombinedClassifier`)

And that's it! Let's go through a simple example.
다 됐다! 간단한 예제를 통해 알아보자.

## Setup

To try the code for this tutorial:
이 튜토리얼을 위한 코드를 실행하기 위해서,

1.  [Install TensorFlow](../../get_started/os_setup.md) if you haven't
already.
1.  아직 설치되지 않았다면, TensorFlow를 설치한다 [Install TensorFlow](../../get_started/os_setup.md)

2.  Download [the tutorial code](
https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py).
2.  튜토리얼 코드를 다운로드 한다 [the tutorial code](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)

3.  Install the pandas data analysis library. tf.learn doesn't require pandas, but it does support it, and this tutorial uses pandas. To install pandas	:
3.  pandas 데이터 분석 라이브러리를 설치. tf.learn은 pandas 을 필요로 하지 않지만 지원한다. 그리고 이 튜토리얼은 pandas를 사용한다. pandad 를 설치하기 위해서 :

		1. Get `pip`:
		1. 'pip' 설치 :

       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```


       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```


       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```

    2. Use `pip` to install pandas:
		2. 'pip' 를 이용하여 pandas 설치 :

       ```shell
       $ sudo pip install pandas
       ```

    If you have trouble installing pandas, consult the [instructions]
(http://pandas.pydata.org/pandas-docs/stable/install.html) on the pandas site.
		pandas 설치에 문제가 있다면, pandas 사이트의 [instructions](http://pandas.pydata.org/pandas-docs/stable/install.html) 를 참고하라.

4. Execute the tutorial code with the following command to train the linear
model described in this tutorial:
4. 이 튜토리얼에 소개된 선형 모델을 학습시키기 위해 이어지는 명령어로 튜토리얼 코드를 실행하자

   ```shell
   $ python wide_n_deep_tutorial.py --model_type=wide_n_deep
   ```

Read on to find out how this code builds its linear model.
이 코드가 어떻게 이 선형 모델을 만들어가는 알아보기 위해 계속 읽어보자.


## Define Base Feature Columns
## Base Feature Columns 정의

First, let's define the base categorical and continuous feature columns that
we'll use. These base columns will be the building blocks used by both the wide
part and the deep part of the model.
우선, 사용할 base categorical feature columns 과 continuous feature columns 을 정의하자.
이들 base column 들은 모델의 wide 부분과 deep 부분에 사용될 building block 들로 된다.

```python
import tensorflow as tf

# Categorical base columns.
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["female", "male"])
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Continuous base columns.
age = tf.contrib.layers.real_valued_column("age")
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
```

## The Wide Model: Linear Model with Crossed Feature Columns
## The Wide Model: Linear Model with Crossed Feature Columns

The wide model is a linear model with a wide set of sparse and crossed feature
columns:
wide 모델은 다양한 sparse 와 crosseed feature 열들의 집합의 선형 모델이다:

```python
wide_columns = [
  gender, native_country, education, occupation, workclass, marital_status, relationship, age_buckets,
  tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4)),
  tf.contrib.layers.crossed_column([age_buckets, race, occupation], hash_bucket_size=int(1e6))]
```

Wide models with crossed feature columns can memorize sparse interactions
between features effectively. That being said, one limitation of crossed feature
columns is that they do not generalize to feature combinations that have not
appeared in the training data. Let's add a deep model with embeddings to fix
that.
crossed feature columns 의 Wide model 은 효율적으로 feature들 간의 sparse 작용들을 기억할 수 있다.
그럼에도, 한가지 crossed feature columns 의 한계는 이들은 학습 데이터에서 나타나지 않는
feature 조합들을 만들어내지 못한다는 것이다.

## The Deep Model: Neural Network with Embeddings
## The Deep Model: Neural Network with Embeddings

The deep model is a feed-forward neural network, as shown in the previous
figure. Each of the sparse, high-dimensional categorical features are first
converted into a low-dimensional and dense real-valued vector, often referred to
as an embedding vector. These low-dimensional dense embedding vectors are
concatenated with the continuous features, and then fed into the hidden layers
of a neural network in the forward pass. The embedding values are initialized
randomly, and are trained along with all other model parameters to minimize the
training loss. If you're interested in learning more about embeddings, check out
the TensorFlow tutorial on [Vector Representations of Words]
(https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html), or
[Word Embedding](https://en.wikipedia.org/wiki/Word_embedding) on Wikipedia.

We'll configure the embeddings for the categorical columns using
`embedding_column`, and concatenate them with the continuous columns:

```python
deep_columns = [
  tf.contrib.layers.embedding_column(workclass, dimension=8),
  tf.contrib.layers.embedding_column(education, dimension=8),
  tf.contrib.layers.embedding_column(marital_status, dimension=8),
  tf.contrib.layers.embedding_column(gender, dimension=8),
  tf.contrib.layers.embedding_column(relationship, dimension=8),
  tf.contrib.layers.embedding_column(race, dimension=8),
  tf.contrib.layers.embedding_column(native_country, dimension=8),
  tf.contrib.layers.embedding_column(occupation, dimension=8),
  age, education_num, capital_gain, capital_loss, hours_per_week]
```

The higher the `dimension` of the embedding is, the more degrees of freedom the
model will have to learn the representations of the features. For simplicity, we
set the dimension to 8 for all feature columns here. Empirically, a more
informed decision for the number of dimensions is to start with a value on the
order of $$k\log_2(n)$$ or $$k\sqrt[4]n$$, where $$n$$ is the number of unique
features in a feature column and $$k$$ is a small constant (usually smaller than
10).

Through dense embeddings, deep models can generalize better and make predictions
on feature pairs that were previously unseen in the training data. However, it
is difficult to learn effective low-dimensional representations for feature
columns when the underlying interaction matrix between two feature columns is
sparse and high-rank. In such cases, the interaction between most feature pairs
should be zero except a few, but dense embeddings will lead to nonzero
predictions for all feature pairs, and thus can over-generalize. On the other
hand, linear models with crossed features can memorize these “exception rules”
effectively with fewer model parameters.

Now, let's see how to jointly train wide and deep models and allow them to
complement each other’s strengths and weaknesses.

## Combining Wide and Deep Models into One

The wide models and deep models are combined by summing up their final output
log odds as the prediction, then feeding the prediction to a logistic loss
function. All the graph definition and variable allocations have already been
handled for you under the hood, so you simply need to create a
`DNNLinearCombinedClassifier`:

```python
import tempfile
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```

## Training and Evaluating The Model

Before we train the model, let's read in the Census dataset as we did in the
[TensorFlow Linear Model tutorial](../wide/). The code for
input data processing is provided here again for your convenience:

```python
import pandas as pd
import urllib

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

# Download the training and test data to temporary files.
# Alternatively, you can download them yourself and change train_file and
# test_file to your own paths.
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

# Read the training and test data sets into Pandas dataframe.
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
df_train[LABEL_COLUMN] = (df_train['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)
```

After reading in the data, you can train and evaluate the model:
