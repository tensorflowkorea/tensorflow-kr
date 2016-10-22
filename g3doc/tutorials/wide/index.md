# 텐서플로우 선형 모델 튜토리얼

이번 강의에서 우리는 이진 분류 문제를 사람에 나이, 성별, 교육, 그리고 직업(특성들)에 관한 인구조사 데이터를 가지고 한 사람의 연봉이 50,000불이 넘는지를 TensorFlow에 TF.Learn API를 사용해서 풀어 볼 것이다(목표 레이블). 우리는 **로지스틱 회귀** 모델을 주어진 개인들에 정보를 가지고 교육 할 것이고 모델은 개인의 연봉이 50,000달러 이상일 가능성으로 해석 될 수 있는 0과1 사이의 숫자를 출력한다.

## 설치

이번 튜토리얼 코드를 실행해보기 위해서:

1.  텐서플로우를 설치하지 않았다면 [텐서플로 설치](../../get_started/os_setup.md) 

2.  [튜토리얼 코드](
    https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py) 다운로드.

3.  pandas 데이터 분석 라이브러리 설치.

 tf.learn에 pandas가 필수적인 요소는 아니지만 tf.learn이 판다를 지원하고 또한 이번 튜토리얼에서 판다를 하기 때문. pandas를 설치하기 위해서:

    1.  `pip` 설치:

       ```shell
       # Ubuntu/Linux 64-bit
       $ sudo apt-get install python-pip python-dev

       # Mac OS X
       $ sudo easy_install pip
       $ sudo easy_install --upgrade six
      ```

    2. `pip`로 pandas 설치하기:

       ```shell
       $ sudo pip install pandas
       ```

    만약에 판다 설치에 어려움을 느낀다면, 판다 사이트에서 [설명]
(http://pandas.pydata.org/pandas-docs/stable/install.html) 을 참고하시오.

4. 이 튜토리얼에 설명된 선형모델을 훈련하기 위해 튜토리얼 코드를 아래의 명령어로 실행하시오:

   ```shell
   $ python wide_n_deep_tutorial.py --model_type=wide
   ```
코드가 어떻게 선형모델을 구축하는지 계속 읽어 보자.

## 인구조사 데이터 읽어보기

우리가 사용할 데이터 세트는 [소득 인구조사 데이터세트]
(https://archive.ics.uci.edu/ml/datasets/Census+Income). [훈련 데이터]
(https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) 그리고 
[테스트 데이터](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test) 를 수동 또는 코드를 이용해서 내려받을 수 있습니다.

```python
import tempfile
import urllib
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)
```
CSV 파일들에 다운로드가 완료됐다면, [Pandas] (http://pandas.pydata.org/) 데이터프레임에 입력시켜 보자.

```python
import pandas as pd
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
```

이번 과제가 이진 분류 문제이기 때문에 수입이 50,000달러가 넘는다면 1을 그렇지 않다면 0에 값을 가지는 열의 이름이 "label"인 표를 만들 것이다.

```python
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
```

다음으로, 데이터프레임에서 어떤 열들이 목표 label을 예측하는 데 사용 될 수 있는지 살펴보자. 열들은 categorical 또는 continuous 두 타입으로 구분 될 수 있다.



*   만약에 값이 오직 유한집합 범주 안에 있을 때 **categorical**열이라 불린다. 
     예를 들어 사람에 국적(미국, 인도, 일본 등)이나 교육 수준(고등학교, 대학 등)이 categorical 열들이다.
        
  
    
*  만약에 값이 어떤 수치로 나올 수 있다면 **continuous**열 이라 불린다. 예를 들어, 한 사람에 소득이 continuous열이다.


```python
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
```

수입 인구조사 데이터 세트에 나오는 열 리스트:

|열 이름    | 타입        | 설명                       | {.sortable}
| -------------- | ----------- | --------------------------------- |
| age               | Continuous    | 나이                                   |
| workclass      | Categorical | 고용주 타입                      |
:                :             : (정부, 군대, 기업 등)             :
| fnlwgt         | Continuous  | The number of people the census   |
:                :             : takers believe that observation   :
:                :             : represents (sample weight). This  :
:                :             : variable will not be used.        :
| education      | Categorical | 최고 학력                         |
| education_num  | Continuous  | 숫자 형식의 최고 학력             |
| marital_status | Categorical | 혼인 여부                         |
| occupation     | Categorical | 직업                              |
| relationship   | Categorical | Wife, Own-child, Husband,         |
:                :             : Not-in-family, Other-relative,    :
:                :             : Unmarried.                        :
| race           | Categorical | 인종 ( 백인, Asian-Pac-Islander, |
:                :             : Amer-Indian-Eskimo, Other, 흑인. :
| gender         | Categorical | 성별 (남, 여)                     |
| capital_gain   | Continuous  | 기록된 양도 소득.                 |
| capital_loss   | Continuous  | 기록된 자본 손실.                 |
| hours_per_week | Continuous  | 주당 근무시간.                    |
| native_country | Categorical | 출생지                            |
| income         | Categorical | ">50K" 또는 "<=50K", 개인의 일년  |
:                :             : 수입이 5만불 이상인지 아닌지 뜻함 : 


##데이터를 텐서들로 바꾸기 

TF.Learn 모델을 구축 할 때, 입력 데이터는 Input Builder 함수에 의해서 명시된다.
이 builder 함수는 TF.Learn에 `fit` 이나 `evaluate`와 같은 메소드들에게 넘겨 질때 까지 호출되지 않는다.
이 함수의 목적은 입력 데이터를 [Tensors](https://www.tensorflow.org/versions/r0.9/api_docs/python/framework.html#Tensor) 나
[SparseTensors](https://www.tensorflow.org/versions/r0.9/api_docs/python/sparse_ops.html#SparseTensor) 형태로 구성하기 위함에 있다.
더 구체적으로, Input Builder 함수는 다음과 같은 한 쌍을 반환한다:

1.  `feature_cols`: A dict from feature column names to `Tensors` or
    `SparseTensors`.
2.  `label`: A `Tensor` containing the label column.

`feature_cols`에 키들은 다음 부분에서 열을 구성하는 데 사용 될 것이다.
 
우리는 `fit` 과 `evaluate` 메소드들을 서로 다른 데이터로 호출 하고 싶으므로,
서로 같지만 단지 다른 데이터를 `input_fn`에 전달하는 input builder 함수인 `train_input_fn` 그리고 `test_input_fn`를 정의 했습니다.

여기서 눈여겨볼 것은 `input_fn`가 그래프 실행 중이 아니라 텐서플로 그래프를 생성하는 도중에 호출된다는 것입니다.
Input Builder가 반환하는 것은 입력 데이터를 대표하는 텐서플로 연산 기본단위인 `Tensor`(또는 `SparseTensor`)입니다.

우리 모델은 입력 데이터의 정숫값을 대표하는 *constant* 텐서로 나타낸다, 이 경우에는 `df_train` 나 `df_test`에 열에 값을 대표한다.
이 방법이 텐서플로에 데이터를 전달하는 가장 간단한 방법이다.
다른 심화한 방법으로는 파일이나 다른 데이터 소스를 대표하는 Input Reader를 만들어서 파일을 텐서플로가 그래프를 실행하는 동안에 읽어 나가는 것이다.

훈련 또는 테스트 데이터 프레임에 있는 각각의 continuous 열들은 밀집 데이터를 대표하기 좋은 `Tensor`로 변환될 것입니다.

우리는 categorical 데이터를 반드시 `SparseTensor`로 나타내야 합니다.
Categorical 데이터 포맷은 희소 데이터를 대표하기 좋습니다.

```python
import tensorflow as tf

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

## Selecting and Engineering Features for the Model

모델의 선택과 공학 특징  


Selecting and crafting the right set of feature columns is key to learning an
effective model.


올바른 특성 열 세트를 선택하고 만드는 것이 효과적인 모델 학습에 핵심 입니다. 
 
 
 A **feature column** can be either one of the raw columns in
the original dataframe (let's call them **base feature columns**), or any new
columns created based on some transformations defined over one or multiple base
columns (let's call them **derived feature columns**). 



Basically, "feature
column" is an abstract concept of any raw or derived variable that can be used
to predict the target label.


근본적으로, "feature column"은 목표 레이블을 예상하는데 사용 가능한 추상적인 개념의 비가공 또는 파생 변수이다.  

### Base Categorical Feature Columns

To define a feature column for a categorical feature, we can create a
`SparseColumn` using the TF.Learn API.

categorical 특성을 위한 특성열을 정의 하기 위해서 우리는 TF.Learn API로 `SparseColumn` 생성할 수 있습니다.  
 
 If you know the set of all possible
feature values of a column and there are only a few of them, you can use
`sparse_column_with_keys`.
 만약에 모든 열에 특성 값 세트를 알고 있고 또한 몇게 안된다면, `sparse_column_with_keys`를 사용 할 수 있습니다.
 
 
 
 Each key in the list will get assigned an
auto-incremental ID starting from 0.
 
리스트에 안에 각각의 키들은 0부터 시작해서 자동으로 증가하는 아이디가 할당된다.
 
 
 For example, for the `gender` column we can
assign the feature string "female" to an integer ID of 0 and "male" to 1 by
doing:

예를 들어 우리는 아래와 같이 성별 열에 특성 문자열 "female"에게 숫자 아이디 1을 그리고 "male"에게는 1을 할당 할수 있다

```python
gender = tf.contrib.layers.sparse_column_with_keys(
  column_name="gender", keys=["female", "male"])
```

What if we don't know the set of possible values in advance?

만약에 가능한 값의 세트를 미리 알 수 없다면 어떻게 해야하나 ?
 
 Not a problem.
 문제 없다.
우리는 `sparse_column_with_hash_bucket`를 대신 사용 할 수 있다. 
 We
can use `sparse_column_with_hash_bucket` instead:

```python
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
```

What will happen is that each possible value in the feature column `education`
will be hashed to an integer ID as we encounter them in training. 
어
  `education`에 각 특성열에 가능한 값들은 훈련중에 정수 아이디로 헤시 뒤어질 것이다 


See an example
illustration below:
아래의 실례를 봐라 

ID  | Feature
--- | -------------
... |
9   | `"Bachelors"`
... |
103 | `"Doctorate"`
... |
375 | `"Masters"`
... |

No matter which way we choose to define a `SparseColumn`, each feature string
will be mapped into an integer ID by looking up a fixed mapping or by hashing.
Note that hashing collisions are possible, but may not significantly impact the
model quality. Under the hood, the `LinearModel` class is responsible for
managing the mapping and creating `tf.Variable` to store the model parameters
(also known as model weights) for each feature ID. The model parameters will be
learned through the model training process we'll go through later.

We'll do the similar trick to define the other categorical features:
우리는 다른 categorical 특성들을 정의하기 위해 비슷한 기술을 사용 할 것이다.

```python
race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=[
  "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
```

### Base Continuous Feature Columns

Similarly, we can define a `RealValuedColumn` for each continuous feature column
that we want to use in the model:

```python
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
```

### Making Continuous Features Categorical through Bucketization

Sometimes the relationship between a continuous feature and the label is not
linear. As an hypothetical example, a person's income may grow with age in the
early stage of one's career, then the growth may slow at some point, and finally
the income decreases after retirement. In this scenario, using the raw `age` as
a real-valued feature column might not be a good choice because the model can
only learn one of the three cases:

1.  Income always increases at some rate as age grows (positive correlation),
1.  Income always decreases at some rate as age grows (negative correlation), or
1.  Income stays the same no matter at what age (no correlation)

If we want to learn the fine-grained correlation between income and each age
group seperately, we can leverage **bucketization**. Bucketization is a process
of dividing the entire range of a continuous feature into a set of consecutive
bins/buckets, and then converting the original numerical feature into a bucket
ID (as a categorical feature) depending on which bucket that value falls into.
So, we can define a `bucketized_column` over `age` as:

```python
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

where the `boundaries` is a list of bucket boundaries. In this case, there are
10 boundaries, resulting in 11 age group buckets (from age 17 and below, 18-24,
25-29, ..., to 65 and over).

### Intersecting Multiple Columns with CrossedColumn

Using each base feature column separately may not be enough to explain the data.
For example, the correlation between education and the label (earning > 50,000
dollars) may be different for different occupations. Therefore, if we only learn
a single model weight for `education="Bachelors"` and `education="Masters"`, we
won't be able to capture every single education-occupation combination (e.g.
distinguishing between `education="Bachelors" AND occupation="Exec-managerial"`
and `education="Bachelors" AND occupation="Craft-repair"`). To learn the
differences between different feature combinations, we can add **crossed feature
columns** to the model.

```python
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
```

We can also create a `CrossedColumn` over more than two columns. Each
constituent column can be either a base feature column that is categorical
(`SparseColumn`), a bucketized real-valued feature column (`BucketizedColumn`),
or even another `CrossColumn`. Here's an example:

```python
age_buckets_x_race_x_occupation = tf.contrib.layers.crossed_column(
  [age_buckets, race, occupation], hash_bucket_size=int(1e6))
```

## Defining The Logistic Regression Model

After processing the input data and defining all the feature columns, we're now
ready to put them all together and build a Logistic Regression model. In the
previous section we've seen several types of base and derived feature columns,
including:

*   `SparseColumn`
*   `RealValuedColumn`
*   `BucketizedColumn`
*   `CrossedColumn`

All of these are subclasses of the abstract `FeatureColumn` class, and can be
added to the `feature_columns` field of a model:

```python
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  model_dir=model_dir)
```

The model also automatically learns a bias term, which controls the prediction
one would make without observing any features (see the section "How Logistic
Regression Works" for more explanations). The learned model files will be stored
in `model_dir`.

## Training and Evaluating Our Model

After adding all the features to the model, now let's look at how to actually
train the model. Training a model is just a one-liner using the TF.Learn API:

```python
m.fit(input_fn=train_input_fn, steps=200)
```

After the model is trained, we can evaluate how good our model is at predicting
the labels of the holdout data:

```python
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print "%s: %s" % (key, results[key])
```

The first line of the output should be something like `accuracy: 0.83557522`,
which means the accuracy is 83.6%. Feel free to try more features and
transformations and see if you can do even better!

If you'd like to see a working end-to-end example, you can download our [example
code]
(https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)
and set the `model_type` flag to `wide`.

## Adding Regularization to Prevent Overfitting

Regularization is a technique used to avoid **overfitting**. Overfitting happens
when your model does well on the data it is trained on, but worse on test data
that the model has not seen before, such as live traffic. Overfitting generally
occurs when a model is excessively complex, such as having too many parameters
relative to the number of observed training data. Regularization allows for you
to control your model's complexity and makes the model more generalizable to
unseen data.

In the Linear Model library, you can add L1 and L2 regularizations to the model
as:

```
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_race_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir)
```

One important difference between L1 and L2 regularization is that L1
regularization tends to make model weights stay at zero, creating sparser
models, whereas L2 regularization also tries to make the model weights closer to
zero but not necessarily zero. Therefore, if you increase the strength of L1
regularization, you will have a smaller model size because many of the model
weights will be zero. This is often desirable when the feature space is very
large but sparse, and when there are resource constraints that prevent you from
serving a model that is too large.

In practice, you should try various combinations of L1, L2 regularization
strengths and find the best parameters that best control overfitting and give
you a desirable model size.

## How Logistic Regression Works

Finally, let's take a minute to talk about what the Logistic Regression model
actually looks like in case you're not already familiar with it. We'll denote
the label as $$Y$$, and the set of observed features as a feature vector
$$\mathbf{x}=[x_1, x_2, ..., x_d]$$. We define $$Y=1$$ if an individual earned >
50,000 dollars and $$Y=0$$ otherwise. In Logistic Regression, the probability of
the label being positive ($$Y=1$$) given the features $$\mathbf{x}$$ is given
as:

$$ P(Y=1|\mathbf{x}) = \frac{1}{1+\exp(-(\mathbf{w}^T\mathbf{x}+b))}$$

where $$\mathbf{w}=[w_1, w_2, ..., w_d]$$ are the model weights for the features
$$\mathbf{x}=[x_1, x_2, ..., x_d]$$. $$b$$ is a constant that is often called
the **bias** of the model. The equation consists of two parts—A linear model and
a logistic function:

*   **Linear Model**: First, we can see that $$\mathbf{w}^T\mathbf{x}+b = b +
    w_1x_1 + ... +w_dx_d$$ is a linear model where the output is a linear
    function of the input features $$\mathbf{x}$$. The bias $$b$$ is the
    prediction one would make without observing any features. The model weight
    $$w_i$$ reflects how the feature $$x_i$$ is correlated with the positive
    label. If $$x_i$$ is positively correlated with the positive label, the
    weight $$w_i$$ increases, and the probability $$P(Y=1|\mathbf{x})$$ will be
    closer to 1. On the other hand, if $$x_i$$ is negatively correlated with the
    positive label, then the weight $$w_i$$ decreases and the probability
    $$P(Y=1|\mathbf{x})$$ will be closer to 0.

*   **Logistic Function**: Second, we can see that there's a logistic function
    (also known as the sigmoid function) $$S(t) = 1/(1+\exp(-t))$$ being applied
    to the linear model. The logistic function is used to convert the output of
    the linear model $$\mathbf{w}^T\mathbf{x}+b$$ from any real number into the
    range of $$[0, 1]$$, which can be interpreted as a probability.

Model training is an optimization problem: The goal is to find a set of model
weights (i.e. model parameters) to minimize a **loss function** defined over the
training data, such as logistic loss for Logistic Regression models. The loss
function measures the discrepancy between the ground-truth label and the model's
prediction. If the prediction is very close to the ground-truth label, the loss
value will be low; if the prediction is very far from the label, then the loss
value would be high.

## Learn Deeper

If you're interested in learning more, check out our [Wide & Deep Learning
Tutorial](../wide_and_deep/) where we'll show you how to combine
the strengths of linear models and deep neural networks by jointly training them
using the TF.Learn API.
