# 소개

TensorFlow Learn을 시작하기 위한 간단한 몇 가지 API를 소개합니다.
더 많은 예제들을 보시려면, 이 링크를 참고해주세요. [examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/skflow).


## 일반적인 팁들

- estimator로 전달하기 전에 0 mean이나 단위 표준편차로 리스케일하는 것이 좋습니다. Stochastic Gradient Descent는 변수들의 스케일이 너무 다르면 제대로 작동하지 않습니다.

- 카테고리 변수 역시 estimator에 넣기 전에 전처리가 필요합니다.

## 선형 분류기(Linear Classifier)

간단한 선형 분류기 예:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics

    iris = datasets.load_iris()
    classifier = learn.TensorFlowLinearClassifier(n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
    print("Accuracy: %f" % score)

## 선형 회귀(Linear Regressor)

간단한 선형 회귀 예:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics, preprocessing

    boston = datasets.load_boston()
    X = preprocessing.StandardScaler().fit_transform(boston.data)
    regressor = learn.TensorFlowLinearRegressor()
    regressor.fit(X, boston.target)
    score = metrics.mean_squared_error(regressor.predict(X), boston.target)
    print ("MSE: %f" % score)

## 깊은 인공 신경망(Deep Neural Network)

각각 10, 20, 30개의 히든 유닛을 가진 3개의 레이어에 대한 예제:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics

    iris = datasets.load_iris()
    classifier = learn.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
    print("Accuracy: %f" % score)

## 커스텀 모델(Custom model)

TensorFlowEstimator에 커스텀 모델을 전달하는 예제:

    from tensorflow.contrib import learn
    from sklearn import datasets, metrics

    iris = datasets.load_iris()

    def my_model(X, y):
        """This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability."""
        layers = learn.ops.dnn(X, [10, 20, 10], keep_prob=0.5)
        return learn.models.logistic_regression(layers, y)

    classifier = learn.TensorFlowEstimator(model_fn=my_model, n_classes=3)
    classifier.fit(iris.data, iris.target)
    score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
    print("Accuracy: %f" % score)

## 모델 저장/불러오기(Saving / Restoring models)

각 estimator는 모델 정보가 저장 될 폴더 경로를 인자로 가지는 ``save`` 메소드를 가지고 있습니다. 다시 불러오기 위해서 당신은 단지
``learn.TensorFlowEstimator.restore(path)`` 명령을 사용하면 됩니다. 이는 당신 클래스의 객체를 리턴할 것입니다.

예제 코드:

    from tensorflow.contrib import learn

    classifier = learn.TensorFlowLinearRegression()
    classifier.fit(...)
    classifier.save('/tmp/tf_examples/my_model_1/')

    new_classifier = TensorFlowEstimator.restore('/tmp/tf_examples/my_model_2')
    new_classifier.predict(...)

## 요약(Summaries)

괜찮은 시각화와 요약을 위해서 당신은 ``fit`` 메소드에 ``logdir`` 인자를 추가하면 됩니다. 이는 ``loss`` 와 당신 모델 변수들에 대한 히스토그램을 기록할 것입니다. 당신은 또한 ``tf.summary`` 를 사용하여 커스텀 모델에 대한 요약을 추가할 수도 있습니다. 


    classifier = learn.TensorFlowLinearRegression()
    classifier.fit(X, y, logdir='/tmp/tf_examples/my_model_1/')

위 명령 수행 후 아래 명령을 실행하고:

    tensorboard --logdir=/tmp/tf_examples/my_model_1

reported url로 가면 아래 내용 확인 가능.

Graph visualization: Text classification RNN Graph image

Loss visualization: Text classification RNN Loss image


## 더 많은 예제들은

이 링크를 참고해주세요. [examples folder](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/skflow) for:

-  카테고리 변수를 처리하는 간단한 방법 - 단어들은 단지 하나의 카테고리 변수에 지나지 않는다.
-  텍스트 분류 - 단어와 문자들에 대한 RNN, CNN 예제들
-  언어 모델링과 텍스트 sequence to sequence.
-  이미지들( CNNs) - 숫자 인식 예제를 보시오.
-  심화 내용 - 다양한 DNNs 과 CNNs에 관한 예제들