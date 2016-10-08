# MNIST Data Download

코드: [tensorflow/examples/tutorials/mnist/](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/)

이 튜토리얼의 목적은 (고전적인) MNIST 데이터를 활용한 필기 숫자의 분류(classification)를 위해 데이터를 어떻게 다운로드 받아야 하는지를 알려주는 것입니다. 

## 튜토리얼 파일

이 튜토리얼은 다음 파일을 참조합니다.

파일 | 목적
--- | ---
[`input_data.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/input_data.py) | 학습과 추정을 위한 MNIST 데이터셋을 다운로드하는 코드

## 데이터 준비

MNIST는 머신러닝의 고전적인 문제입니다. 이 문제는 필기 숫자들의 그레이스케일 28x28 픽셀 이미지를 보고, 0부터 9까지의 모든 숫자들에 대해 이미지가 어떤 숫자를 나타내는지 판별하는 것입니다.

![MNIST Digits](../../../images/mnist_digits.png "MNIST Digits")

좀 더 많은 정보를 원하시면, [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
또는 [Chris Olah's visualizations of MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)를 참고하면 됩니다.

### Download

[Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/) 또한 다운로드를 위한 학습과 테스트 데이터를 호스팅하고 있습니다

파일 | 목적
--- | ---
[`train-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) | 학습 셋 이미지 - 55000개의 트레이닝 이미지, 5000개의 검증 이미지
[`train-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) | 이미지와 매칭되는 학습 셋 레이블 
[`t10k-images-idx3-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz) | 테스트 셋 이미지 - 10000개의 이미지
[`t10k-labels-idx1-ubyte.gz`](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz) | 이미지와 매칭되는 테스트 셋 레이블

`input_data.py`파일에서 `maybe_download()`함수는 학습을 위한 파일들을 로컬 데이터 폴더에 넣을 수 있는지를 확인해줍니다. 

폴더명은 `fully_connected_feed.py`파일의 맨 위에 있는 플래그 변수에 의해 정해지며 원한다면 바꿀 수 있습니다.

### 풀기(Unpack, 언팩)와 변형(Reshape) 

파일들 자체는 표준 이미지 포맷이 아니며 직접 `input_data.py`에 있는 `extract_images()`와 `extract_labels()` 함수를 사용하여 언패킹할 수 있습니다.

이미지 데이터는 `[image index, pixel index]` 형태의 이차원 텐서(여기선 2차원 배열을 의미함)로 추출될 수 있습니다. 각 엔트리는 특정 이미지에서 특정 픽셀의 휘도값이며, `[0, 255]`에서 `[0, 1]`까지 재조정됩니다. "image index"는 데이터셋에 있는 이미지를 가리키며, 0부터 데이터셋의 크기까지 카운팅됩니다. 그리고 "pixel index"는 어떤 이미지에서의 특정 픽셀을 가리키며, 0부터 이미지에 존재하는 픽셀의 갯수까지 존재합니다.

`train-*`파일들에 있는 60000개의 예시들은 학습을 위한 55000개의 예시들과 검증을 위한 5000개의 예시들로 나뉘어집니다. 데이터셋에 있는 모든 28x28 픽셀의 그레이스케일 이미지의 크기는 784이고 따라서 학습 셋 이미지를 위한 출력값 텐서는 `[55000, 784]`의 형태가 됩니다.

레이블 데이터는 각 예시를 위한 클래스 식별자를 값으로써 가지며 `[image index]` 형태의 일차원 텐서로 추출될 수 있습니다. 학습 셋 레이블은 `[55000]`의 형태가 될 것 입니다.

### 데이터셋 객체

이 기본 코드는 다운로드와 압축풀기 그리고 다음의 데이터셋들을 위해 이미지와 레이블을 변형할 것입니다.

데이터셋 | 목적
--- | ---
`data_sets.train` | 초기 학습을 위한 55000개의 이미지들과 레이블들
`data_sets.validation` | 학습 정확도의 반복적 검증을 위한 5000개의 이미지와 레이블들
`data_sets.test` | 학습 정확도의 마지막 테스팅을 위한 10000개의 이미지와 레이블들

`read_data_sets()`함수는 각 세가지 데이터 셋을 위한 `DataSet`인스턴스를 가진 딕셔너리를 리턴합니다. `DataSet.next_batch()`메서드는 `batch_size`개의 이미지 리스트와 레이블들로 이루어진 튜플을 실행중인 TensorFlow 세션에 넣기위해 사용될 수 있습니다.

```python
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
```
