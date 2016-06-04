# 하우투


## 변수: 생성, 초기화, 저장, 복구

텐서플로우 변수들은 텐서를 가지고 있는 메모리상의 버퍼입니다. 그것들을 이용해서 학습중에 어떻게 모델 파라미터들을 보관하고 업데이트 하는지를 배웁니다.

[튜토리얼 보기](variables/index.md)


## 텐서플로우 구조 101

텐서플로우 구조를 사용하여 어느정도 규모가 되는 모델을 학습하는 방법을 차근차근 상세하게 설명합니다. MNIST 필기 숫자 인식을 예제로 사용합니다.

[튜토리얼 보기](../tutorials/mnist/tf/index.md)


## 텐서보드: 학습 시각화

텐서보드 is a useful tool for visualizing the training and evaluation of
your model(s).  This tutorial describes how to build and run 텐서보드 as well
as how to add Summary ops to automatically output data to the Events files that
텐서보드 uses for display.

[튜토리얼 보기](summaries_and_tensorboard/index.md)


## 텐서보드: 그래프 시각화

This tutorial describes how to use the graph visualizer in 텐서보드 to help
you understand the dataflow graph and debug it.

[튜토리얼 보기](graph_viz/index.md)


## 데이터 로딩

This tutorial describes the three main methods of getting data into your
TensorFlow program: Feeding, Reading and Preloading.

[튜토리얼 보기](reading_data/index.md)

## 분산처리

This tutorial describes how to execute TensorFlow programs using a cluster of
TensorFlow servers.

[튜토리얼 보기](distributed/index.md)


## 쓰레드와 큐

This tutorial describes the various constructs implemented by TensorFlow
to facilitate asynchronous and concurrent training.

[튜토리얼 보기](threading_and_queues/index.md)


## 커스텀 연산자

TensorFlow already has a large suite of node operations from which you can
compose in your graph, but here are the details of how to add you own custom Op.

[튜토리얼 보기](adding_an_op/index.md)


## 텐서플로우 코드 작성 스타일

코드의 가독성을 높이고, 에러를 줄이며, 일관성을 장려하기 위해, 텐서플로우 개발자와 사용자들이 따라야 할 스타일 가이드입니다.

[스타일 가이드 보기](style_guide.md)


## 문서화

TensorFlow's documentation is largely generated from its source code. Here is an
introduction to the formats we use, a style guide, and instructions on how to
build updated documentation from the source.

[튜토리얼 보기](documentation/index.md)


## 커스텀 데이터 포맷

상당한 양의 커스텀 데이터를 가지고 있는 경우, 텐서플로우가 데이터 본래의 포맷으로 직접 읽어들이게 하는 방법입니다.

[튜토리얼 보기](new_data_formats/index.md)


## GPU 사용하기

GPU상에서 모델을 구축하고 실행하는 방법을 설명하는 튜토리얼입니다.

[튜토리얼 보기](using_gpu/index.md)


## 변수 공유

When deploying large models on multiple GPUs, or when unrolling complex LSTMs
or RNNs, it is often necessary to access the same Variable objects from
different locations in the model construction code.

The "Variable Scope" mechanism is designed to facilitate that.

[튜토리얼 보기](variable_scope/index.md)

## 모델 파일

If you're developing a tool to load, analyze, or manipulate TensorFlow model
files, it's useful to understand a bit about the format in which they're stored.
This guide covers the details of the saved model format.

[튜토리얼 보기](../how_tos/tool_developers/index.md)

## 트랜스퍼 학습을 이용한 부분 학습

Training a full object recognition model like Inception takes a long time and a
lot of images. This example shows how to use the technique of transfer learning
to retrain just the final layer of a fully-trained model to recognize new
categories of objects, which is a lot faster and easier than completely
retraining a new model.

[튜토리얼 보기](../how_tos/image_retraining/index.md)

## 모델 Export 와 Import

This tutorial describes how to export everything pertaining to a running
model and import it later for various purposes.

[튜토리얼 보기](../how_tos/meta_graph/index.md)

## 텐서플로우로 뉴럴 네트워크를 정량화 하기

This guide shows how you can convert a float model into one using eight-bit
quantized parameters and calculations. It also describes how the quantization
process works under the hood.

[튜토리얼 보기](../how_tos/quantization/index.md)
