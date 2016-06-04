# 하우투


## 변수: 생성, 초기화, 저장, 복구

텐서플로우 변수들은 텐서를 가지고 있는 메모리상의 버퍼입니다. 그것들을 이용해서 학습중에 어떻게 모델 파라미터들을 보관하고 업데이트 하는지를 배웁니다.

[튜토리얼 보기](variables/index.md)


## 텐서플로우 구조 101

A step-by-step walk through of the details of using TensorFlow infrastructure
to train models at scale, using MNIST handwritten digit recognition as a toy
example.

[튜토리얼 보기](../tutorials/mnist/tf/index.md)


## 텐서보드: Visualizing Learning

TensorBoard is a useful tool for visualizing the training and evaluation of
your model(s).  This tutorial describes how to build and run TensorBoard as well
as how to add Summary ops to automatically output data to the Events files that
TensorBoard uses for display.

[튜토리얼 보기](summaries_and_tensorboard/index.md)


## TensorBoard: Graph Visualization

This tutorial describes how to use the graph visualizer in TensorBoard to help
you understand the dataflow graph and debug it.

[튜토리얼 보기](graph_viz/index.md)


## Reading Data

This tutorial describes the three main methods of getting data into your
TensorFlow program: Feeding, Reading and Preloading.

[튜토리얼 보기](reading_data/index.md)

## Distributed TensorFlow

This tutorial describes how to execute TensorFlow programs using a cluster of
TensorFlow servers.

[튜토리얼 보기](distributed/index.md)


## Threading and Queues

This tutorial describes the various constructs implemented by TensorFlow
to facilitate asynchronous and concurrent training.

[튜토리얼 보기](threading_and_queues/index.md)


## Adding a New Op

TensorFlow already has a large suite of node operations from which you can
compose in your graph, but here are the details of how to add you own custom Op.

[튜토리얼 보기](adding_an_op/index.md)


## How to write TensorFlow code

Tensorflow Style Guide is set of style decisions that both developers
and users of Tensorflow should follow to increase the readability of their code,
reduce the number of errors, and promote consistency.

[View Style Guide](style_guide.md)


## Writing Documentation

TensorFlow's documentation is largely generated from its source code. Here is an
introduction to the formats we use, a style guide, and instructions on how to
build updated documentation from the source.

[튜토리얼 보기](documentation/index.md)


## Custom Data Readers

If you have a sizable custom data set, you may want to consider extending
TensorFlow to read your data directly in it's native format.  Here's how.

[튜토리얼 보기](new_data_formats/index.md)


## Using GPUs

This tutorial describes how to construct and execute models on GPU(s).

[튜토리얼 보기](using_gpu/index.md)


## Sharing Variables

When deploying large models on multiple GPUs, or when unrolling complex LSTMs
or RNNs, it is often necessary to access the same Variable objects from
different locations in the model construction code.

The "Variable Scope" mechanism is designed to facilitate that.

[튜토리얼 보기](variable_scope/index.md)

## A Tool Developer's Guide to TensorFlow Model Files

If you're developing a tool to load, analyze, or manipulate TensorFlow model
files, it's useful to understand a bit about the format in which they're stored.
This guide covers the details of the saved model format.

[튜토리얼 보기](../how_tos/tool_developers/index.md)

## How to Retrain Inception using Transfer Learning

Training a full object recognition model like Inception takes a long time and a
lot of images. This example shows how to use the technique of transfer learning
to retrain just the final layer of a fully-trained model to recognize new
categories of objects, which is a lot faster and easier than completely
retraining a new model.

[튜토리얼 보기](../how_tos/image_retraining/index.md)

## How to Export and Import a Model

This tutorial describes how to export everything pertaining to a running
model and import it later for various purposes.

[튜토리얼 보기](../how_tos/meta_graph/index.md)

## How to Quantize Neural Networks with TensorFlow

This guide shows how you can convert a float model into one using eight-bit
quantized parameters and calculations. It also describes how the quantization
process works under the hood.

[튜토리얼 보기](../how_tos/quantization/index.md)
