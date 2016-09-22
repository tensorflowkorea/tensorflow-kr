<!-- This file is machine generated: DO NOT EDIT! -->

# 유틸리티 (contrib)
[TOC]

텐서를 처리하는 유틸리티

## 여러 유틸리티 함수

- - -

### `tf.contrib.util.constant_value(tensor)` {#constant_value}

만약 효율적으로 계산이 가능하다면 tensor의 상수 값을 반환합니다.

이 함수는 텐서가 주어지면 부분적으로 평가를 진행합니다. 성공하는 경우 numpy ndarray 값을 반환합니다.

해야할일(mrry): 이 함수는 손쉽게 확장이 가능하도록 gradients와 ShapeFunctions과 같은 등록 매커니즘을 고려합니다.

주의: 만약 `constant_value(tensor)` 가 non-`None` 결과를 반환하면 tensor에 별다른 값을 부여할 수 없게 됩니다. 이 함수는 구축된 그래프에 영향을 미치도록 허용합니다. 그리고 정적 형상 최적화(permits static shape optimizations)을 허용합니다. 

##### Args:


*  <b>`tensor`</b>: 평가된 텐서

##### Returns:

  numpy ndarray는 'tensor'의 상수 값이거나 계산하지 않았다면 None을 포함합니다.

##### Raises:


*  <b>`TypeError`</b>: 텐서가 작동하지 않는 경우에 타입에러가 발생함.


- - -

### `tf.contrib.util.make_tensor_proto(values, dtype=None, shape=None)` {#make_tensor_proto}

Create a TensorProto.

##### Args:


*  <b>`values`</b>: Values 를 TensorProto에 둡니다.
*  <b>`dtype`</b>: Optional tensor_pb2 데이터 타입 값
*  <b>`shape`</b>: 정수 리스트 형태로 텐서의 차원을 나타냅니다.

##### 반환(Returns):

  TensorProto는 타입에 의존적입니다. TensorProto는 "tensor_content"를 포함합니다. 파이썬 프로그램에서 직접적으로 유용한 기능은 아닙니다.
  값을 평가하기 위해 tensor_util.MakeNdarray(proto)를 이용해 proto를 다시 numpy ndarray로 변환해야합니다.

##### Raises:


*  <b>`TypeError`</b>: 타입이 제공되지 않은 경우.
*  <b>`ValueError`</b>: 인자가 부적절한 값일 경우

make_tensor_proto 는 파이썬 스칼라 값인 "values" 를 받아들입니다. values는 파이썬의 리스트 형태입니다. numpy ndarray 와 numpy scalar와 같습니다. 

만약 "values" 가 파이썬의 스칼라 혹은 리스트 형태라면, make_tensor_proto
first 는 numpy ndarray로 변환됩니다. 만약 dtype 이 없다면, numpy 데이터형을 추론을 시도합니다. 달리 말하면 반환되는 numpy 배열은 주어진 dtype에 호환되는 데이터타입이 됩니다.

In either case above, the numpy ndarray (either the caller provided
or the auto converted) must have the compatible type with dtype.

make_tensor_proto then converts the numpy array to a tensor proto.

If "shape" is None, the resulting tensor proto represents the numpy
array precisely.

Otherwise, "shape" specifies the tensor's shape and the numpy array
can not have more elements than what "shape" specifies.


- - -

### `tf.contrib.util.make_ndarray(tensor)` {#make_ndarray}

Create a numpy ndarray from a tensor.

Create a numpy ndarray with the same shape and data as the tensor.

##### Args:


*  <b>`tensor`</b>: A TensorProto.

##### Returns:

  A numpy array with the tensor contents.

##### Raises:


*  <b>`TypeError`</b>: if tensor has unsupported type.


- - -

### `tf.contrib.util.ops_used_by_graph_def(graph_def)` {#ops_used_by_graph_def}

Collect the list of ops used by a graph.

Does not validate that the ops are all registered.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### Returns:

  A list of strings, each naming an op used by the graph.


- - -

### `tf.contrib.util.stripped_op_list_for_graph(graph_def)` {#stripped_op_list_for_graph}

Collect the stripped OpDefs for ops used by a graph.

This function computes the `stripped_op_list` field of `MetaGraphDef` and
similar protos.  The result can be communicated from the producer to the
consumer, which can then use the C++ function
`RemoveNewDefaultAttrsFromGraphDef` to improve forwards compatibility.

##### Args:


*  <b>`graph_def`</b>: A `GraphDef` proto, as from `graph.as_graph_def()`.

##### Returns:

  An `OpList` of ops used by the graph.

##### Raises:


*  <b>`ValueError`</b>: If an unregistered op is used.


