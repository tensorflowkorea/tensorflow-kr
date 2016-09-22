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

주의: 만약 `constant_value(tensor)` 가 non-`None` 결과를 반하면 tensor에 별다른 값을 부여할 수 없게 됩니다. 이 함수는 구축된 그래프에 영향을 미치도록 허용합니다. 그리고 정적 형상 최적화(permits static shape optimizations)을 허용합니다. 

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


*  <b>`values`</b>: Values to put in the TensorProto.
*  <b>`dtype`</b>: Optional tensor_pb2 DataType value.
*  <b>`shape`</b>: List of integers representing the dimensions of tensor.

##### Returns:

  A TensorProto. Depending on the type, it may contain data in the
  "tensor_content" attribute, which is not directly useful to Python programs.
  To access the values you should convert the proto back to a numpy ndarray
  with tensor_util.MakeNdarray(proto).

##### Raises:


*  <b>`TypeError`</b>: if unsupported types are provided.
*  <b>`ValueError`</b>: if arguments have inappropriate values.

make_tensor_proto accepts "values" of a python scalar, a python list, a
numpy ndarray, or a numpy scalar.

If "values" is a python scalar or a python list, make_tensor_proto
first convert it to numpy ndarray. If dtype is None, the
conversion tries its best to infer the right numpy data
type. Otherwise, the resulting numpy array has a compatible data
type with the given dtype.

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


