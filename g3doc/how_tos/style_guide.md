# 텐서플로우 스타일 가이드

텐서플로우의 개발자들과 사용자들이 따라야 할 코딩 스타일에 대한 내용입니다. 코드의 가독성을 좋게 하고, 에러를 줄이며, 일관성을 높이는 것이 목적입니다.

[TOC]

## 파이썬 스타일

들여쓰기 2칸이라는 것을 제외하면, 일반적으로
[PEP8 Python style guide](https://www.python.org/dev/peps/pep-0008/) 가이드를 따릅니다.

## 파이썬 2, 3 호환

* 모든 코드는 파이썬 2 및 3과 호환되어야 합니다.

* 모든 파이썬 화일들에 다음 라인들이 들어 있어야 합니다:

```
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
```

* 코드의 호환성을 위해 `six` 를 사용하십시요. (예: `six.range`).


## 베이젤(Bazel) BUILD 규칙

텐서플로우는 Bazel 빌드 시스템을 사용하며, 그에 따라 다음의 조건들이 요구됩니다:

* 모든 BUILD 화일은 다음의 헤더를 포함해야 합니다:

```
# Description:
# <...>

package(
    default_visibility = ["//visibility:private"],
    features = ["-parse_headers"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])
```

* 모든 BUILD 화일의 마지막에는 다음 내용이 있어야 합니다:

```
filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//third_party/tensorflow:__subpackages__"],
)
```

* 새 BUILD 화일을 추가하는 경우, 다음의 내용을 `tensorflow/BUILD` 화일에 추가하여 `all_opensource_files` 타겟이 되도록 합니다.

```
"//third_party/tensorflow/<directory>:all_files",
```

* 모든 파이썬 BUILD 타겟(라이브러리와 테스트)에 다음의 내용을 추가하십시요:

```
srcs_version = "PY2AND3",
```


## 텐서

* Operations that deal with batches may assume that the first dimension of a Tensor is the batch dimension.


## 파이썬 operations

A *Python operation* is a function that, given input tensors and parameters,
creates a part of the graph and returns output tensors.

* The first arguments should be tensors, followed by basic python parameters.
 The last argument is `name` with a default value of `None`.
 If operation needs to save some `Tensor`s to Graph collections,
 put the arguments with names of the collections right before `name` argument.

* Tensor arguments should be either a single tensor or an iterable of tensors.
  E.g. a "Tensor or list of Tensors" is too broad. See `assert_proper_iterable`.

* Operations that take tensors as arguments should call `convert_to_tensor`
 to convert non-tensor inputs into tensors if they are using C++ operations.
 Note that the arguments are still described as a `Tensor` object
 of a specific dtype in the documentation.

* Each Python operation should have an `op_scope` like below.
 Pass list of input tensors, `name` and a default name of the op as arguments.

* Operations should contain an extensive Python comment with Args and Returns
 declarations that explain both the type and meaning of each value. Possible
 shapes, dtypes, or ranks should be specified in the description.
 [See documentation details](documentation/index.md)

* For increased usability include an example of usage with inputs / outputs
 of the op in Example section.

Example:

    def my_op(tensor_in, other_tensor_in, my_param, other_param=0.5,
              output_collections=(), name=None):
    """My operation that adds two tensors with given coefficients.

    Args:
      tensor_in: `Tensor`, input tensor.
      other_tensor_in: `Tensor`, same shape as `tensor_in`, other input tensor.
      my_param: `float`, coefficient for `tensor_in`.
      other_param: `float`, coefficient for `other_tensor_in`.
      output_collections: `tuple` of `string`s, name of the collection to
                          collect result of this op.
      name: `string`, name of the operation.

    Returns:
      `Tensor` of same shape as `tensor_in`, sum of input values with coefficients.

    Example:
      >>> my_op([1., 2.], [3., 4.], my_param=0.5, other_param=0.6,
                output_collections=['MY_OPS'], name='add_t1t2')
      [2.3, 3.4]
    """
    with tf.op_scope([tensor_in, other_tensor_in], name, "my_op"):
      tensor_in = tf.convert_to_tensor(tensor_in)
      other_tensor_in = tf.convert_to_tensor(other_tensor_in)
      result = my_param * tensor_in + other_param * other_tensor_in
      tf.add_to_collections(output_collections, result)
      return result

Usage:

    output = my_op(t1, t2, my_param=0.5, other_param=0.6,
                   output_collections=['MY_OPS'], name='add_t1t2')


## 레이어

A *Layer* is a Python operation that combines variable creation and/or one or many
other graph operations. Follow the same requirements as for regular Python
operation.

* If a layer creates one or more variables, the layer function
 should take next arguments also following order:
  - `initializers`: Optionally allow to specify initializers for the variables.
  - `regularizers`: Optionally allow to specify regularizers for the variables.
  - `trainable`: which control if their variables are trainable or not.
  - `scope`: `VariableScope` object that variable will be put under.
  - `reuse`: `bool` indicator if the variable should be reused if
             it's present in the scope.

* Layers that behave differently during training should have:
  - `is_training`: `bool` to indicate if a training graph is been built.


예:

    def conv2d(inputs,
               num_filters_out,
               kernel_size,
               stride=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               normalization_fn=add_bias,
               normalization_params=None,
               initializers=None,
               regularizers=None,
               trainable=True,
               scope=None,
               reuse=None):
      ... see implementation at tensorflow/contrib/layers/python/layers/layers.py ...

