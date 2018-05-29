# 새 작업을 추가하세요

전제조건:

* C++에 어느정도 친숙할 것.
* 반드시  [TensorFlow binary](../../get_started/os_setup.md#pip-installation)가 설치되어 있거나,
  [downloaded TensorFlow source](../../get_started/os_setup.md#installing-from-sources)가 있어야만,
     빌드 할 수 있음

만약 당신이 존재하는 라이브러리로 감싸져있지 않은 작업을 포함 하길 원한다면, custom Op를 생성할 수 있습니다.
당신의 custom Op를 포함하기 위해서, '이하'의 항목을 충족해야 합니다.

이하:
* C++ 파일에서 새로운 작업을  등록하세요. 그 작업 등록은  실행에서 독립적이고, 그 작업이 들먹여 지는 방법의 의미론을 말합니다.(?)
     예를들어,이것은 작업의 이름을 정의하고 입력과 출력들을 구체적으로 명시합니다.
* C++안에서 그 작업을 실행하세요. 이 실행은 "커널"이라고 불립니다. 그리고 각색의 구조들(CPUs, GPUs) 또는 입출력 형태들을 위한 다양한 커널들이 존재 할 수 있습니다.
* 경우에 따라, 파이썬 래퍼(wrapper)를 만드세요. 이 래퍼는 작업을 생성하는 공용의 API입니다. 기본적인 래퍼는 작업 등록으로 부터 발생되어집니다. 그리고 그것은 직접적으로 사용 되어질 수 있거나 추가 되어질 수 있습니다.
* 경우에 따라, 그 작업을 위해 경사도(gradients)를 계산할 함수를 써넣으세요.
* 경우에 따라, 그 작업을 위해 입출력 모양들을 설명할 함수를 써넣으세요. 이것이 작업 추론으로 하여금 당신의 작업을 다룰 수 있도록 허락합니다.
* 전형적으로, 파이썬에서 그 작업을 테스트 하세요. 만약 당신이 기울기들을 정의한다면, 파이썬으로 그 것들을 식별할 수 있을 것입니다.  [`GradientChecker`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/gradient_checker.py).

[TOC]

## 작업의 인터페이스를 정의하세요

텐써플로우 시스템으로 작업을 등록함으로, 당신은 그 작업의 인터페이스를 정의할 수 있습니다.
등록에서, 당신의 작업 이름과  그 작업의 입력들(형태들와 이름들)과 출력들(형태들과 이름들) 그리고 'docstrings' 과 그 작업이 요구할지도 모를 어떤 속성들을 명시합니다.

이것이 어떻게 동작할지 보기 위해서는, 당신이 'int32'들의 텐서를 챙겨서, 그것의 복사본을 출력하는 작업을 만들고 싶어함에도 불구하고 그 첫번째 요소는 0으로 세트한다고 가정해보세요.
[`tensorflow/core/user_ops`][user_ops]`/zero_out.cc` 파일을 생성하세요. 그리고
 Create file [`tensorflow/core/user_ops`][user_ops]`/zero_out.cc` and '이하'의 작업을 위한 인터페이스를 정의하는  `REGISTER_OP` macro 에의 요청을 추가하세요.

이하 :
```c++
#"tensorflow/core/framework/op.h"를 포함하세요 (Include)

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

이 `ZeroOut` 작업은 텐서 한개를 32비트 정수의 `to_zero`를 입력으로 이해합니다. 그리고 텐서 한 개를 32비트 정수의 `zeroed`로 출력합니다.

> 이름 명명에 관해 주목할 점 : 작업(Op)의 이름은 유일해야하고 'CamelCase'여야 합니다. 밑줄 (`_`)로 시작하는 이름들은 내부 사용을 위해 예약되어집니다.

## 작업을 하기 위해 커널을 실행

당신이 인터페이스를 정의한 후에, 하나 혹은 더 많은 작업의 실행을 제공하세요. 이 커널들 중 한개를 생성하기 위해서, `OpKernel`를 확장하는 클래스 한개를 생성하고 `Compute` 메소드를 오버라이드 하세요.
`Compute`메소드는 입출력 텐써와 같은 유용한 것들에 접근 하게 하는 `OpKernelContext*`타입의  `context` 매개변수 한 개를 제공합니다.

> 중요한 메모 : 당신의 작업커널(OpKernel)의 인스턴스들은 동시에 접근되어질지도 모릅니다. 당신의 `Compute`메소드는 다양한 쓰레드들로 부터 안전하게 연결 되어질 것임에 틀림없습니다.
> 뮤택스의 클래스 멤버와의 연결을 지키세요.(아니면 클래스 맴버를 통한 상태를 공유하지 않는게 낫습니다!  
> 작업 상태를 계속 파악 하기 위해서 [`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h)의 사용을 고려하세요.

당신의 커널을 당신이 먼저 만들어 놓은 파일에 추가하세요. 그 커널은 '이하'의 것과 같이 보일 것입니다.

이하:
```c++
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }

    // Preserve the first input value if possible.
    if (N > 0) output(0) = input(0);
  }
};
```

커널을 실행 한 뒤, 텐서플로우 시스템에 그것을 등록합니다. 등록할 때, 당신은 이 커널이 동작하게 될 다른 제약사항들을 명시 합니다.
예를 들어, 당신이 하나의 커널을 CPUs를 위해 그리고 다른 하나는 GPUs를 위해 만들수도 있습니다.
`ZeroOut` 작업을 위한 이 일을 하기 위해서, `zero_out.cc`를 따라서 추가하세요.

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

## 작업 라이브러리를 빌드
### 텐써플로우 바이너리 설치도 병행

당신의 시스템에서 동작 할 수 있는 `g++` 또는 `clang`과 같은 `C++` 컴파일러로 `zero_out.cc`을 컴파일 할 수 있어야 합니다.
바이너리 PIP 패키지는 당신의 작업을 시스템이 명시한 곳에서 컴파일 해야만 하는 라이브러리와 헤더파일을 설치합니다.
그러나, 텐써플로우 파이썬 라이브러리는 `get_include` 함수를 제공합니다. 이 함수는 헤더 디렉토리를 얻게 합니다.
여기 이 함수의 출력된 값을 우분투 머신에서 볼 수 있습니다.

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python2.7/site-packages/tensorflow/include'

```
가령 당신이 설치된 `g++`를 가졌다고 가정해 본다면, 당신이 다이나믹 라이브러리 안에서 작업을 컴파일 하는 것을 가능하도록 해주는 커맨드들의 흐름들이 '이하'에 있습니다.

이하:
```bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC
```

맥 OS에서, "-undefined dynamic_lookup"라는 추가적인 표시사항은  .so 파일을 빌드할 때 필수 적으로 필요합니다.

> gcc 5버전에서의 주의사항 : gcc 5는 새로운 C++을 사용합니다. [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx).
텐써플로우 웹사이트에서 이용 가능한 바이너리 pip 패키지들은 더 오래된 ABI를 사용하는 gcc4로 빌드되어졌습니다.
만약 당신이 gcc5로 작업 라이브러리를 컴파일 한다면,  `-D_GLIBCXX_USE_CXX11_ABI=0`를 커맨드라인에 추가해야합니다.
왜냐하면, 그 라이브러리를 오래된 abi와 호환가능하게 해야하기 때문입니다.

### 텐써플로우 소스 설치와 함께

만약 당신이 텐써플로우를 다 설치 했다면, 당신의 작업을 컴파일하는 텐써플로우의 빌드 시스템을 이용할 수 있습니다.
Bazel 빌드 규칙([`tensorflow/core/user_ops`][user_ops] 디렉토리)을 따라 빌드 파일을 가져다 놓으세요.

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

'이하'의 `zero_out.so`를 빌드하는 명령을 실행하세요.

이하:
```bash
$ bazel build -c opt //tensorflow/core/user_ops:zero_out.so
```

> 알림:
표준 `cc_library` 규칙으로, 당신이 공유된 라이브러리 ( `.so` 파일)를 생성할 수 있음에도 불구하고, `tf_custom_op_library`매크로를 사용할 것을 강력하게 권고합니다.
이것이 어떤 의존들(dependencies)을 추가하고,공유된 라이브러리가 텐써플로우의 플러그인 로딩 구조와 호환이 되는지 점검합니다.

## 파이썬에서의 작업 실행

텐써플로우 파이썬 API는 역동적인 라이브러리를 로드하는 것과 텐써플로우 프레임워크에 작업을 등록하기 위해서 [load_op_library](../../api_docs/python/framework#load_op_library) 함수를 제공합니다.
`load_op_library`는 작업을 위한 파이썬 래퍼들을 담고 있는 파이썬 모듈을 반환합니다.
게다가, 당신이 그 작업을 빌드 했다면, 파이썬으로 부터 이하의 작업을 실행 할 수 있습니다.
이하:

```파이썬
import tensorflow as tf
zero_out_module = tf.load_op_library('zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# 프린트들
array([[1, 0],
       [0, 0]], dtype=int32)
```

> 알림: 발생된 함수는 ([PEP8](https://www.python.org/dev/peps/pep-0008/))을 준수하기 위해서 뱀형(snake/_case)이름을 받을 것입니다.
> 그래서 만약   C++ 파일에서 `ZeroOut`으로 작업이름을 명명한다면, 파이썬 함수는  `zero_out`로 쓰여질 것입니다.

파이썬 모듈로 부터 일반적인 함수 `import`-able로써, 그 작업이 사용가능해지도록 하기위해서, 파이썬 소스파일(이하 참조 : [zero_out_op_1.py](https://www.tensorflow.org/code/tensorflow/g3doc/how_tos/adding_an_op/zero_out_op_1.py))에서  `load_op_library`을 가지고 있는 것이 유용할지도 모릅니다.

이하:
```python
tf로써 텐써플로우를 임포트

_zero_out_module = tf.load_op_library('zero_out_op_kernel_1.so')
zero_out = _zero_out_module.zero_out
```

## 이것이 작동하는지 확인하세요.

당신이 성공적으로 작업을 수행했다는 것을 확인할 좋은 방법은 테스트를 작성 하는 것입니다. `tensorflow/python/kernel_tests/zero_out_op_test.py` 파일을 '이하'의 내용으로 작성하세요.

이하:
```python
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])
```

이렇게 한 이후에 당신의 테스트를 실행해 보세요.

```sh
$ bazel test tensorflow/python:zero_out_op_test
```

## 유효성
## Validation

이 예제는 작업이 먼저 어떤 모양의 텐써에 적용 했다는 것을 가정합니다. 만약에 이것을 오직 벡터에만 적용한다면 어떻게 될까요?
이 말은 확인(check)을 OpKernel 구현 위에 추가한다는 것을 의미 합니다.

```c++
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

이것은 '입력은 한 벡터다'라고 주장합니다. 만약 입력이 있지 않다면, 이것은 `InvalidArgument` 상태를 세팅한 것를 되돌립니다.
[`OP_REQUIRES` macro][validation-macros]은 3요소를 가지고 있습니다.

*	`context`의 `SetStatus()`메소드 를 위해서`OpKernelContext` 혹은 `OpKernelConstruction` 포인터(참조: [`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h) 둘 중 하나가 될수 있는 `context`.
*	'상태(condition)'. 예를들어, [`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h)에서의 텐써 모양을 확인하기 위한 함수들이 있습니다.
*	`Status` 객체에 의해 보여지는 '에러 그자체' (참조 : [`tensorflow/core/lib/core/status.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/status.h)
	`Status`는 타입(종종 `InvalidArgument`이긴 하나, 타입들의 리스트를 봅니다)과 메시지를 가집니다. 에러를 구성하는 것을 위한 함수들은 [`tensorflow/core/lib/core/errors.h`][validation-macros]에서 찾을지도 모릅니다.

그렇지 않으면, 만약 당신이 어떤 함수로 부터 반환되어진 `Status`객체가 오류인지 아닌지를 테스트 하고, 그것을 반환하기를 원한다면 [`OP_REQUIRES_OK`][validation-macros]를 사용하세요. 이 두가지 매크로들은 오류에 걸린 함수로 부터 되돌아 옵니다.

## 작업 등록

### 속성들

작업들은 속성을 가질 수 있습니다.그리고, 속성의 값은 작업이 그래프에 추가되어질때 할당되어 집니다. 이 속성들은 작업의 환경 설정을 위해 사용되어 지며, 속성들의 값은 커널 구현과 작업 등록의 입출력 형태안에서 접근되어 질 수 있습니다.
입력이 가능 할때, 속성보다 입력들이 좀 더 유연하기 때문에, 속성보단 입력을 사용할 것을 권장합니다.
입력들은 모든 단계들을 바꿀수 있고, feed를 사용할 준비 등등을 할 수 있습니다.
속성들은 특징(숫자 혹은 입출력의 형태)에 영향을 주거나 단계별로 변경할 수 없는 환경설정들과 같은 입력 을 끝마칠 수 없는 것들을 위해 사용되어 집니다.

당신은 작업을 등록 할 때, `Attr`메소드를 사용하는 속성의 이름과 타입을 명시함으로  속성을 정의합니다.

`Attr` 메소드에서 예상할 수 있는 형태:

```
<name>: <attr-type-expr>
```

`<name>`이 한 글자로 시작하고, 글자와 숫자로 쓴 문자와 밑줄, 그리고 `<attr-type-expr>`로 구성되어 질 수 있는 곳은 '이하'에 표현된 폼의 형태입니다. (#attr-types)

이하:
예를들어, 만약 당신이 `ZeroOut`작업이 사용자 지정 색인을 보존하기 원한다면, 단지 0번째 요소 대신에 다음과 같은 작업을 등록 할 수 있습니다.
<code class="lang-c++"><pre>
REGISTER\_OP("ZeroOut")
    <b>.Attr("preserve\_index: int")</b>
    .Input("to\_zero: int32")
    .Output("zeroed: int32");
</pre></code>

당신의 커널은 `context` 파라미터를 통해서 이것의 constructor안에 있는 이 속성에 접근 할 수 있습니다.

<code class="lang-c++"><pre>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction\* context) : OpKernel(context) {<b>
    // Get the index of the value to preserve
    OP\_REQUIRES\_OK(context,
                   context-&gt;GetAttr("preserve\_index", &preserve\_index\_));
    // Check that preserve\_index is positive
    OP\_REQUIRES(context, preserve\_index_ &gt;= 0,
                errors::InvalidArgument("Need preserve\_index &gt;= 0, got ",
                                        preserve\_index_));
  </b>}
  void Compute(OpKernelContext\* context) override {
    // ...
  }
 <b>private:
  int preserve\_index\_;</b>
};
</pre></code>

`Compute` 메소드에서 사용 가능 한 것 :

<code class="lang-c++"><pre>
  void Compute(OpKernelContext\* context) override {
    // ...
<br/>    <b>// Check that preserve\_index is in range
    OP\_REQUIRES(context, preserve\_index_ &lt; input.dimension(0),
                errors::InvalidArgument("preserve\_index out of range"));<br/>
    </b>// Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output\_flat(i) = 0;
    }<br/>
    <b>// Preserve the requested input value
    output\_flat(preserve\_index\_) = input(preserve\_index\_);</b>
  }
</pre></code>

> [backwards compatibility](#backwards-compatibility) 를 보호하기 위해서, 당신이 '이하'존재하는 작업에 속성을 추가할 때  [default value](#default-values-constraints)를 명시해야 합니다.
> 이하:
> <code class="lang-c++"><pre>
> REGISTER\_OP("ZeroOut")
>     <b>.Attr("preserve\_index: int = 0")</b>
>     .Input("to\_zero: int32")
>     .Output("zeroed: int32");
> </pre></code>

### 속성 타입들

'이하'의 타입들은 속성에서 지원됩니다.

이하:
* `string`: 바이트들의 연속 (UTF8이 필수는 아님).
* `int`: 부호가 붙은 정수형.
* `float`: 부동 소수점 숫자.
* `bool`: 참 혹은 거짓.
* `type`: [`DataType`][DataTypeString]의 불 참조 값들 중에 하나.
* `shape`: [`TensorShapeProto`][TensorShapeProto].
* `tensor`: [`TensorProto`][TensorProto].
* `list(<type>)`: `<type>`이 상위 타입들 중 하나인 곳에서 `<type>`의 리스트.
  `list(list(<type>))`가 유효하지 않음을 주의하세요 .

참조 : 최종 리스트를 위한  [`op_def_builder.cc:FinalizeAttr`][FinalizeAttr]

#### 기본 값 & 제약사항

속성들은 기본 값들을 가지고 있을것이다. 그리고 속성의 어떤 타입들은 제약사항을 가질 수 있다. 제약사항이 있는 속성을 정의하기 위해선, '아래'의 `<attr-type-expr>`를 이용 할 수 있습니다.
이하:

* `{'<string1>', '<string2>'}`: 값은 `<string1>` 혹은 `<string2>` 둘중 하나를 가지고 있는  'string'이여야만 합니다.
     당신이 이 문법을 사용할 때, 타입의 이름인 `string`은 암시되어집니다.  
     이것은 'enum'을 모방합니다 :
  ```c++
  REGISTER_OP("EnumExample")
      .Attr("e: {'apple', 'orange'}");
  ```

* `{<type1>, <type2>}`: 값은 `type`타입이고, `<type1>` 이나 `<type2>`이 '[tensor types](../../resources/dims_types.md#data-types)'에 의해 지원되는 곳에서  `<type1>` 혹은 `<type2>` 중 하나여야만 합니다.
      당신은 '속성의 타입이 `type`이다.'라고 명시하지 않습니다. 이것은 당신이 `{...}`안에서 타입의 리스트를 가질때 암시되어집니다.
      예를 들어 이 경우엔,  `t`속성이  `int32`, `float`, `bool` 중 하나여야만 하는 타입 입니다 :

  ```c++
  REGISTER_OP("RestrictedTypeExample")
      .Attr("t: {int32, float, bool}");
  ```

* 일반적인 타입의 제약사항들을 위해 여기 몇가지 손쉬운 방법이 있습니다 :
    * `numbertype`: `type`타입은 숫자형으로 제한됩니다. (non-string and non-bool)
    * `realnumbertype`: 복잡한 타입이 없이 `numbertype`와 같습니다.
    * `quantizedtype`: quantized 숫자를 제외한  `numbertype`와 같습니다.

	이러한 것들로 허가되어진 타입들의 구체적인 리스트들은 함수들('이하'참조)에 의해 정의 되어집니다.
	이하 :
    [`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h).에 있는 `NumberTypes()`와 같습니다.

        이 사례에서 `t` 속성은 반드시 숫자 타입들중 하나여야만 합니다 :

    ```c++
    REGISTER_OP("NumberType")
        .Attr("t: numbertype");
    ```

        이 작업(op)을 위해서:

    ```python
    tf.number_type(t=tf.int32)  # Valid
    tf.number_type(t=tf.bool)   # Invalid
    ```
* `int >= <n>`: 이 값은 자연수인 `<n>`보다 크거나 같은 값이어야 합니다.

    예를들어, '아래'의 작업 등록은 `a`속성이 최소 `2`인 값을 가지고 있다는 것을 명시합니다.
    아래:

  ```c++
  REGISTER_OP("MinIntExample")
      .Attr("a: int >= 2");
  ```

* `list(<type>) >= <n>`: `<n>`보다 크거나 같은 길이를 가진 `<type>` 타입의 리스트입니다.

    예를 들어, '아래'의 작업등록은 `a`속성이 `int32` 혹은 `float` 둘중 하나의 타입의 리스트이고, 적어도 그것 들 중에서 3이있어야만 한다는 것을 명시합니다.
    아래:

  ```c++
  REGISTER_OP("TypeListExample")
      .Attr("a: list({int32, float}) >= 3");
  ```

발생된 코드에 값을 선택적으로 하는 속성을 위한 기본 값을 할당하기 위해서는, 끝 부분에 `= <default>`를 추가하세요.
예:

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

기본값에 대해 지원 되는 문법은 GraphDef 의미를 결과로 내는 것들 중에서 프로토(proto)표시에 사용 되어진다.

모든 타입의 기본값을 명시하는 방법에 대한 예 :

```c++
REGISTER_OP("AttrDefaultExampleForAllTypes")
   .Attr("s: string = 'foo'")
   .Attr("i: int = 0")
   .Attr("f: float = 1.0")
   .Attr("b: bool = true")
   .Attr("ty: type = DT_INT32")
   .Attr("sh: shape = { dim { size: 1 } dim { size: 2 } }")
   .Attr("te: tensor = { dtype: DT_INT32 int_val: 5 }")
   .Attr("l_empty: list(int) = []")
   .Attr("l_int: list(int) = [2, 3, 5, 7]");
```

특히 `type` 타입의 값들을 사용하는 것에 주의하세요.  [타입을 위한 `DT_*` 이름들](../../resources/dims_types.md#data-types).

### 다형성 (Polymorphism)
#### 타입 다형성  

Input으로서 다른 타입을 가질 수 있는 작업이나 다른 ouput 타입을 내보내는 작업에 대하여, 당신은 작업등록(Op registration)안의 [input타입이나 output타입](#inputs-outputs)에서 [속성](#attrs)을 명시해줄 수 있습니다. 전형적으로 그런 뒤 당신은 각 지원된 타입들에 대해 `OpKernel`를 등록할 수 있습니다.

예를 들어서, 당신이 `ZeroOut` 작업을 `int32`타입이나 `float`타입에서 하고 싶다면, 당신의 작업 등록(Op Registration)은 다음과 같을 것입니다:

<code class="lang-c++"><pre>
REGISTER\_OP("ZeroOut")
    <b>.Attr("T: {float, int32}")</b>
    .Input("to\_zero: <b>T</b>")
    .Output("zeroed: <b>T</b>");
</pre></code>

당신의 작업등록은 이제 input타입이 `float`타입 또는 `int32`타입에서 이루어지고, 둘 다 `T`타입을 가지고 있기 때문에, output타입이 (input 타입과) 같은 타입일 것이라고 명시합니다.

> <a id="naming"></a>이름짓기에 관한 메모: 입력(Inputs), 출력(Outputs), 그리고 속성은 일반적으로 이름이 snake\_case로 주어져야 합니다. 한 가지 예외는 속성(attrs)이 input의 타입이나 input의 타입으로 주어진 경우입니다(?). 그러한 속성들은 작업이 그래프에 추가되었고 작업의 함수에 보이지 않을 때 추론될 수 있습니다. 예를 들어, 이 ZeroOut의 마지막 정의는 파이썬 함수를 다음과 같이 보이도록 만들것입니다:
>
> ```python
> def zero_out(to_zero, name=None):
>   """...
>   Args:
>     to_zero: A `Tensor`. Must be one of the following types:
>         `float32`, `int32`.
>     name: A name for the operation (optional).
>
>   Returns:
>     A `Tensor`. Has the same type as `to_zero`.
>   """
> ```
>
> 만약 `to_zero`가 `int32`텐서로 넘겨졌다면, `T`는 자동적으로 `int32`로 설정됩니다(사실상 `DT_INT32`이겠죠?). 그러한 추론된 속성(attrs)들은 Capitalized 또는 CamelCase의 이름으로 주어질 것입니다.
>
> 이것을 output 타입을 결정하는 attr 타입을 가지고 있는 작업과 비교해보세요!
>
> ```c++
> REGISTER_OP("StringToNumber")
>     .Input("string_tensor: string")
>     .Output("output: out_type")
>     .Attr("out_type: {float, int32}");
>     .Doc(R"doc(
> Converts each string in the input Tensor to the specified numeric type.
> )doc");
> ```
>
> 이러한 경우에, 유저는 output 타입을 생성된 파이썬 같이 명시해줘야합니다:
>
> ```python
> def string_to_number(string_tensor, out_type=None, name=None):
>   """Converts each string in the input Tensor to the specified numeric type.
>
>   Args:
>     string_tensor: A `Tensor` of type `string`.
>     out_type: An optional `tf.DType` from: `tf.float32, tf.int32`.
>       Defaults to `tf.float32`.
>     name: A name for the operation (optional).
>
>   Returns:
>     A `Tensor` of type `out_type`.
>   """
> ```

<code class="lang-c++"><pre>
\#include "tensorflow/core/framework/op_kernel.h"<br/>
class ZeroOut<b>Int32</b>Op : public OpKernel {
  // as before
};<br/>
class ZeroOut<b>Float</b>Op : public OpKernel {
 public:
  explicit ZeroOut<b>Float</b>Op(OpKernelConstruction\* context)
      : OpKernel(context) {}<br/>
  void Compute(OpKernelContext\* context) override {
    // Grab the input tensor
    const Tensor& input\_tensor = context-&gt;input(0);
    auto input = input\_tensor.flat&lt;<b>float</b>&gt;();<br/>
    // Create an output tensor
    Tensor* output = NULL;
    OP\_REQUIRES\_OK(context,
                   context-&gt;allocate\_output(0, input_tensor.shape(), &output));
    auto output\_flat = output-&gt;template flat&lt;<b>float</b>&gt;();<br/>
    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i &lt; N; i++) {
      output\_flat(i) = 0;
    }<br/>
    // Preserve the first input value
    if (N &gt; 0) output\_flat(0) = input(0);
  }
};<br/><b>
// Note that TypeConstraint&lt;int32&gt;("T") means that attr "T" (defined
// in the Op registration above) must be "int32" to use this template
// instantiation.</b>
REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    <b>.TypeConstraint&lt;int32&gt;("T"),</b>
    ZeroOutOp<b>Int32</b>);
<b>REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;float&gt;("T"),
    ZeroOutFloatOp);
</b></pre></code>

> [backwards compatibility](#backwards-compatibility)를 방지하기 위해서, 다음과 같이
> 존재하는 작업에 속성(attr)을 추가할 때, 당신은
> [default값](#default-values-constraints)를 명시해줘야 합니다.
>
> <code class="lang-c++"><pre>
> REGISTER\_OP("ZeroOut")
>   <b>.Attr("T: {float, int32} = DT_INT32")</b>
>   .Input("to\_zero: T")
>   .Output("zeroed: T")
> </pre></code>


당신이 더 많은 타입들을 추가하고 싶다고 해봅시다. `double`이라고 해볼까요?:

<code class="lang-c++"><pre>
REGISTER\_OP("ZeroOut")
    <b>.Attr("T: {float, <b>double,</b> int32}")</b>
    .Input("to\_zero: <b>T</b>")
    .Output("zeroed: <b>T</b>");
</pre></code>

위처럼 장황한 코드로 또다른 `OpKernel`을 작성하는 것 대신에, 당신은 C++ 템플릿을 사용할 수 있을 것입니다. 이렇게 하더라도 당신은 overload당 하나의 커널등록 (`REGISTER\_KERNEL\_BUILDER` call)을 가질 것입니다.


<code class="lang-c++"><pre>
<b>template &lt;typename T&gt;</b>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction\* context) : OpKernel(context) {}<br/>
  void Compute(OpKernelContext\* context) override {
    // Grab the input tensor
    const Tensor& input\_tensor = context-&gt;input(0);
    auto input = input\_tensor.flat<b>&lt;T&gt;</b>();<br/>
    // Create an output tensor
    Tensor* output = NULL;
    OP\_REQUIRES\_OK(context,
                   context-&gt;allocate\_output(0, input_tensor.shape(), &output));
    auto output\_flat = output-&gt;template flat<b>&lt;T&gt;</b>();<br/>
    // Set all the elements of the output tensor to 0
    const int N = input.size();
    for (int i = 0; i &lt; N; i++) {
      output\_flat(i) = 0;
    }<br/>
    // Preserve the first input value
    if (N &gt; 0) output\_flat(0) = input(0);
  }
};<br/>
// Note that TypeConstraint&lt;int32&gt;("T") means that attr "T" (defined
// in the Op registration above) must be "int32" to use this template
// instantiation.</b>
REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;int32&gt;("T"),
    <b>ZeroOutOp&lt;int32&gt;</b>);
REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;float&gt;("T"),
    <b>ZeroOutOp&lt;float&gt;</b>);
<b>REGISTER\_KERNEL\_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE\_CPU)
    .TypeConstraint&lt;double&gt;("T"),
    ZeroOutOp&lt;double&gt;);
</b></pre></code>


Overload가 한 두개가 아니라면, 당신은 매크로에 등록을 추가시킬 수 있습니다.

```c++
#include "tensorflow/core/framework/op_kernel.h"

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
```


당신이 커널을 등록하는 타입들의 목록에 따라서, 당신은 이곳에서 제공하는 매크로를 사용할 수 있을 것입니다 ->
[`tensorflow/core/framework/register_types.h`][register_types]:

```c++
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

REGISTER_OP("ZeroOut")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T");

template <typename T>
class ZeroOutOp : public OpKernel { ... };

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
```

#### List Inputs and Outputs

In addition to being able to accept or produce different types, ops can consume
or produce a variable number of tensors.

In the next example, the attr `T` holds a *list* of types, and is used as the
type of both the input `in` and the output `out`.  The input and output are
lists of tensors of that type (and the number and types of tensors in the output
are the same as the input, since both have type `T`).

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

You can also place restrictions on what types can be specified in the list. In
this next case, the input is a list of `float` and `double` tensors. The Op
accepts, for example, input types `(float, double, float)` and in that case the
output type would also be `(float, double, float)`.

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

If you want all the tensors in a list to be of the same type, you might do
something like:

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

This accepts a list of `int32` tensors, and uses an `int` attr `N` to
specify the length of the list.

This can be made [type polymorphic](#type-polymorphism) as well.  In the next
example, the input is a list of tensors (with length `"N"`) of the same (but
unspecified) type (`"T"`), and the output is a single tensor of matching type:

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

By default, tensor lists have a minimum length of 1. You can change that default
using
[a `">="` constraint on the corresponding attr](#default-values-constraints).
In this next example, the input is a list of at least 2 `int32` tensors:

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

The same syntax works with `"list(type)"` attrs:

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

### Inputs and Outputs

To summarize the above, an Op registration can have multiple inputs and outputs:

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

Each input or output spec is of the form:

```
<name>: <io-type-expr>
```

where `<name>` begins with a letter and can be composed of alphanumeric
characters and underscores. `<io-type-expr>` is one of the following type
expressions:

* `<type>`, where `<type>` is a supported input type (e.g. `float`, `int32`,
  `string`). This specifies a single tensor of the given type.

  See
  [the list of supported Tensor types](../../resources/dims_types.md#data-types).

  ```c++
  REGISTER_OP("BuiltInTypesExample")
      .Input("integers: int32")
      .Input("complex_numbers: complex64");
  ```

* `<attr-type>`, where `<attr-type>` is the name of an [Attr](#attrs) with type
  `type` or `list(type)` (with a possible type restriction). This syntax allows
  for [polymorphic ops](#polymorphism).

  ```c++
  REGISTER_OP("PolymorphicSingleInput")
      .Attr("T: type")
      .Input("in: T);

  REGISTER_OP("RestrictedPolymorphicSingleInput")
      .Attr("T: {int32, int64}")
      .Input("in: T);
  ```

  Referencing an attr of type `list(type)` allows you to accept a sequence of
  tensors.

  ```c++
  REGISTER_OP("ArbitraryTensorSequenceExample")
      .Attr("T: list(type)")
      .Input("in: T")
      .Output("out: T");

  REGISTER_OP("RestrictedTensorSequenceExample")
      .Attr("T: list({int32, int64})")
      .Input("in: T")
      .Output("out: T");
  ```

  Note that the number and types of tensors in the output `out` is the same as
  in the input `in`, since both are of type `T`.

* For a sequence of tensors with the same type: `<number> * <type>`, where
  `<number>` is the name of an [Attr](#attrs) with type `int`.  The `<type>` can
  either be
  [a specific type like `int32` or `float`](../../resources/dims_types.md#data-types),
  or the name of an attr with type `type`.  As an example of the first, this
  Op accepts a list of `int32` tensors:

  ```c++
  REGISTER_OP("Int32SequenceExample")
      .Attr("NumTensors: int")
      .Input("in: NumTensors * int32")
  ```

  Whereas this Op accepts a list of tensors of any type, as long as they are all
  the same:

  ```c++
  REGISTER_OP("SameTypeSequenceExample")
      .Attr("NumTensors: int")
      .Attr("T: type")
      .Input("in: NumTensors * T")
  ```

* For a reference to a tensor: `Ref(<type>)`, where `<type>` is one of the
  previous types.

> A note on naming: Any attr used in the type of an input will be inferred.  By
> convention those inferred attrs use capital names (like `T` or `N`).
> Otherwise inputs, outputs, and attrs have names like function parameters
> (e.g. `num_outputs`).  For more details, see the
> [earlier note on naming](#naming).

For more details, see
[`tensorflow/core/framework/op_def_builder.h`][op_def_builder].

### Backwards compatibility

In general, changes to specifications must be backwards-compatible: changing the
specification of an Op must not break prior serialized `GraphDef` protocol
buffers constructed from older specfications.  The details of `GraphDef`
compatibility are [described here](../../resources/versions.md#graphs).

There are several ways to preserve backwards-compatibility.

1. Any new attrs added to an operation must have default values defined, and
   with that default value the Op must have the original behavior. To change an
   operation from not polymorphic to polymorphic, you *must* give a default
   value to the new type attr to preserve the original signature by default. For
   example, if your operation was:

   ```c++
   REGISTER_OP("MyGeneralUnaryOp")
       .Input("in: float")
       .Output("out: float");
   ```

   you can make it polymorphic in a backwards-compatible way using:

   ```c++
   REGISTER_OP("MyGeneralUnaryOp")
       .Input("in: T")
       .Output("out: T")
       .Attr("T: numerictype = DT_FLOAT");
   ```

2. You can safely make a constraint on an attr less restrictive.  For example,
   you can change from `{int32, int64}` to `{int32, int64, float}` or `type`.
   Or you may change from `{"apple", "orange"}` to `{"apple", "banana",
   "orange"}` or `string`.

3. You can change single inputs / outputs into list inputs / outputs, as long as
   the default for the list type matches the old signature.

4. You can add a new list input / output, if it defaults to empty.

5. Namespace any new Ops you create, by prefixing the Op names with something
   unique to your project. This avoids having your Op colliding with any Ops
   that might be included in future versions of Tensorflow.

6. Plan ahead! Try to anticipate future uses for the Op. Some signature changes
   can't be done in a compatible way (for example, making a list of the same
   type into a list of varying types).

The full list of safe and unsafe changes can be found in
[`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc).
If you cannot make your change to an operation backwards compatible, then create
a new operation with a new name with the new semantics.

Also note that while these changes can maintain `GraphDef` compatibility, the
generated Python code may change in a way that isn't compatible with old
callers.  The Python API may be kept compatible by careful changes in a
hand-written Python wrapper, by keeping the old signature except possibly adding
new optional arguments to the end.  Generally incompatible changes may only be
made when TensorFlow's changes major versions, and must conform to the
[`GraphDef` version semantics](../../resources/versions.md#graphs).

## GPU Support

You can implement different OpKernels and register one for CPU and another for
GPU, just like you can [register kernels for different types](#polymorphism).
There are several examples of kernels with GPU support in
[`tensorflow/core/kernels/`](https://www.tensorflow.org/code/tensorflow/core/kernels/).
Notice some kernels have a CPU version in a `.cc` file, a GPU version in a file
ending in `_gpu.cu.cc`, and some code shared in common in a `.h` file.

For example, the [`pad` op](../../api_docs/python/array_ops.md#pad) has
everything but the GPU kernel in [`tensorflow/core/kernels/pad_op.cc`][pad_op].
The GPU kernel is in
[`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc),
and the shared code is a templated class defined in
[`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h).
One thing to note, even when the GPU kernel version of `pad` is used, it still
needs its `"paddings"` input in CPU memory.  To mark that inputs or outputs are
kept on the CPU, add a `HostMemory()` call to the kernel registration, e.g.:

```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

### Compiling the kernel for the GPU device

Look at
[cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/g3doc/how_tos/adding_an_op/cuda_op_kernel.cu.cc)
for an example that uses a CUDA kernel to implement an op. The
`tf_custom_op_library` accepts a `gpu_srcs` argument in which the list of source
files containing the CUDA kernels (`*.cu.cc` files) can be specified. For use
with a binary installation of TensorFlow, the CUDA kernels have to be compiled
with NVIDIA's `nvcc` compiler. Here is the sequence of commands you can use to
compile the
[cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/g3doc/how_tos/adding_an_op/cuda_op_kernel.cu.cc)
and
[cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/g3doc/how_tos/adding_an_op/cuda_op_kernel.cc)
into a single dynamically loadable library:

```bash
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
cuda_op_kernel.cu.o -I $TF_INC -fPIC -lcudart
```

`cuda_op_kernel.so` produced above can be loaded as usual in Python, using the
`tf.load_op_library` function.

## Implement the gradient in Python

Given a graph of ops, TensorFlow uses automatic differentiation
(backpropagation) to add new ops representing gradients with respect to the
existing ops (see
[Gradient Computation](../../api_docs/python/train.md#gradient-computation)).
To make automatic differentiation work for new ops, you must register a gradient
function which computes gradients with respect to the ops' inputs given
gradients with respect to the ops' outputs.

Mathematically, if an op computes \\(y = f(x)\\) the registered gradient op
converts gradients \\(\partial / \partial y\\) with respect to \\(y\\) into
gradients \\(\partial / \partial x\\) with respect to \\(x\\) via the chain
rule:

$$\frac{\partial}{\partial x}
    = \frac{\partial}{\partial y} \frac{\partial y}{\partial x}
    = \frac{\partial}{\partial y} \frac{\partial f}{\partial x}.$$

In the case of `ZeroOut`, only one entry in the input affects the output, so the
gradient with respect to the input is a sparse "one hot" tensor.  This is
expressed as follows:

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  first_grad = array_ops.reshape(grad, [-1])[0]
  to_zero_grad = sparse_ops.sparse_to_dense(index, shape, first_grad, 0)
  return [to_zero_grad]  # List of one Tensor, since we have one input
```

Details about registering gradient functions with
[`ops.RegisterGradient`](../../api_docs/python/framework.md#RegisterGradient):

* For an op with one output, the gradient function will take an
  [`Operation`](../../api_docs/python/framework.md#Operation) `op` and a
  [`Tensor`](../../api_docs/python/framework.md#Tensor) `grad` and build new ops
  out of the tensors
  [`op.inputs[i]`](../../api_docs/python/framework.md#Operation.inputs),
  [`op.outputs[i]`](../../api_docs/python/framework.md#Operation.outputs), and `grad`.  Information
  about any attrs can be found via
  [`op.get_attr`](../../api_docs/python/framework.md#Operation.get_attr).

* If the op has multiple outputs, the gradient function will take `op` and
  `grads`, where `grads` is a list of gradients with respect to each output.
  The result of the gradient function must be a list of `Tensor` objects
  representing the gradients with respect to each input.

* If there is no well-defined gradient for some input, such as for integer
  inputs used as indices, the corresponding returned gradient should be
  `None`.  For example, for an op taking a floating point tensor `x` and an
  integer index `i`, the gradient function would `return [x_grad, None]`.

* If there is no meaningful gradient for the op at all, use
  `ops.NoGradient("OpName")` to disable automatic differentiation.

Note that at the time the gradient function is called, only the data flow graph
of ops is available, not the tensor data itself.  Thus, all computation must be
performed using other tensorflow ops, to be run at graph execution time.

## Implement a shape function in Python

The TensorFlow Python API has a feature called "shape inference" that provides
information about the shapes of tensors without having to execute the
graph. Shape inference is supported by "shape functions" that are registered for
each op type, and perform two roles: asserting that the shapes of the inputs are
compatible, and specifying the shapes for the outputs. A shape function is a
Python function that takes an
[`Operation`](../../api_docs/python/framework.md#Operation) as input, and
returns a list of
[`TensorShape`](../../api_docs/python/framework.md#TensorShape) objects (one per
output of the op). To register a shape function, apply the
[`tf.RegisterShape` decorator](../../api_docs/python/framework.md#RegisterShape)
to a shape function. For example, the
[`ZeroOut` op defined above](#define-the-ops-interface) would have a shape function like
the following:

```python
@tf.RegisterShape("ZeroOut")
def _zero_out_shape(op):
  """Shape function for the ZeroOut op.

  This is the unconstrained version of ZeroOut, which produces an output
  with the same shape as its input.
  """
  return [op.inputs[0].get_shape()]
```

A shape function can also constrain the shape of an input. For the version of
[`ZeroOut` with a vector shape constraint](#validation), the shape function
would be as follows:

```python
@tf.RegisterShape("ZeroOut")
def _zero_out_shape(op):
  """Shape function for the ZeroOut op.

  This is the constrained version of ZeroOut, which requires the input to
  have rank 1 (a vector).
  """
  input_shape = op.inputs[0].get_shape().with_rank(1)
  return [input_shape]
```

If your op is [polymorphic with multiple inputs](#polymorphism), use the
properties of the operation to determine the number of shapes to check:

```
@tf.RegisterShape("IntListInputExample")
def _int_list_input_example_shape(op):
  """Shape function for the "IntListInputExample" op.

  All inputs and the output are matrices of the same size.
  """
  output_shape = tf.TensorShape(None)
  for input in op.inputs:
    output_shape = output_shape.merge_with(input.get_shape().with_rank(2))
  return [output_shape]
```

Since shape inference is an optional feature, and the shapes of tensors may vary
dynamically, shape functions must be robust to incomplete shape information for
any of the inputs. The [`merge_with`](../../api_docs/python/framework.md)
method allows the caller to assert that two shapes are the same, even if either
or both of them do not have complete information. Shape functions are defined
for all of the
[standard Python ops](https://www.tensorflow.org/code/tensorflow/python/ops/),
and provide many different usage examples.

[core-array_ops]:https://www.tensorflow.org/code/tensorflow/core/ops/array_ops.cc
[python-user_ops]:https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py
[tf-kernels]:https://www.tensorflow.org/code/tensorflow/core/kernels/
[user_ops]:https://www.tensorflow.org/code/tensorflow/core/user_ops/
[pad_op]:https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc
[standard_ops-py]:https://www.tensorflow.org/code/tensorflow/python/ops/standard_ops.py
[standard_ops-cc]:https://www.tensorflow.org/code/tensorflow/cc/ops/standard_ops.h
[python-BUILD]:https://www.tensorflow.org/code/tensorflow/python/BUILD
[validation-macros]:https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h
[op_def_builder]:https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h
[register_types]:https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h
[FinalizeAttr]:https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc#FinalizeAttr
[DataTypeString]:https://www.tensorflow.org/code/tensorflow/core/framework/types.cc#DataTypeString
[python-BUILD]:https://www.tensorflow.org/code/tensorflow/python/BUILD
[types-proto]:https://www.tensorflow.org/code/tensorflow/core/framework/types.proto
[TensorShapeProto]:https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto
[TensorProto]:https://www.tensorflow.org/code/tensorflow/core/framework/tensor.proto
