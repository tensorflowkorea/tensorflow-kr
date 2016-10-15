# Writing TensorFlow Documentation

TensorFlow의 문서는 
[Markdown](https://daringfireball.net/projects/markdown/)에서 유지되고 있고 
`g3doc/` 디렉토리에 존재합니다. *Introduction*, *Overview*, *Tutorials*, *How-Tos*
부분은 수작업으로 수정됩니다.

`g3doc/api_docs` 디렉토리에 있는 어떤 것이든 코드에 있는 주석으로부터 생성되기 때문에
직접 수정해선 안됩니다. `tools/docs/gen_docs.sh` 스크립트는
API문서를 생성합니다. 스크립트를 인자 없이 호출하면 Python API 문서만을 재작성합니다. 
(Ops에 관한 문서는 Python, C++ 상관없이 작성됩니다.). `-a` 를 인자로 전달하면 
C++ 문서 또한 재작성합니다. 이것은 반드시 `tools/docs` 디렉토리에서 호출되어야 합니다. 
그리고 `-a` 를 인자로 전달하려면 `doxygen` 의 설치가 요구됩니다.

## Python API Documentation

Ops, classes, utility 함수는 `image_ops.py` 와 같은 Python 모듈에 정의되어 있습니다. 
그 모듈의 docstring은 해당 Python 파일에 대해 생성되는 마크다운 파일의 시작부분에 삽입됩니다. 
그래서 `image_ops.md` 는 `image_ops.py` 모듈에 있는 docstring으로 시작합니다. 
`python/framework/gen_docs_combined.py` 는 마크다운 파일이 생성되는 모든 _libraries_(_라이브러리_) 목록을 포함하고 있습니다. 
만약에 새로운 라이브러리를 더한다면 (API 문서에 독립된 섹션을 생성하는 것), 
반드시 `gen_docs_combined.py` 에 있는 라이브러리 목록에 추가해야 합니다. 
C++ API는 하나의 라이브러리 파일만 존재합니다, 그것의 마크다운은 `api_docs/cc/index.md` 에서 생성된 `gen_cc_md.py` 의 문자열입니다. 
나머지 C++문서들은 doxygen에 의해 만들어진 XML파일로부터 생성됩니다.

라이브러리로 등록된 파일의 모듈 docstring에서, (빈 줄의 시작 부분에) `@@<python-name>` 문법으로 
Ops, classes, functions를 호출해 그것들을 위해 생성된 문서를 삽입할 수 있습니다. 
호출한 op, function 또는 class는 같은 파일에서 정의될 필요가 없습니다.
이것은 Ops, classes, functions가 기록되는 순서를 정할 수 있게 해줍니다.
고수준의 문서를 적절히 배치해서 논리적인 순서로 나누세요. 

모든 공개된 op, class, function 은 반드시 라이브러리의 서두에서 `@@` 로 호출되어야 합니다. 
그렇게 하지 않을 경우 `doc_gen_test` 가 실패하게 됩니다.

Ops를 위한 문서는 자동적으로 Python wrapper 또는 C++ Ops registrations로 부터 발췌합니다. 
둘 중 Python wrappers를 우선적으로 가져옵니다.

* Python wrappers는 `python/ops/*.py` 에 있습니다.
* C++ Ops registrations는 `core/ops/*.cc` 에 있습니다.

Classes와 Utility Functions를 위한 문서는 docstring에서 발췌합니다.

## Op Documentation Style Guide

이상적으로 제시된 순서를 따라 아래와 같은 정보를 제공해야 합니다:

* op가 무엇을 하는지를 설명하는 짧은 문장.
* op에 인자를 전달할 때 어떤 일이 생기는지에 대한 짧은 설명.
* op가 어떻게 작동하는지를 보여주는 예시(수도코드가 가장 좋음).
* 요구사항, 경고, 중요한 내용 (하나라도 있다면).
* 입력, 출력, 속성, op 생성자의 다른 변수에 대한 설명.

위 항목에 대해 설명된 자세한 정보는
[이곳](#description-of-the-docstring-sections).

글을 마크다운 (.md) 포멧으로 적으세요. 기본적인 문법에 관한 레퍼런스는
[여기](https://daringfireball.net/projects/markdown/)에 있습니다. 방정식 표기를 위해 
[MathJax](https://www.mathjax.org)을 사용할 수 있습니다. 이것들은 
[tensorflow.org](https://www.tensorflow.org)에서 적절히 표현됩니다. 하지만
[github](https://github.com/tensorflow/tensorflow)에 나타나지는 않습니다.

### Writing About Code

Put backticks around these things when they're used in text:

- Argument names (e.g. `input`, `x`, `tensor`)
- Returned tensor names (e.g. `output`, `idx`, `out`)
- Data types (e.g. `int32`, `float`, `uint8`)
- Other op names referenced in text (e.g. `list_diff()`, `shuffle()`)
- Class names (e.g. `Tensor` when you actually mean a `Tensor` object; don't
  capitalize or use backticks if you're just explaining what an op does to a
  tensor, or a graph, or an operation in general)
- File names (e.g. `image_ops.py`, or `/path-to-your-data/xml/example-name`)

Put three backticks around sample code and pseudocode examples. And use `==>`
instead of a single equal sign when you want to show what an op returns. For
example:

    ```
    # 'input' is a tensor of shape [2, 3, 5]
    (tf.expand_dims(input, 0)) ==> [1, 2, 3, 5]
    ```

If you're providing a Python code sample, add the python style label to ensure proper syntax highlighting:

```markdown
 ```python
 # some Python code
 ```
```

Put single backticks around math expressions or conditions. For example:

```markdown
This operation requires that `-1-input.dims() <= dim <= input.dims()`.
```

### Tensor Dimensions

When you're talking about a tensor in general, don't capitalize the word tensor.
When you're talking about the specific object that's provided to an op as an
argument or returned by an op, then you should capitalize the word Tensor and
add backticks around it because you're talking about a `Tensor` object that gets
passed.

Don't use the word `Tensors` to describe multiple Tensor objects unless you
really are talking about a `Tensors` object. Better to say "a list of `Tensor`
objects.", or, maybe, "`Tensor`s".

When you're talking about the size of a tensor, use these guidelines:

Use the term "dimension" to refer to the size of a tensor. If you need to be
specific about the size, use these conventions:

- Refer to a scalar as a "0-D tensor"
- Refer to a vector as a "1-D tensor"
- Refer to a matrix as a "2-D tensor"
- Refer to tensors with 3 or more dimensions as 3-D tensors or n-D tensors. Use
  the word "rank" only if it makes sense, but try to use "dimension" instead.
  Never use the word "order" to describe the size of a tensor.

Use the word "shape" to describe in detail the dimensions of a tensor, and show
the shape in square brackets with backticks. For example:

```markdown
If `input` is a 3-D tensor with shape `[3, 4, 3]`, this operation will return
a 3-D tensor with shape `[6, 8, 6]`.
```

### Links

To link to something else in the `g3docs` tree, use a relative path, like
`[tf.parse_example](../api_docs/python/ops.md#parse_example)`
Do not use absolute paths for internal links, as this will break the website
generator.

To link to source code, use a link starting with:
`https://www.tensorflow.org/code/`, followed by
the file name starting at the github root. For instance, a link to this file
should be written as
`https://www.tensorflow.org/code/tensorflow/g3doc/how_tos/documentation/index.md`.
This ensures that [tensorflow.org](https://www.tensorflow.org/) can forward the link to the
branch of the code corresponding to the version of the documentation you're
viewing. Do not include url parameters in the URL.


### Ops defined in C++

All Ops defined in C++ must be documented as part of the `REGISTER_OP`
declaration. The docstring in the C++ file is processed to automatically add
some information for the input types, output types, and Attr types and default
values.

예시:

```c++
REGISTER_OP("PngDecode")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: uint8")
    .Doc(R"doc(
Decodes the contents of a PNG file into a uint8 tensor.

contents: PNG file contents.
channels: Number of color channels, or 0 to autodetect based on the input.
  Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.
  If the input has a different number of channels, it will be transformed
  accordingly.
image:= A 3-D uint8 tensor of shape `[height, width, channels]`.
  If `channels` is 0, the last dimension is determined
  from the png contents.
)doc");
```

Results in this piece of Markdown:

```markdown
### tf.image.png_decode(contents, channels=None, name=None) {#png_decode}

Decodes the contents of a PNG file into a uint8 tensor.

#### Args:

*  <b>contents</b>: A string Tensor. PNG file contents.
*  <b>channels</b>: An optional int. Defaults to 0.
    Number of color channels, or 0 to autodetect based on the input.
    Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.  If the
    input has a different number of channels, it will be transformed accordingly.
*  <b>name</b>: A name for the operation (optional).

#### Returns:

  A 3-D uint8 tensor of shape `[height, width, channels]`.
  If `channels` is 0, the last dimension is determined
  from the png contents.
```

Much of the argument description is added automatically. In particular, the doc
generator automatically adds the name and type of all inputs, attrs, and
outputs. In the above example, `<b>contents</b>: A string Tensor.` was added
automatically. You should write your additional text to flow naturally after
that description.

For inputs and output, you can prefix your additional text with an equal sign to
prevent the automatically added name and type. In the above example, the
description for the output named `image` starts with `=` to prevent the addition
of `A uint8 Tensor.` before our text `A 3-D uint8 Tensor...`. You cannot prevent
the addition of the name, type, and default value of attrs this way, so write
your text carefully.

### Ops defined in Python

If your op is defined in a `python/ops/*.py` file, then you need to provide
text for all of the arguments and output (returned) tensors.

You should conform to the usual Python docstring conventions, except that you
should use Markdown in the docstring. The doc generator does not auto-generate
any text for ops that are defined in Python, so what you write is what you get.

간단한 예시:

```python
def foo(x, y, name="bar"):
  """Computes foo.

  Given two 1-D tensors `x` and `y`, this operation computes the foo.

  For example:

  ```
  # x is [1, 1]
  # y is [2, 2]
  tf.foo(x, y) ==> [3, 3]
  ```

  Args:
    x: A `Tensor` of type `int32`.
    y: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32` that is the foo of `x` and `y`.

  Raises:
    ValueError: If `x` or `y` are not of type `int32`.
  """

  ...
```

## Description of the Docstring Sections

여기에 더 자세한 내용과 docstring의 각 속성에 대한 예시가 있습니다.

### Short sentence that describes what the op does.

예시:

```markdown
Concatenates tensors.
```

```markdown
Flips an image horizontally from left to right.
```

```markdown
Computes the Levenshtein distance between two sequences.
```

```markdown
Saves a list of tensors to a file.
```

```markdown
Extracts a slice from a tensor.
```

### Short description of what happens when you pass arguments to the op.

예시:

```markdown
Given a tensor input of numerical type, this operation returns a tensor of
the same type and size with values reversed along dimension `seq_dim`. A
vector `seq_lengths` determines which elements are reversed for each index
within dimension 0 (usually the batch dimension).
```

```markdown
This operation returns a tensor of type `dtype` and dimensions `shape`, with
all elements set to zero.
```

### Example showing how the op works.

`squeeze()` op 에 좋은 수도코드 예시가 있습니다:

    shape(input) => `[1, 2, 1, 3, 1, 1]`
    shape(squeeze(input)) =>  `[2, 3]`

`tile()` op 는 좋은 설명문의 예시를 제공합니다:

    For example, tiling `[a, b, c, d]` by 2 produces
    `[[a, b, c, d], [a, b, c, d]]`.

It is often helpful to show code samples in Python. Never put them in the C++
Ops file, and avoid putting them in the Python Ops doc. Put them in the module
or class docstring where the Ops constructors are called out.

Here's an example from the module docsting in `image_ops.py`:

    Tensorflow can convert between images in RGB or HSV. The conversion
    functions work only on `float` images, so you need to convert images in
    other formats using [`convert_image_dtype`](#convert-image-dtype).

    Example:

    ```python
    # Decode an image and convert it to HSV.
    rgb_image = tf.image.decode_png(...,  channels=3)
    rgb_image_float = tf.image.convert_image_dtype(rgb_image, tf.float32)
    hsv_image = tf.image.rgb_to_hsv(rgb_image)
    ```

### Requirements, caveats, important notes.

예시:

```markdown
This operation requires that: `-1-input.dims() <= dim <= input.dims()`
```

```
Note: This tensor will produce an error if evaluated. Its value must
be fed using the `feed_dict` optional argument to `Session.run()`,
`Tensor.eval()`, or `Operation.run()`.
```

### Descriptions of arguments and output (returned) tensors.

Keep the descriptions brief and to the point. You should not have to explain
how the operation works in the argument sections.시

Mention if the Op has strong constraints on the dimensions of the input or
output tensors. Remember that for C++ Ops, the type of the tensor is
automatically added as either as "A ..type.. Tensor" or "A Tensor with type
in {...list of types...}". In such cases, if the Op has a constraint on the
dimensions either add text such as "Must be 4-D" or start the description with
`=` (to prevent the tensor type to be added) and write something like
"A 4-D float tensor".

For example, here are two ways to document an image argument of a C++ op (note
the "=" sign):

```markdown
image: Must be 4-D. The image to resize.
```

```markdown
image:= A 4-D `float` tensor. The image to resize.
```

In the documentation, these will be rendered to markdown as

```markdown
image: A `float` Tensor. Must be 4-D. The image to resize.
```

```markdown
image: A 4-D `float` Tensor. The image to resize.
```

### Optional arguments descriptions ("attrs")

The doc generator always describe attrs type and default value, if any.
You cannot override that with an equal sign because the description is very
different in the C++ and Python generated docs.

Phrase any additional attr description so that it flows well after the type
and default value.

`image_ops.py` 에 있는 예시입니다:

```c++
REGISTER_OP("PngDecode")
    .Input("contents: string")
    .Attr("channels: int = 0")
    .Output("image: uint8")
    .Doc(R"doc(
Decode a PNG-encoded image to a uint8 tensor.

The attr `channels` indicates the desired number of color channels for the
decoded image.

Accepted values are:

*   0: Use the number of channels in the PNG-encoded image.
*   1: output a grayscale image.

...

contents: 0-D. The PNG-encoded image.
channels: Number of color channels for the decoded image.
image: 3-D with shape `[height, width, channels]`.
)doc");
```

이것은 아래와 같은 "Args" 부분을 만들어 냅니다:

```markdown
  contents: A string Tensor. 0-D. The PNG-encoded image.
  channels: An optional `int`. Defaults to 0. Number of color channels for the
    decoded image.
  name: A name for the operation (optional).
```




