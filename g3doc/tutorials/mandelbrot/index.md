# 만델브로트 집합

[만델브로트 집합](https://en.wikipedia.org/wiki/Mandelbrot_set)(Mandelbrot set)을 시각화하는 것은 머신 러닝과는 별 상관이 없지만, TensorFlow를 일반적인 수학에 사용하는 방법에 대한 재미있는 예시로 활용할 수 있습니다. 이 문서에서 소개되는 시각화 과정은 비교적 단순한(naive) 구현 방법을 사용했지만, 요점을 잘 보여줍니다. (더 아름다운 시각화를 위해, 더 정교한 구현 방법을 사용할 수도 있을 것입니다.)

참고: 이 튜토리얼은 IPython notebook 에서의 구현을 바탕으로 작성되었습니다.

## 기본 설정

시작하기 전, 몇 개의 라이브러리를 import 해야 합니다.

```python
# 시뮬레이션을 위한 라이브러리 import
import tensorflow as tf
import numpy as np

# 시각화를 위한 라이브러리 import
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
```

이제 우리는 반복 횟수가 주어졌을 때 그림을 그리기 위한 함수를 정의합니다.

```python
def DisplayFractal(a, fmt='jpeg'):
  """반복 횟수들의 배열을 다채로운 프랙탈 이미지로 나타냅니다."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))
```

## 세션 및 변수 초기화

이러한 작업을 위해서는 주로 인터랙티브 세션(interactive session)을 사용합니다. (일반적인 세션도 괜찮습니다.)

```python
   sess = tf.InteractiveSession()
```

NumPy와 TensorFlow를 자유롭게 연결해 쓸 수 있다는 것은 편리합니다.

```python
# Numpy를 이용하여 [-2,2]x[-2,2]의 복소수에 대한 2차원 배열 생성

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y
```
이제 TensorFlow 텐서들을 정의하고 초기화합니다.

```python
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))
```

TensorFlow의 변수는 사용하기 전에 명시적으로 초기화해야 합니다.

```python
tf.initialize_all_variables().run()
```

## 정의 및 계산 실행

계산 과정에 대한 추가적인 부분을 더 작성하고...

```python
# 새로운 z값의 계산: z^2 + x
zs_ = zs*zs + xs

# 새로운 z값은 발산(diverge)했을까?
not_diverged = tf.complex_abs(zs_) < 4

# z값 및 반복 횟수 업데이트
#
# 참고: 이 코드는 zs가 발산한 뒤에도 계속 계산을 진행합니다.
#       이것은 낭비이며, 약간 더 복잡하더라도 더 좋은 방법이
#       있을 것입니다.
#
step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, tf.float32))
  )
```

... 이것을 200번 정도 실행합니다.

```python
for i in range(200): step.run()
```

결과를 보도록 하죠.

```python
DisplayFractal(ns.eval())
```

![jpeg](../../images/mandelbrot_output.jpg)

나쁘지 않군요!


