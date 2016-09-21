# 편미분 방정식

텐서플로우는 머신러닝만을 위한 것이 아닙니다.  [편미분](
https://en.wikipedia.org/wiki/Partial_differential_equation)을 시뮬레이션 하기
위해 텐서플로우를 사용한 (약간 지루한) 예제가 있습니다. 빗방울이 떨어지는
사각형 연못의 표면을 시뮬레이션 해보겠습니다.

Note: 이 튜토리얼은 IPython notebook 으로 작성되었습니다.

## 기본 설정

몇 가지를 임포트 해야합니다.

```python
#시뮬레이션을 위한 라이브러리 임포트
import tensorflow as tf
import numpy as np

#보여주기위한 임포트
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
```

연못 표면의 상태를 이미지로 보여주는 함수.

```python
def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))
```

여기서 편의를 위해 인터랙티브 텐서플로우 세션을 시작합니다. 만약 .py 파일로
실행한다면 일반 세션도 잘 동작합니다.

```python
sess = tf.InteractiveSession()
```

## 편의를 위한 함수


```python
def make_kernel(a):
  """2차 배열을 콘볼루션 커널로 변환"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """간단한 2차 콘볼루션 연산"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """2차 배열의 라플라시안 계산"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)
```

## PDE 정의

자연에서 발견되는 대부분의 연못과 같이 우리의 연못은 500 x 500 의 정사각형
입니다.

```python
N = 500
```

연못을 만들고 빗방울을 떨어뜨려봅시다.

```python
# 조건 초기화 -- 몇개의 빗방울이 연못에 떨어뜨립니다

# 전부 0으로 설정
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# 몇개의 빗방울이 연못의 임의 위치에 떨어뜨립니다
for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

DisplayArray(u_init, rng=[-0.1, 0.1])
```

![jpeg](../../images/pde_output_1.jpg)


이제 미분을 적용해봅시다.


```python
# 매개변수:
# eps -- 시간 해상도
# damping -- 파고감쇠
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# 시뮬레이션 상태를 위한 변수 생성
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# 이산화된 PDE 갱신 규칙
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# 상태 갱신 명령
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))
```

## 시뮬레이션 실행

이제 흥미로운 부분입니다 -- 간단한 for 반복문으로 실시간 진행합니다.

```python
# 초기 조건에 대한 상태 초기화
tf.initialize_all_variables().run()

# 1000 번의 PDE 수행
for i in range(1000):
  # 단계 시뮬레이션
  step.run({eps: 0.03, damping: 0.04})
  DisplayArray(U.eval(), rng=[-0.1, 0.1])
```

![jpeg](../../images/pde_output_2.jpg)

보세요! 물결을!

