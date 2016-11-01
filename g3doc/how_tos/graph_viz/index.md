# TensorBoard: 그래프 시각화

TensorFlow computation 그래프는 강력하지만 복잡하다. 그래프 시각화가 이해와 디버그를 도와줄 수 있다. 여기에 시각화 작동 예시가 있습니다.

![TensorFlow 그래프의 시각화](../../images/graph_vis_animation.gif "TensorFlow 그래프의 시각화")
*TensorFlow 그래프의 시각화.*

자신의 그래프를 보기 위해, 작업의 로그 디렉토리를 대상으로 실행하고 위쪽 판에 있는 그래프 탭을 클릭하고 위 외쪽 모서리에 있는 메뉴를 사용해 적당한 작업을 선택하십시요. TensorBoard를 어떻게 실행하는지에 대한 심화된 정보와 필요한 정보를 기록하고 있는지 확인하려면 [TensorBoard: 시각화 학습](../../how_tos/summaries_and_tensorboard/index.md)를 보시면 됩니다.

[여기](https://www.tensorflow.org/tensorboard/index.html#graphs)를 클릭해서 그래프 시각화를 포함해 [MNIST](../../tutorials/mnist/beginners/index.md) 트레이닝 세션 데이터를 보고 TensorBoard 인스턴스와 상호작용 할 수 있습니다.

## Name scoping and nodes

일반적으로 TensorFlow 그래프는 수천의 많은 노드를 가질 수 있고 한 눈에 쉽게 보거나 
표준 그래프 툴을 이용해 배치하는 것 까지 할 수 있습니다.
간단히, 변수 이름은 범위화 되어 있고  시각화를 이 정보를 이용해 그래프에 있는 노드들을 계층 형태로 정의합니다. 
기본적으로 이 계층 형태의 최상위만 볼 수 있습니다. 여기에 
[`tf.name_scope`](../../api_docs/python/framework.md#name_scope)를 사용해 
`hidden` name scope 아래에 세 가지 기능을 정의한 예시가 있습니다:

```python
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
```

이것은 아래와 같은 세가지 op 이름을 만듭니다:

* *hidden*/alpha
* *hidden*/weights
* *hidden*/biases

기본적으로 시각화는 이 세 가지를 `hidden` 노드에 접어놓습니다.
추가적인 디테일은 유지됩니다. 노드를 펼치기 위해 오측 상단에 있는 주황색 `+` 표시를 클릭하거나 노드 자체를 더블 클릭하면 
세 개의 `alpha`, `weights`, `biases` 서브노드를 볼 수 있습니다.

여기에 좀 더 복잡한 노드의 초기 상태와 펼쳐진 상태를 보여주는 실제 예제가 있습니다.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/pool1_collapsed.png" alt="Unexpanded name scope" title="Unexpanded name scope" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/pool1_expanded.png" alt="Expanded name scope" title="Expanded name scope" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      최상단 name scope <code>pool_1</code>의 초기 화면. 우측 상단의 오렌지색 <code>+</code> 버튼을 클릭하거나 노드를 더블 클릭하면 펼칠 수 있다.
    </td>
    <td style="width: 50%;">
      <code>pool_1</code> name scope가 펼치진 모습. 우측 상단의 오렌지색 <code>-</code> 버튼을 클릭하거나 노드를 더블 클릭하면 name scope를 접을 수 있다.
    </td>
  </tr>
</table>

name scope로 노드를 묶는 것은 읽기 쉬운 그래프를 만드는데 중요하다.
만약 당신이 모델을 만들고 있다면, name scope가 시각화된 결과를 통제할 수 있게 해 줄 것이다.
**name scope를 잘 사용할수록 시각화가 잘 된다.**

위 그림은 시각화의 두 번째 면을 보여줍니다. TensorFlow 그래프는 두 가지 종류의 연결을 가지고 있습니다: 
데이터 종속과 컨트롤 종속. 데이터 종속은 두 ops 사이 tensor의 흐름을 보여주고 
화살표의 실선으로 나타냅니다. 반면 컨트롤 종속은 점선을 사용합니다. 
펼쳐진 모습에서(위 그림의 오른쪽 부분) `CheckNumerics`와 `control_dependency`를 연결하는 
점선을 제외하고 모든 연결은 데이터 종속입니다.

레이아웃을 간단하게 하는 두 번째 트릭이 있습니다. 대부분의 TensorFlow 그래프는 
다른 노드와 많이 연결된 몇 개의 노드로 이루어져 있습니다. 예를 들어, 많은 노드는 
초기화 단계에서 컨트롤 종속이 있을 수 있습니다. `init` 노드와 그 노드에 종속된 것들 사이의에
모든 간선을 그리는 것은 뭉쳐진 모습을 생성하게 됩니다.

뭉침(clustter)을 줄이기 위해, 시각화는 상위에 있는 노드를 분리해서 
우측에 있는 *auxiliary(보조)* 공간에 두고 간선을 그리지 않습니다. 
선 대신에 연결을 나타내기 위해 작은 *노드 아이콘*을 그립니다. 
보조 노드를 떼어놓는 것은 일반적으로 중요한 정보를 없애지 않습니다. 왜냐하면 
이 노드들은 보통 부기 기능(bookkeeping function)과 관련있기 때문입니다. 
메인 그래프과 보조 영역 사이에서 노드를 어떻게 움직이는지 보려면 [Interaction](#interaction)를 보세요.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/conv_1.png" alt="conv_1 is part of the main graph" title="conv_1 is part of the main graph" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/save.png" alt="save is extracted as auxiliary node" title="save is extracted as auxiliary node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      <code>conv_1</code> 노드는 <code>save</code> 와 연결되어 있습니다. 우측에 작은 <code>save</code> 노드 아이콘이 있는 것을 주의하세요.
    </td>
    <td style="width: 50%;">
      <code>save</code> 는 상위에 있고 보조 노드로 표시됩니다. <code>conv_1</code> 과의 연결은 그것의 왼쪽에 있는 노드 아이콘으로 표현됩니다. <code>save</code>가 많은 연결을 가지고 있기 때문에, 클러스터를 더 줄이기 위해, 처음 5개만 보여주고 나머지는 <code>... 12 more</code> 로 축약합니다.
    </td>
  </tr>
</table>

마지막 구조 단순화는 *series collapsing* 입니다. 이름의 마지막 숫자만 
다르고 같은 구조를 가진 노드, 순차적 모티프는 아래에 나와 있는 것 처럼 
하나의 노드 스택으로 접을 수 있습니다. 긴 배열을 가진 네트워크의 경우, 이것이 
모양을 굉장히 단순하게 합니다. 계층 노드와 마찬가지로, 더블 클릭으로 이 series를 펼칠 수 있습니다. 
어떻게 특정 노드 셋이 접힌 것을 비활성화/활성화 하는지는 [Interaction](#interaction)를 보세요.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/series.png" alt="Sequence of nodes" title="Sequence of nodes" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/series_expanded.png" alt="Expanded sequence of nodes" title="Expanded sequence of nodes" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      노드 시퀸스가 접힌 모습.
    </td>
    <td style="width: 50%;">
      더블 클릭 후에 펼쳐진 모습의 일부분.
    </td>
  </tr>
</table>

마지막으로, 가독성을 높이기 위해, 시각화는 상수와 요약 노드를 위해 특별한 아이콘을 사용합니다. 
간단히, 노드 기호 표가 있습니다:

기호 | 의미
--- | ---
![Name scope](../../images/namespace_node.png "Name scope") | *High-level* 노드는 name scope를 나타냅니다. high-level 노드를 펼치기 위해 더블 클릭 하세요.
![Sequence of unconnected nodes](../../images/horizontal_stack.png "Sequence of unconnected nodes") | 서로 연결되지 않은 숫자가 메겨진 노드의 시퀸스.
![Sequence of connected nodes](../../images/vertical_stack.png "Sequence of connected nodes") | 서로 연결된 숫자가 메겨진 노드의 시퀸스.
![Operation node](../../images/op_node.png "Operation node") | 각각의 연산 노드.
![Constant node](../../images/constant.png "Constant node") | 상수.
![Summary node](../../images/summary.png "Summary node") | 요약 노드.
![Data flow edge](../../images/dataflow_edge.png "Data flow edge") | 간선은 연산 사이의 데이터 흐름을 보여줍니다.
![Control dependency edge](../../images/control_edge.png "Control dependency edge") | 간선은 연산 사이의 컨트롤 종속을 보여줍니다.
![Reference edge](../../images/reference_edge.png "Reference edge") | 레퍼런스 간선은 나가는 연산 노드가 들어오는 tensor를 변형할 수 있다는 것을 보여줍니다.

## Interaction {#interaction}

줌과 이동으로 그래프를 탐색할 수 있습니다. 화면을 클릭하고 드래그하고 줌하기 위해 
스크롤 제스쳐를 사용하세요. 연산 그룹을 보여주는 name scope를 펼치기 위해 
노드를 더블클릭 하거나 노드의 `+` 버튼을 누르세요. 줌과 이동을 했을 때 현재 화면을 
쉽게 쫓기 위해 우측 하단 모서리에 미니맵이 있습니다.

열린 노드를 닫기 위해 다시 한번 더블 클릭 하거나 노드의 `-` 버튼을 누르세요. 
노드를 선택하기 위해 한번 누르셔도 됩니다. 노드는 어두운 색으로 변하고, 노드에 대한 세부 사항과 
연결된 노드가 시각화의 우측 상단 모서리 정보 카드에 표시됩니다.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/infocard.png" alt="Info card of a name scope" title="Info card of a name scope" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/infocard_op.png" alt="Info card of operation node" title="Info card of operation node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      정보 카드가 <code>conv2</code> name scope의 세부 정보를 보여줍니다. 입력과 출력은 name scope 내의 연산 노드의 입력과 출력에서 결합됩니다. name scope에 대한 속성은 보여지지 않습니다.
    </td>
    <td style="width: 50%;">
      정보 카드가 <code>DecodeRaw</code> 연산 노드의 세부 정보를 보여줍니다. 입력과 출력에 더해서, 카드는 현재 연산에 관련된 디바이스와 속성들을 보여줍니다.
    </td>
  </tr>
</table>

TensorBoard는 그래프의 레이아웃을 바꿀 수 있는 몇 가지 방법을 제공합니다. 
이것이 그래프의 연산 의미를 바꾸지는 않지만 네트워크 구조를 좀 더 명확하게 합니다. 
노드를 우클릭하거나 정보 카드 하단에 있는 버튼을 눌러서 레이아웃에 아래와 같은 변화를 
줄 수 있습니다:

* 노드는 메인 그래프와 보조 영역 사이를 이동할 수 있습니다.
* 연속된(series) 노드를 그룹 해제해서 이 노드들이 그룹으로 함께 보여지지 않게 합니다. 
그룹 해제된 series는 같은 방법으로 재그룹화 할 수 있습니다.

셀렉션은 또한 상위 노드를 이해하는데 도움이 됩니다. 어떤 상위 노드를 선택하면 
다른 연결을 위해 대응하는 노드 아이콘도 또한 선택됩니다. 예를 들어, 이것은 어떤 노드가 
저장되고 어떤것이 안됐는지 보는 것을 쉽게 만들어 줍니다.

정보 카드에 있는 노드 이름을 클릭하면 해당 노드를 선택합니다. 필요하다면, 노드를 볼 수 있도록 
화면이 자동으로 이동합니다.

마지막으로, 상단 표에 있는 컬러 메뉴를 이용해 그래프에 두 가지 색을 선택할 수 있습니다. 
기본 *Structure View* 는 구조를 보여줍니다: 두 상위 레벨의 노드가 같은 구조를 가졌을 때, 같은 색으로 표현됩니다. 
유일한 구조를 가진 노드는 회색입니다. 어떤 디바이스가 다른 연산을 실행하고 있는지 보여주는 두 번째 화면이 있습니다. 
Name scope는 내부 연산에 대한 장치의 비율에 비례하여 색이 지정됩니다.

아래의 이미지는 실시간 그래프 일부분의 모습을 제공합니다.

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="../../images/colorby_structure.png" alt="Color by structure" title="Color by structure" />
    </td>
    <td style="width: 50%;">
      <img src="../../images/colorby_device.png" alt="Color by device" title="Color by device" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      Structure view: 회색 노드는 유일한 구조를 가지고 있습니다. 주황색 <code>conv1</code> 과 <code>conv2</code> 노드는 같은 구조를 가지고 있고 다른 색을 가진 노드와 비슷합니다.
    </td>
    <td style="width: 50%;">
      Device view: Name scope는 내부 연산 노드의 장치 비율에 비례해서 색이 지정됩니다. 여기서는 보라색이 GPU를 의미하고 초록색이 CPU를 의미합니다.
    </td>
  </tr>
</table>

## Tensor shape information

연속된 `GraphDef` 가 tensor 형태를 포함할 때, 그래프 시각화는 tensor dimension으로 간선에 이름을 붙이고 
간선의 두께는 총 tensor 크기를 반영합니다. `GraphDef` 에 tensor 형태를 포함하기 위해 
그래프를 연속화할 때 실제 그래프 객체(`sess.graph`와 같이)를 `SummaryWriter` 에 전달합니다.
아래의 이미지는 tensor 형태 정보를 가진 CIFAR-10 모델을 보여줍니다:

<table width="100%;">
  <tr>
    <td style="width: 100%;">
      <img src="../../images/tensor_shapes.png" alt="CIFAR-10 model with tensor shape information" title="CIFAR-10 model with tensor shape information" />
    </td>
  </tr>
  <tr>
    <td style="width: 100%;">
      tensor 형태 정보를 가진 IFAR-10 모델.
    </td>
  </tr>
</table>

## Runtime statistics

보통 실행 상에서 총 메모리 사용량, 총 계산 시간, 노드의 tensor 형태와 같은 
런타임 메타데이터를 수집하는 것이 도움이 됩니다. 아래의 코드 예제는 
[simple MNIST tutorial](http://tensorflow.org/tutorials/mnist/beginners/index.md)를 변형해 훈련과 테스트 섹션에서 발췌한 
요약과 런타임 통계를 기록하는 짧은 단락이다. 
어떻게 요약을 기록하는지 세부 세항은 [Summaries Tutorial](../../how_tos/summaries_and_tensorboard/index.md#serializing-the-data)을 보세요. 
전체 소스는 [여기](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py).

```python
  # 모델을 트레이닝하고 또한 요약을 작성한다.
  # 매 10번째 순서 마다, 테스트 셋의 정확도를 측정하고 테스트 요약을 작성한다.
  # 다른 모든 순서에서 트레이닝 데이터로 트레이닝 단계를 실행하고 트레이닝 요약을 작성한다.

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # 요약과 테스트 셋 정확도를 기록한다
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # 트레이닝 셋 요약을 기록하고 트레이닝한다 
      if i % 100 == 99:  # 실행 통계를 기록한다
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # 요약을 기록한다
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
```

This code will emit runtime statistics for every 100th step starting at step99.

When you launch tensorboard and go to the Graph tab, you will now see options
under "Session runs" which correspond to the steps where run metadata was added.
Selecting one of these runs will show you the snapshot of the network at that
step, fading out unused nodes. In the controls on the left hand side, you will
be able to color the nodes by total memory or total compute time. Additionally,
clicking on a node will display the exact total memory, compute time, and
tensor output sizes.


<table width="100%;">
  <tr style="height: 380px">
    <td>
      <img src="../../images/colorby_compute_time.png" alt="Color by compute time" title="Color by compute time"/>
    </td>
    <td>
      <img src="../../images/run_metadata_graph.png" alt="Run metadata graph" title="Run metadata graph" />
    </td>
    <td>
      <img src="../../images/run_metadata_infocard.png" alt="Run metadata info card" title="Run metadata info card" />
    </td>
  </tr>
</table>
