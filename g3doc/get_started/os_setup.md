# 다운로드와 셋업

텐서플로우는 바이너리 패키지나 깃허브 소스를 이용해 설치할 수 있습니다.

## 필요사항

텐서플로우 파이썬 API는 파이썬 2.7과 파이썬 3.3+을 지원합니다.

GPU 버전(아직 리눅스용만 있습니다)은 Cuda Toolkit 7.5 과
cuDNN v4 와 가장 잘 작동합니다. 소스를 이용해 설치하면 다른 버전(Cuda toolkit >= 7.0 과
cuDNN 6.5(v2), 7.0(v3), v5)도 사용할 수 있습니다. 자세한 내용은 [Cuda 설치](#optional-install-cuda-gpus-on-linux) 부분을 참고해 주세요.

## 개요

여러가지 설치 방법을 지원하고 있습니다:

*  [Pip 설치](#pip-installation): 이 방식으로 텐서플로우를 설치하거나 업그레이드할 때는
   이 전에 작성했던 파이썬 프로그램에 영향을 미칠 수 있습니다.
*  [Virtualenv 설치](#virtualenv-installation): 텐서플로우를 각각의 디렉토리 안에
   설치하므로 다른 프로그램에 영향을 미치지 않습니다.
*  [아나콘다(Anaconda) 설치](#anaconda-installation): 텐서플로우를 각 아나콘다 환경에
   설치하므로 다른 프로그램에 영향을 미치지 않습니다.
*  [도커(Docker) 설치](#docker-installation): 텐서플로우를 도커 컨테이너에서 실행하므로
   컴퓨터의 다른 프로그램과 분리되어 운영됩니다.
*  [소스에서 설치](#installing-from-sources): 텐서플로우를 pip wheel을 이용하여
   빌드하고 설치합니다.

만약 Pip, Virtualenv, 아나콘다(Anaconda) 나 도커(Docker)를 잘 알고 있다면 필요에 맞게 설치 과정을 응용해도 좋습니다. pip 패키지 이름이나 도커 이미지 이름은 각 설치 섹션에 기재되어 있습니다.

설치시 에러가 발생하면 [자주 발생하는 문제](#common-problems)를 참고하세요.

<a id="pip-installation"></a>
## Pip 설치

[Pip](https://en.wikipedia.org/wiki/Pip_(package_manager)는 파이썬 패키지를 설치하고 관리하는 패키지 매니저 프로그램입니다.

설치되는 동안 추가되거나 업그레이드 될 파이썬 패키지 목록은 [setup.py 파일의 REQUIRED_PACKAGES 섹션](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)에 있습니다.

pip가 설치되어 있지 않다면 먼저 pip를 설치해야 합니다(python 3일 경우는 pip3):

```bash
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev

# Mac OS X
$ sudo easy_install pip
$ sudo easy_install --upgrade six
```

적절한 텐서플로우 바이너리를 선택합니다:

```bash
# Ubuntu/Linux 64-bit, CPU 전용, Python 2.7
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU 버전, Python 2.7
# CUDA toolkit 7.5 와 CuDNN v4 필수. 다른 버전을 사용하려면 아래 "소스에서 설치" 섹션을 참고하세요.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU 전용, Python 2.7
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py2-none-any.whl

# Ubuntu/Linux 64-bit, CPU 전용, Python 3.4
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU 버전, Python 3.4
# CUDA toolkit 7.5 와 CuDNN v4 필수. 다른 버전을 사용하려면 아래 "소스에서 설치" 섹션을 참고하세요.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, CPU 전용, Python 3.5
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp35-cp35m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU 버전, Python 3.5
# CUDA toolkit 7.5 와 CuDNN v4 필수. 다른 버전을 사용하려면 아래 "소스에서 설치" 섹션을 참고하세요.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp35-cp35m-linux_x86_64.whl

# Mac OS X, CPU 전용, Python 3.4 or 3.5:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py3-none-any.whl
```

텐서플로우를 설치합니다:

```bash
# Python 2
$ sudo pip install --upgrade $TF_BINARY_URL

# Python 3
$ sudo pip3 install --upgrade $TF_BINARY_URL
```

NOTE: 만약 텐서플로우 0.7.1 버전 이하에서 업그레이드하는 경우라면 protobuf 업데이트를 반영하기 위해 반드시 `pip uninstall`을 사용하여 텐서플로우 이전 버전과 protobuf 를 언인스톨한 후 진행해야 합니다.

[설치 후 테스트](#test-the-tensorflow-installation)를 해 보세요.

<a id="virtualenv-installation"></a>
## Virtualenv 설치

[Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) 은
각기 다른 파이썬 프로젝트에서 필요한 패키지들의 버전이 충돌되지 않도록 다른 공간에서 운영되도록 하는 툴입니다.
텐서플로우를 Virtualenv 로 설치하면 기존 파이썬 패키지들을 덮어쓰지 않게됩니다.

[Virtualenv](https://pypi.python.org/pypi/virtualenv) 설치 과정은 다음과 같습니다:

*  pip 와 Virtualenv 를 설치합니다.
*  Virtualenv 환경을 만듭니다.
*  Virtualenv 환경을 활성화 하고 그 안에서 텐서플로우를 설치합니다.
*  설치 후에는 텐서플로우를 사용하고 싶을 때마다 Virtualenv 환경을 활성화하면 됩니다.

pip 와 Virtualenv 를 설치합니다:

```bash
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev python-virtualenv

# Mac OS X
$ sudo easy_install pip
$ sudo pip install --upgrade virtualenv
```

디렉토리 `~/tensorflow` 에 Virtualenv 환경을 만듭니다:

```bash
$ virtualenv --system-site-packages ~/tensorflow
```

환경을 활성화 합니다:

```bash
$ source ~/tensorflow/bin/activate  # bash를 사용할 경우
$ source ~/tensorflow/bin/activate.csh  # csh을 사용할 경우
(tensorflow)$  # 프롬프트가 바뀌게 됩니다
```

이제 pip 설치 방식과 동일하게 텐서플로우를 설치합니다.
먼저 적절한 바이너리를 선택합니다:

```bash
# Ubuntu/Linux 64-bit, CPU 전용, Python 2.7
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU 버전, Python 2.7
# CUDA toolkit 7.5 와 CuDNN v4 필수. 다른 버전을 사용하려면 아래 "소스에서 설치" 섹션을 참고하세요.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU 전용, Python 2.7
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py2-none-any.whl

# Ubuntu/Linux 64-bit, CPU 전용, Python 3.4
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU 버전, Python 3.4
# CUDA toolkit 7.5 와 CuDNN v4 필수. 다른 버전을 사용하려면 아래 "소스에서 설치" 섹션을 참고하세요.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, CPU 전용, Python 3.5
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp35-cp35m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU 버전, Python 3.5
# CUDA toolkit 7.5 와 CuDNN v4 필수. 다른 버전을 사용하려면 아래 "소스에서 설치" 섹션을 참고하세요.
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp35-cp35m-linux_x86_64.whl

# Mac OS X, CPU 전용, Python 3.4 or 3.5:
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py3-none-any.whl
```

최종적으로 텐서플로우를 설치합니다:

```bash
# Python 2
(tensorflow)$ pip install --upgrade $TF_BINARY_URL

# Python 3
(tensorflow)$ pip3 install --upgrade $TF_BINARY_URL
```

Virtualenv 활성화하고 [설치 테스트](#test-the-tensorflow-installation)를 할 수 있습니다.

텐서플로우 작업을 마쳤을 때에는 환경을 비활성화 합니다.

```bash
(tensorflow)$ deactivate

$  # 프롬프트가 원래대로 되돌아 옵니다
```

나중에 텐서플로우를 다시 사용하려면 Virtualenv 환경을 다시 활성화해야 합니다:

```bash
$ source ~/tensorflow/bin/activate  # bash를 사용할 경우
$ source ~/tensorflow/bin/activate.csh  # csh을 사용할 경우
(tensorflow)$  # 프롬프트가 변경되었습니다
# 텐서플로우를 사용한 프로그램을 실행시킵니다
...
# 텐서플로우 사용을 마쳤을 때에는 환경을 비활성화 합니다
(tensorflow)$ deactivate
```

## Anaconda 설치

[Anaconda](https://www.continuum.io/why-anaconda) 는 많은 수학, 과학 패키지를 포함하고 있는 파이썬 배포판입니다. Anaconda 는 "conda" 로 불리는 패키지 매니저를 사용하여 Virtualenv 와 유사한 [환경 시스템](http://conda.pydata.org/docs/using/envs.html)을 제공합니다.

Virtualenv 처럼 conda 환경은 각기 다른 파이썬 프로젝트에서 필요한 패키지들의 버전이 충돌되지 않도록 다른 공간에서 운영합니다.
텐서플로우를 Anaconda 환경으로 설치하면 기존 파이썬 패키지들을 덮어쓰지 않게됩니다.

*  Anaconda를 설치합니다.
*  conda 환경을 만듭니다.
*  conda 환겨을 활성화 하고 그 안에 텐서플로우를 설치합니다.
*  설치 후에는 텐서플로우를 사용하고 싶을 때마다 conda 환경을 활성화하면 됩니다.

Anaconda를 설치합니다:

[Anaconda 다운로드 사이트](https://www.continuum.io/downloads)의 안내를 따릅니다.

Create a conda environment called `tensorflow`:

```bash
# Python 2.7
$ conda create -n tensorflow python=2.7

# Python 3.4
$ conda create -n tensorflow python=3.4

# Python 3.5
$ conda create -n tensorflow python=3.5
```

Activate the environment and use pip to install TensorFlow inside it.
Use the `--ignore-installed` flag to prevent errors about `easy_install`.

```bash
$ source activate tensorflow
(tensorflow)$  # Your prompt should change
```

Now, install TensorFlow just as you would for a regular Pip installation. First select the correct binary to install:

```bash
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only, Python 2.7:
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py2-none-any.whl

# Ubuntu/Linux 64-bit, CPU only, Python 3.4
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.4
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, CPU only, Python 3.5
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0rc0-cp35-cp35m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp35-cp35m-linux_x86_64.whl

# Mac OS X, CPU only, Python 3.4 or 3.5:
(tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py3-none-any.whl
```

Finally install TensorFlow:

```bash
# Python 2
(tensorflow)$ pip install --upgrade $TF_BINARY_URL

# Python 3
(tensorflow)$ pip3 install --upgrade $TF_BINARY_URL
```

With the conda environment activated, you can now
[test your installation](#test-the-tensorflow-installation).

When you are done using TensorFlow, deactivate the environment.

```bash
(tensorflow)$ source deactivate

$  # Your prompt should change back
```

To use TensorFlow later you will have to activate the conda environment again:

```bash
$ source activate tensorflow
(tensorflow)$  # Your prompt should change.
# Run Python programs that use TensorFlow.
...
# When you are done using TensorFlow, deactivate the environment.
(tensorflow)$ source deactivate
```

## Docker installation

[Docker](http://docker.com/) is a system to build self contained versions of a
Linux operating system running on your machine.  When you install and run
TensorFlow via Docker it completely isolates the installation from pre-existing
packages on your machine.

We provide 4 Docker images:

* `gcr.io/tensorflow/tensorflow`: TensorFlow CPU binary image.
* `gcr.io/tensorflow/tensorflow:latest-devel`: CPU Binary image plus source
code.
* `gcr.io/tensorflow/tensorflow:latest-gpu`: TensorFlow GPU binary image.
* `gcr.io/tensorflow/tensorflow:latest-devel-gpu`: GPU Binary image plus source
code.

We also have tags with `latest` replaced by a released version (e.g., `0.9.0rc0-gpu`).

With Docker the installation is as follows:

*  Install Docker on your machine.
*  Create a [Docker
group](http://docs.docker.com/engine/installation/ubuntulinux/#create-a-docker-group)
to allow launching containers without `sudo`.
*  Launch a Docker container with the TensorFlow image.  The image
   gets downloaded automatically on first launch.

See [installing Docker](http://docs.docker.com/engine/installation/) for instructions
on installing Docker on your machine.

After Docker is installed, launch a Docker container with the TensorFlow binary
image as follows.

```bash
$ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow
```

The option `-p 8888:8888` is used to publish the Docker container᾿s internal port to the host machine, in this case to ensure Jupyter notebook connection.

The format of the port mapping `hostPort:containerPort`. You can speficy any valid port number for the host port but has to be `8888` for the container port portion.

For NVidia GPU support install latest NVidia drivers and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
Run with

```bash
$ nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu
```

For more details see (TensorFlow docker readme)[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker].

You can now [test your installation](#test-the-tensorflow-installation) within the Docker container.

<a id="test-the-tensorflow-installation"></a>
## Test the TensorFlow installation

### (Optional, Linux) Enable GPU Support

If you installed the GPU version of TensorFlow, you must also install the Cuda
Toolkit 7.5 and cuDNN v4.  Please see [Cuda installation](#optional-install-cuda-gpus-on-linux).

You also need to set the `LD_LIBRARY_PATH` and `CUDA_HOME` environment
variables.  Consider adding the commands below to your `~/.bash_profile`.  These
assume your CUDA installation is in `/usr/local/cuda`:

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
```

### Run TensorFlow from the Command Line

See [common problems](#common-problems) if an error happens.

Open a terminal and type the following:

```bash
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```

### Run a TensorFlow demo model

All TensorFlow packages, including the demo models, are installed in the Python library.
The exact location of the Python library depends on your system, but is usually one of:

```bash
/usr/local/lib/python2.7/dist-packages/tensorflow
/usr/local/lib/python2.7/site-packages/tensorflow
```

You can find out the directory with the following command (make sure to use the Python you installed TensorFlow to, for example, use `python3` instead of `python` if you installed for Python 3):

```bash
$ python -c 'import os; import inspect; import tensorflow; print(os.path.dirname(inspect.getfile(tensorflow)))'
```

The simple demo model for classifying handwritten digits from the MNIST dataset
is in the sub-directory `models/image/mnist/convolutional.py`.  You can run it from the command
line as follows (make sure to use the Python you installed TensorFlow with):

```bash
# Using 'python -m' to find the program in the python search path:
$ python -m tensorflow.models.image.mnist.convolutional
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
...etc...

# You can alternatively pass the path to the model program file to the python
# interpreter (make sure to use the python distribution you installed
# TensorFlow to, for example, .../python3.X/... for Python 3).
$ python /usr/local/lib/python2.7/dist-packages/tensorflow/models/image/mnist/convolutional.py
...
```

## Installing from sources

When installing from source you will build a pip wheel that you then install
using pip. You'll need pip for that, so install it as described
[above](#pip-installation).

### Clone the TensorFlow repository

```bash
$ git clone https://github.com/tensorflow/tensorflow
```

Note that these instructions will install the latest master branch
of tensorflow. If you want to install a specific branch (such as a release branch),
pass `-b <branchname>` to the `git clone` command and `--recurse-submodules` for
r0.8 and earlier to fetch the protobuf library that TensorFlow depends on.

### Installation for Linux

#### Install Bazel

Follow instructions [here](http://bazel.io/docs/install.html) to install the
dependencies for bazel. Then download the latest stable bazel version using the
[installer for your system](https://github.com/bazelbuild/bazel/releases) and
run the installer as mentioned there:

```bash
$ chmod +x PATH_TO_INSTALL.SH
$ ./PATH_TO_INSTALL.SH --user
```

Remember to replace `PATH_TO_INSTALL.SH` with the location where you
downloaded the installer.

Finally, follow the instructions in that script to place `bazel` into your
binary path.

#### Install other dependencies

```bash
# For Python 2.7:
$ sudo apt-get install python-numpy swig python-dev python-wheel
# For Python 3.x:
$ sudo apt-get install python3-numpy swig python3-dev python3-wheel
```

#### Configure the installation

Run the `configure` script at the root of the tree.  The configure script
asks you for the path to your python interpreter and allows (optional)
configuration of the CUDA libraries (see [below](#configure-tensorflows-canonical-view-of-cuda-libraries)).

This step is used to locate the python and numpy header files.

```bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
```

#### Optional: Install CUDA (GPUs on Linux)

In order to build or run TensorFlow with GPU support, both NVIDIA's Cuda Toolkit (>= 7.0) and
cuDNN (>= v2) need to be installed.

TensorFlow GPU support requires having a GPU card with NVidia Compute Capability >= 3.0.
Supported cards include but are not limited to:

* NVidia Titan
* NVidia Titan X
* NVidia K20
* NVidia K40

##### Check NVIDIA Compute Capability of your GPU card

https://developer.nvidia.com/cuda-gpus

##### Download and install Cuda Toolkit

https://developer.nvidia.com/cuda-downloads

Install version 7.5 if using our binary releases.

Install the toolkit into e.g. `/usr/local/cuda`

##### Download and install cuDNN

https://developer.nvidia.com/cudnn

Download cuDNN v4 (v5 is currently a release candidate and is only supported when
installing TensorFlow from sources).

Uncompress and copy the cuDNN files into the toolkit directory.  Assuming the
toolkit is installed in `/usr/local/cuda`, run the following commands (edited
to reflect the cuDNN version you downloaded):

``` bash
tar xvzf cudnn-7.5-linux-x64-v4.tgz
sudo cp cudnn-7.5-linux-x64-v4/cudnn.h /usr/local/cuda/include
sudo cp cudnn-7.5-linux-x64-v4/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

##### Configure TensorFlow's canonical view of Cuda libraries

When running the `configure` script from the root of your source tree, select
the option `Y` when asked to build TensorFlow with GPU support. If you have
several versions of Cuda or cuDNN installed, you should definitely select
one explicitly instead of relying on the system default. You should see
prompts like the following:

``` bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow

Please specify which gcc nvcc should use as the host compiler. [Default is
/usr/bin/gcc]: /usr/bin/gcc-4.9

Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave
empty to use system default]: 7.5

Please specify the location where CUDA 7.5 toolkit is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda

Please specify the Cudnn version you want to use. [Leave empty to use system
default]: 4.0.4

Please specify the location where the cuDNN 4.0.4 library is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cudnn-r4-rc/

Please specify a list of comma-separated Cuda compute capabilities you want to
build with. You can find the compute capability of your device at:
https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your
build time and binary size. [Default is: \"3.5,5.2\"]: 3.5

Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Setting up CUPTI include
Setting up CUPTI lib64
Configuration finished
```

This creates a canonical set of symbolic links to the Cuda libraries on your system.
Every time you change the Cuda library paths you need to run this step again before
you invoke the bazel build command. For the Cudnn libraries, use '6.5' for R2, '7.0'
for R3, and '4.0.4' for R4-RC.


##### Build your target with GPU support
From the root of your source tree, run:

```bash
$ bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer

$ bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu
# Lots of output. This tutorial iteratively calculates the major eigenvalue of
# a 2x2 matrix, on GPU. The last few lines look like this.
000009/000005 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000001 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000009/000009 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
```

Note that "--config=cuda" is needed to enable the GPU support.

##### Known issues

* Although it is possible to build both Cuda and non-Cuda configs under the same
source tree, we recommend to run `bazel clean` when switching between these two
configs in the same source tree.

* You have to run configure before running bazel build. Otherwise, the build
will fail with a clear error message. In the future, we might consider making
this more convenient by including the configure step in our build process.

### Installation for Mac OS X

We recommend using [homebrew](http://brew.sh) to install the bazel and SWIG
dependencies, and installing python dependencies using easy_install or pip.

Of course you can also install Swig from source without using homebrew. In that
case, be sure to install its dependency [PCRE](http://www.pcre.org) and not PCRE2.

#### Dependencies

Follow instructions [here](http://bazel.io/docs/install.html) to install the
dependencies for bazel. You can then use homebrew to install bazel and SWIG:

```bash
$ brew install bazel swig
```

You can install the python dependencies using easy_install or pip. Using
easy_install, run

```bash
$ sudo easy_install -U six
$ sudo easy_install -U numpy
$ sudo easy_install wheel
```

We also recommend the [ipython](https://ipython.org) enhanced python shell,
which you can install as follows:

```bash
$ sudo easy_install ipython
```

If you plan to  build with GPU support you will need to make sure you have
GNU coreutils installed via homebrew:

```bash
$ brew install coreutils
```

Next you will need to make sure you have a recent [CUDA
Toolkit](https://developer.nvidia.com/cuda-toolkit) installed by either
downloading the package for your version of OSX directly from
[NVIDIA](https://developer.nvidia.com/cuda-downloads) or by using the [Homebrew
Cask](https://caskroom.github.io/) extension:

```bash
$ brew tap caskroom/cask
$ brew cask install cuda
```

Once you have the CUDA Toolkit installed you will need to setup the required
environment variables by adding the following to your `~/.bash_profile`:

```bash
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"
```

Finally, you will also want to install the [CUDA Deep Neural
Network](https://developer.nvidia.com/cudnn) (cuDNN) library which currently
requires an [Accelerated Computing Developer
Program](https://developer.nvidia.com/accelerated-computing-developer) account.
Once you have it downloaded locally, you can unzip and move the header and
libraries to your local CUDA Toolkit folder:

```bash
$ sudo mv include/cudnn.h /Developer/NVIDIA/CUDA-7.5/include/
$ sudo mv lib/libcudnn* /Developer/NVIDIA/CUDA-7.5/lib
$ sudo ln -s /Developer/NVIDIA/CUDA-7.5/lib/libcudnn* /usr/local/cuda/lib/
```

#### Configure the installation

Run the `configure` script at the root of the tree.  The configure script
asks you for the path to your python interpreter.

This step is used to locate the python and numpy header files as well as
enabling GPU support if you have a CUDA enabled GPU and Toolkit installed. For
example:


```bash
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] N
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow
Please specify which gcc nvcc should use as the host compiler. [Default is /usr/bin/gcc]:
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 7.5
Please specify the location where CUDA 7.5 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify the Cudnn version you want to use. [Leave empty to use system default]: 5
Please specify the location where cuDNN 5 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 3.0
Setting up Cuda include
Setting up Cuda lib
Setting up Cuda bin
Setting up Cuda nvvm
Setting up CUPTI include
Setting up CUPTI lib64
Configuration finished
```

### Create the pip package and install

When building from source, you will still build a pip package and install that.

```bash
$ bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

# To build with GPU support:
$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# The name of the .whl file will depend on your platform.
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-0.9.0rc0-py2-none-any.whl
```

## Setting up TensorFlow for Development

If you're working on TensorFlow itself, it is useful to be able to test your
changes in an interactive python shell without having to reinstall TensorFlow.

To set up TensorFlow such that all files are linked (instead of copied) from the
system directories, run the following commands inside the TensorFlow root
directory:

```bash
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

# To build with GPU support:
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

mkdir _python_build
cd _python_build
ln -s ../bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/* .
ln -s ../tensorflow/tools/pip_package/* .
python setup.py develop
```

Note that this setup still requires you to rebuild the
`//tensorflow/tools/pip_package:build_pip_package` target every time you change
a C++ file; add, delete, or move any python file; or if you change bazel build
rules.

## Train your first TensorFlow neural net model

Starting from the root of your source tree, run:

```bash
$ cd tensorflow/models/image/mnist
$ python convolutional.py
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
Initialized!
Epoch 0.00
Minibatch loss: 12.054, learning rate: 0.010000
Minibatch error: 90.6%
Validation error: 84.6%
Epoch 0.12
Minibatch loss: 3.285, learning rate: 0.010000
Minibatch error: 6.2%
Validation error: 7.0%
...
...
```

## Common Problems

### GPU-related issues

If you encounter the following when trying to run a TensorFlow program:

```python
ImportError: libcudart.so.7.0: cannot open shared object file: No such file or directory
```

Make sure you followed the GPU installation [instructions](#optional-install-cuda-gpus-on-linux).
If you built from source, and you left the Cuda or cuDNN version empty, try specifying them
explicitly.

### Protobuf library related issues

TensorFlow pip package depends on protobuf pip package version
3.0.0b2. Protobuf's pip package downloaded from [PyPI](https://pypi.python.org)
(when running `pip install protobuf`) is a Python only library, that has
Python implementations of proto serialization/deserialization which can be 10x-50x
slower than the C++ implementation. Protobuf also supports a binary extension
for the Python package that contains fast C++ based proto parsing. This
extension is not available in the standard Python only PIP package. We have
created a custom binary pip package for protobuf that contains the binary
extension. Follow these instructions to install the custom binary protobuf pip
package :

```bash
# Ubuntu/Linux 64-bit:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp27-none-linux_x86_64.whl

# Mac OS X:
$ pip install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp27-none-any.whl
```

and for Python 3 :

```bash
# Ubuntu/Linux 64-bit:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.0.0b2.post2-cp34-none-linux_x86_64.whl

# Mac OS X:
$ pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/protobuf-3.0.0b2.post2-cp35-none-any.whl
```

Install the above package _after_ you have installed TensorFlow via pip, as the
standard `pip install tensorflow` would install the python only pip package. The
above pip package will over-write the existing protobuf package.
Note that the binary pip package already has support for protobuf larger than
64MB, that should fix errors such as these :

```bash
[libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
protocol message was rejected because it was too big (more than 67108864 bytes).
To increase the limit (or to disable these warnings), see
CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.

```

### Pip installation issues

#### Cannot import name 'descriptor'

```python
ImportError: Traceback (most recent call last):
  File "/usr/local/lib/python3.4/dist-packages/tensorflow/core/framework/graph_pb2.py", line 6, in <module>
    from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'
```

If you the above error when upgrading to a newer version of TensorFlow, try
uninstalling both TensorFlow and protobuf (if installed) and re-installing
TensorFlow (which will also install the correct protobuf dependency).

#### Can't find setup.py

If, during `pip install`, you encounter an error like:

```bash
...
IOError: [Errno 2] No such file or directory: '/tmp/pip-o6Tpui-build/setup.py'
```

Solution: upgrade your version of pip:

```bash
pip install --upgrade pip
```

This may require `sudo`, depending on how pip is installed.

#### SSLError: SSL_VERIFY_FAILED

If, during pip install from a URL, you encounter an error like:

```bash
...
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Solution: Download the wheel manually via curl or wget, and pip install locally.


#### Operation not permitted

If, despite using `sudo`, you encounter an error like:

```bash
...
Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
Found existing installation: setuptools 1.1.6
Uninstalling setuptools-1.1.6:
Exception:
...
[Errno 1] Operation not permitted: '/tmp/pip-a1DXRT-uninstall/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/_markerlib'
```

Solution: Add an `--ignore-installed` flag to the pip command.


### Linux issues

If you encounter:

```python
...
 "__add__", "__radd__",
             ^
SyntaxError: invalid syntax
```

Solution: make sure you are using Python 2.7.

### Mac OS X: ImportError: No module named copyreg

On Mac OS X, you may encounter the following when importing tensorflow.

```python
>>> import tensorflow as tf
...
ImportError: No module named copyreg
```

Solution: TensorFlow depends on protobuf, which requires the Python package
`six-1.10.0`. Apple's default Python installation only provides `six-1.4.1`.

You can resolve the issue in one of the following ways:

* Upgrade the Python installation with the current version of `six`:

```bash
$ sudo easy_install -U six
```

* Install TensorFlow with a separate Python library:

    *  Using [Virtualenv](#virtualenv-installation).
    *  Using [Docker](#docker-installation).

* Install a separate copy of Python via [Homebrew](http://brew.sh/) or
[MacPorts](https://www.macports.org/) and re-install TensorFlow in that
copy of Python.

### Mac OS X: OSError: [Errno 1] Operation not permitted:

On El Capitan, "six" is a special package that can't be modified, and this
error is reported when "pip install" tried to modify this package. To fix use
"ignore-installed" flag, ie

sudo pip install --ignore-installed six https://storage.googleapis.com/....


### Mac OS X: TypeError: `__init__()` got an unexpected keyword argument 'syntax'

On Mac OS X, you may encounter the following when importing tensorflow.

```
>>> import tensorflow as tf
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py", line 4, in <module>
    from tensorflow.python import *
  File "/usr/local/lib/python2.7/site-packages/tensorflow/python/__init__.py", line 13, in <module>
    from tensorflow.core.framework.graph_pb2 import *
...
  File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py", line 22, in <module>
    serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02 \x03(\x0b\x32 .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')
TypeError: __init__() got an unexpected keyword argument 'syntax'
```

This is due to a conflict between protobuf versions (we require protobuf 3.0.0).
The best current solution is to make sure older versions of protobuf are not
installed, such as:

```bash
$ pip install --upgrade protobuf
```
