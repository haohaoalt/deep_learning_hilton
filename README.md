# 深度学习大作业之跟着Hilton学深度神经网络

## **0. 文件结构说明：**

deep_learning_hilton

- README.md 
- autoencoder_matlab  %Hilton autoencoder 论文matlab代码
- homework

  - ae.py

  - bonus
  - data
  - main.py
  - rbm.py
  - saves
  - utils.py

- paper
  - science.pdf  Reducing the dimensionality of data with neural networks 
  - science_som.pdf

运行方法：

```shell
# 论文matlab代码运行方式
run mnistdeepauto.m in matlab
run mnistclassify.m in matlab
# 论文代码python复现
python main.py
```

## **1. 论文理解**

本文的 动机是Deep Autoencoder 由于网络层数加深, 梯度消失等问题更加明显, 这使它难以被训练. 所以本文提出了一个逐层预训练的策略, 即预先训练多个受限玻尔兹曼机(RBM), 再用这些 RBM 对 Autoencoder 进行初始化, 模型的性能在降维方面超越了 PCA。
起源于2006年science上的Reducing the dimensionality of data with neural networks 是Hinton影响广泛的代表作，这篇文章也标志着deep learning进入炙热时代。

多层感知机上世纪被提出却没有广泛应用其原因在于对多层非线性网络进行权值优化时很难得到全局的参数。因为一般使用数值优化算法（比如BP算法）时需要随机给网络赋一个值，而当这个权值太大的话，就很容易收敛到“差”的局部收敛点，权值太小的话则在进行误差反向传递时离输入层越近的权值更新越慢。而Hinton设计出来的autoencoder深度网络确能够较快的找到比较好的全局最优点，它是用无监督的方法（这里是RBM）先分开对每层网络进行训练，然后将它当作是初始值来微调，实现数据的降维。这种方法被认为是对PCA的一个非线性泛化方法。

每一层网格的预训练都采用RBM方法，给定一张输入图像，我们可以通过调整网络的权值和偏置值使得网络对该输入图像的能量最低。采用多层网络，即把第一层网络的输出作为第二层网络的输入。并且每增加一个网络层，就会提高网络对输入数据重构的log下界概率值，且上层的网络能够提取出其下层网络更高阶的特征。当网络的预训练过程完成后，我们需要把解码部分重新拿回来展开构成整个网络，然后用真实的数据作为样本标签来微调网络的参数。当网络的输入数据是连续值时，只需将可视层的二进制值改为服从方差为1的高斯分布即可，而第一个隐含层的输出仍然为二进制变量。在实验的分层训练过程中，其第一个RBM网络的输入层都是其对应的真实数据，且将值归一化到了（0,1）。而其它RBM的输入层都是上一个RBM网络输出层的概率值；但是在实际的网络结构中，除了最底层的输入层和最顶层RBM的隐含层是连续值外，其它所有层都是一个二值随机变量。 此时最顶层RBM的隐含层是一个高斯分布的随机变量，其均值由该RBM的输入值决定，方差为1。

## **2. matlab代码复现**
Hilton主页
http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html


code主要是2个单独的工程。

- 用MNIST数据库来进行深度的autoencoder压缩，用的是无监督学习，评价标准是重构误差值MSE。

  ![Image](https://tva1.sinaimg.cn/large/007dpYmwly8h2t6ry7jvzj30qw05ejrm.jpg)

  最终结果：

  test_error:   4.58756519908758 

  train_error:  4.38230235524354

  

- MNIST的手写字体识别，网络的预训练部分用的是无监督的，网络的微调部分用的是有监督的。评价标准准是识别率或者错误率。论文数据错误率是1.2%，但是自己运行结果如下:
  
  test_crerr
  
  251.550965446241        7.05819398164967        5.97431327430693        5.73477124302898        5.93366446693345        5.76065634728710        5.61232084409794        5.18238004595161        5.02669525863520        5.25873765794806        5.72458145116393        5.81680083661501        6.33291565567338        6.40311081995889        6.54332412430685        7.10479724092344        7.27548457457018        7.69851781853759        7.71597392397768        7.76431304726476        8.12992500236058        8.62810421015263        8.94743176764003        9.27903476126801        9.48464220092207        9.91930128287717        10.0459292249364        10.6092033047883        10.7176198850718        10.8499250619729

## **3. python代码复现**

+ RBM 受限玻尔兹曼机可以参考西瓜书第五章, [南瓜书对应部分](https://datawhalechina.github.io/pumpkin-book/#/chapter5/chapter5?id=_524)的内容

用 RBM pretrain 的维度为 `[784, 2000, 1000, 500, 30, 500, 1000, 2000, 784]`的 Autoencoder 模型
+ `main.py` 中实现了主要的数据加载、训练、测试的上层逻辑.
+ `utils.py` 中实现了一些辅助操作的函数, 包括 Dataloader.
+ `rbm.py` 和 `ae.py` 中实现了 RBM 和 Autoencoder 模型.

首先对于这个实验上层逻辑应是: 实现四个维度分别为 `[784, 2000]`, `[2000, 1000]`, `[1000, 500]`, `[500, 30]` 的 RBM 模型并训练, 将它们作为 Autoencoder 的 pretrain 加载, 即 RBM 的propup与Autoencoder.encoder的特定层forward一致; propdown与Autoencoder.decoder的一致. pretrain加载后, 再对 Autoencoder 进行finetuning.

> 详细代码见deep_learning_hilton/homework/*

Thanks for the wonderful tutorial and framework provided by [ZhangYikaii](https://github.com/ZhangYikaii) in the [wiki](https://github.com/ZhangYikaii/Auxiliary-Material-for-AI-Platform-Application-Course/wiki/作业-自编码器-(Autoencoder)).

Thanks for https://github.com/G-U-N/Restricted-Boltzmann-machine-for-AutoEncoder
