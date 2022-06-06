深度学习大作业之跟着Hilton学深度神经网络

## **1. 文件结构说明**

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
# run mnistclassify.m in matlab
# 论文代码python复现
python main.py
```

## **2. 论文理解**

​		深度学习由于BP算法在1991年被指出存在梯度消失问题时而陷入低谷。当时还没有Adam等优化器，也还没有卷积神经网络，只有多层感知器。到06年这是第一次提出有效解决梯度消失问题的解决方案：**无监督预训练对权值进行初始化+有监督训练微调**，并且重新让深度学习掀起浪潮。深度自编码器首先用受限玻尔兹曼机进行逐层预训练，得到初始的权值与偏置（权值与偏置的更新过程用对比散度CD-1算法）。然后，自编码得到重构数据，通过BP算法进行全局微调权值与偏置（权值与偏置的更新过程用Polak-Ribiere共轭梯度法）。

>  参考：https://zhuanlan.zhihu.com/p/430436914

​		文章想要解决两个问题：

- **BP算法的梯度消失**

​		多层感知机上世纪被提出却没有广泛应用其原因在于对多层非线性网络进行权值优化时很难得到全局的参数。因为一般使用数值优化算法（比如BP算法）时需要随机给网络赋一个值，而当这个权值太大的话，就很容易收敛到“差”的局部收敛点，权值太小的话则在进行误差反向传递时离输入层越近的权值更新越慢。所以，本文从**初始化**的角度去解决训练结果不好的问题（相对的，优化器是从BP算法的角度优化）。Hinton设计出来的autoencoder深度网络确能够较快的找到比较好的全局最优点，它是用无监督的方法（这里是RBM）先分开对每层网络进行训练，然后将它当作是初始值来微调，实现数据的降维。这种方法被认为是对PCA的一个非线性泛化方法。

-  **神经网络构成的非线性降维算法**

​		在压缩输入向量的维度方面（降维），常用的是PCA算法（线性降维）。基于1986年由Rumelhart提出的**单层**的自编码器（autoencoder），本文提出一个新的降维算法：由一个非线性的、自适应的**多层**的编码器（encoder）来将高维的数据转化为低维的编码，和一个相似的解码器（decoder）来将低维编码恢复成原始高维数据，整个系统构成一个自编码器（autoencoder，但是称为DAE/deep autoencoder更合适，引自该[博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/a819825294/article/details/53516980)）。 该网络用BP算法进行训练，所以也存在上述问题。本文的具体目标就在于优化该算法。

​		RBM 受限玻尔兹曼机参考西瓜书第五章, [南瓜书对应部分](https://datawhalechina.github.io/pumpkin-book/#/chapter5/chapter5?id=_524)的内容。用 RBM pretrain 的维度为 `[784, 2000, 1000, 500, 30, 500, 1000, 2000, 784]`的 Autoencoder 模型。每一层网格的预训练都采用RBM方法，给定一张输入图像，我们可以通过调整网络的权值和偏置值使得网络对该输入图像的能量最低。采用多层网络，即把第一层网络的输出作为第二层网络的输入。并且每增加一个网络层，就会提高网络对输入数据重构的log下界概率值，且上层的网络能够提取出其下层网络更高阶的特征。当网络的预训练过程完成后，我们需要把解码部分重新拿回来展开构成整个网络，然后用真实的数据作为样本标签来微调网络的参数。当网络的输入数据是连续值时，只需将可视层的二进制值改为服从方差为1的高斯分布即可，而第一个隐含层的输出仍然为二进制变量。在实验的分层训练过程中，其第一个RBM网络的输入层都是其对应的真实数据，且将值归一化到了（0,1）。而其它RBM的输入层都是上一个RBM网络输出层的概率值；但是在实际的网络结构中，除了最底层的输入层和最顶层RBM的隐含层是连续值外，其它所有层都是一个二值随机变量。 此时最顶层RBM的隐含层是一个高斯分布的随机变量，其均值由该RBM的输入值决定，方差为1。

## **3. matlab代码复现**

Hilton主页
http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html


code主要是2个单独的工程。

- 用MNIST数据库来进行深度的autoencoder压缩，用的是无监督学习，评价标准是重构误差值MSE。本次是训练4个隐含层的autoencoder深度网络结构，输入层维度为784维，4个隐含层维度分别为1000,500,250,30。整个网络权值的获得流程梳理如下：

  1. 首先训练第一个rbm网络，即输入层784维和第一个隐含层1000维构成的网络。采用的方法是rbm优化，这个过程用的是训练样本，优化完毕后，计算训练样本在隐含层的输出值。
  2. 利用1中的结果作为第2个rbm网络训练的输入值，同样用rbm网络来优化第2个rbm网络，并计算出网络的输出值。并且用同样的方法训练第3个rbm网络和第4个rbm网络。
  3. 将上面4个rbm网络展开连接成新的网络，且分成encoder和decoder部分。并用步骤1和2得到的网络值给这个新网络赋初值。
  4. 由于新网络中最后的输出和最初的输入节点数是相同的，所以可以将最初的输入值作为网络理论的输出标签值，然后采用BP算法计算网络的代价函数和代价函数的偏导数。
  5. 利用步骤3的初始值和步骤4的代价值和偏导值，采用共轭梯度下降法优化整个新网络，得到最终的网络权值。以上整个过程都是无监督的。

  ![Image](https://tva1.sinaimg.cn/large/007dpYmwly8h2t6ry7jvzj30qw05ejrm.jpg)

  最终结果：

  test_error:   4.58756519908758 

  train_error:  4.38230235524354

  

- MNIST的手写字体识别，网络的预训练部分用的是无监督的，网络的微调部分用的是有监督的。评价标准准是识别率或者错误率。在MNIST数据集上，随机初始化的神经网络的最优错误率为1.6%；SVM（support vector machines，支持向量机）则为1.4%；结构为 784 - 500 - 500 - 200 - 10的网络在采用本文提出的预训练方法后错误率下降到1.2%。
  
  test_crerr
  
  251.550965446241        7.05819398164967        5.97431327430693        5.73477124302898        5.93366446693345        5.76065634728710        5.61232084409794        5.18238004595161        5.02669525863520        5.25873765794806        5.72458145116393        5.81680083661501        6.33291565567338        6.40311081995889        6.54332412430685        7.10479724092344        7.27548457457018        7.69851781853759        7.71597392397768        7.76431304726476        8.12992500236058        8.62810421015263        8.94743176764003        9.27903476126801        9.48464220092207        9.91930128287717        10.0459292249364        10.6092033047883        10.7176198850718        10.8499250619729

## **4. python代码复现**

首先对于这个实验上层逻辑应是: 实现四个维度分别为 `[784, 2000]`, `[2000, 1000]`, `[1000, 500]`, `[500, 30]` 的 RBM 模型并训练, 将它们作为 Autoencoder 的 pretrain 加载, 即 RBM 的propup与Autoencoder.encoder的特定层forward一致; propdown与Autoencoder.decoder的一致. pretrain加载后, 再对 Autoencoder 进行finetuning.

+ `main.py` 中实现了主要的数据加载、训练、测试的上层逻辑.
+ `utils.py` 中实现了一些辅助操作的函数, 包括 Dataloader.
+ `rbm.py` 和 `ae.py` 中实现了 RBM 和 Autoencoder 模型.

> 详细代码见deep_learning_hilton/homework/*

这里将自己实验结果及图片放在下边对比：

![](https://img-blog.csdnimg.cn/65d0ea9a77fa46cba8ba8751b4c8970c.png)

<center>auto-encoder result ae: 0.8691</center>

![](https://img-blog.csdnimg.cn/de3d77cb5d3e476da07e8666ee474a5f.png)

<center>auto-encoder result rbm: 0.9083</center>

可以看出RBM预训练自动编码器后的特征可视化性能有一定的提升！

因为自己训练硬件算力的限制，调整超参改善的实验结果参考https://github.com/G-U-N

![](https://img-blog.csdnimg.cn/1cbc85dafc954a56ad74dddacf3a0abe.png)

<center>Feature Visulization after Vanilla AutoEncoder!</center>

![](https://img-blog.csdnimg.cn/00ee8c5e7cb0435ebf05b61c8564bd1f.png)

<center>Feature Visulization after RBM pretrained AutoEncoder  Best!!!👍

​    

## **5. 总结**

- 找对资料，避免重复造轮子；
- 编程实现的能力需要加强；
- 算力的重要性；

> Thanks for the wonderful tutorial and framework provided by [ZhangYikaii](https://github.com/ZhangYikaii) in the [wiki](https://github.com/ZhangYikaii/Auxiliary-Material-for-AI-Platform-Application-Course/wiki/作业-自编码器-(Autoencoder)).
>
> Thanks for https://github.com/G-U-N/Restricted-Boltzmann-machine-for-AutoEncoder
