# 深度学习大作业之跟着Hilton学深度神经网络
Reducing the dimensionality of data with neural networks 
## 1. 论文理解

本文的 Motivation 为 Deep Autoencoder 由于网络层数加深, 梯度消失等问题更加明显, 这使它难以被训练. 所以本文提出了一个逐层 pretrain 的策略, 即预先训练多个受限玻尔兹曼机(RBM), 再用这些 RBM 对 Autoencoder 进行初始化, 模型的性能在降维方面超越了 PCA。
起源于2006年science上的Reducing the dimensionality of data with neural networks 是Hinton影响广泛的代表作，这篇文章也标志着deep learning进入炙热时代。

多层感知机上世纪被提出却没有广泛应用其原因在于对多层非线性网络进行权值优化时很难得到全局的参数。因为一般使用数值优化算法（比如BP算法）时需要随机给网络赋一个值，而当这个权值太大的话，就很容易收敛到“差”的局部收敛点，权值太小的话则在进行误差反向传递时离输入层越近的权值更新越慢。而Hinton设计出来的autoencoder深度网络确能够较快的找到比较好的全局最优点，它是用无监督的方法（这里是RBM）先分开对每层网络进行训练，然后将它当作是初始值来微调，实现数据的降维。这种方法被认为是对PCA的一个非线性泛化方法。

每一层网格的预训练都采用RBM方法，给定一张输入图像，我们可以通过调整网络的权值和偏置值使得网络对该输入图像的能量最低。采用多层网络，即把第一层网络的输出作为第二层网络的输入。并且每增加一个网络层，就会提高网络对输入数据重构的log下界概率值，且上层的网络能够提取出其下层网络更高阶的特征。当网络的预训练过程完成后，我们需要把解码部分重新拿回来展开构成整个网络，然后用真实的数据作为样本标签来微调网络的参数。当网络的输入数据是连续值时，只需将可视层的二进制值改为服从方差为1的高斯分布即可，而第一个隐含层的输出仍然为二进制变量。在实验的分层训练过程中，其第一个RBM网络的输入层都是其对应的真实数据，且将值归一化到了（0,1）。而其它RBM的输入层都是上一个RBM网络输出层的概率值；但是在实际的网络结构中，除了最底层的输入层和最顶层RBM的隐含层是连续值外，其它所有层都是一个二值随机变量。 此时最顶层RBM的隐含层是一个高斯分布的随机变量，其均值由该RBM的输入值决定，方差为1。

## 2. matlab代码复现
Hilton主页
http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html


code主要是2个单独的工程。

- 用MNIST数据库来进行深度的autoencoder压缩，用的是无监督学习，评价标准是重构误差值MSE。

- MNIST的手写字体识别，网络的预训练部分用的是无监督的，网络的微调部分用的是有监督的。评价标准准是识别率或者错误率。

## 3. python代码复现

+ RBM 受限玻尔兹曼机可以参考西瓜书第五章, [南瓜书对应部分](https://datawhalechina.github.io/pumpkin-book/#/chapter5/chapter5?id=_524)的内容

用 RBM pretrain 的维度为 `[784, 2000, 1000, 500, 30, 500, 1000, 2000, 784]`的 Autoencoder 模型
+ `main.py` 中实现了主要的数据加载、训练、测试的上层逻辑.
+ `utils.py` 中实现了一些辅助操作的函数, 包括 Dataloader.
+ `rbm.py` 和 `ae.py` 中实现了 RBM 和 Autoencoder 模型.

首先对于这个实验我们的上层逻辑应是: 实现四个维度分别为 `[784, 2000]`, `[2000, 1000]`, `[1000, 500]`, `[500, 30]` 的 RBM 模型并训练, 将它们作为 Autoencoder 的 pretrain 加载, 即 RBM 的propup与Autoencoder.encoder的特定层forward一致; propdown与Autoencoder.decoder的一致. pretrain加载后, 再对 Autoencoder 进行finetuning.


### 3.1 准备数据集

```python
# main.py
if __name__ == '__main__':
    # ...
    prepare_handle = PrepareFunc(args)
    train_loader, test_loader = prepare_handle.prepare_dataloader(args.dataset)
```

在 `main.py` 里, 如上代码准备数据集.

请阅读 `args` 参数的来源和设置, 搜索它的使用方式, 并在`utils.py`中实现 MNIST 数据集的 Dataloader, 利用`args`对其参数 `batch_size`、`num_workers` 进行合适地赋值.

```python
def prepare_dataloader(self, dataset_name):
    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=self.args.data_dir, train=True, transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = "TODO"
        test_dataset = torchvision.datasets.MNIST(root=self.args.data_dir, train=False, transform=torchvision.transforms.ToTensor(), download=True)
        test_loader = "TODO"
```

> ```python
> if __name__ == '__main__':
>     set_seeds(929, 929, 929, 929)
>     is_colab = 'google.colab' in sys.modules
>     args = parse_option()
> 
>     if args.time_str == '':
>         args.time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
>     if not is_colab:
>         set_gpu(args.gpu)
>     pprint(vars(args))
> ```
>
> 上述这段代码刚刚被省略了, 尝试阅读并理解它们的实现细节(在 `utils.py` 中).

`Dataloader` 将在 `train_rbm` 等函数中以 `for` 循环的形式被调用.

&nbsp;

#### 3.2 实现 RBM 模型 (`30%`)

接下来我们将实现 `class RBM`. 该部分占 `30%` 的分数.

```python
class RBM(nn.Module):
    def __init__(self, in_features, out_features, k=2):
        super(RBM, self).__init__()
        self.fc = "TODO"
        self.bias_v = "TODO"
        self.bias_h = "TODO"
        self.k = k

    def sample_p(self, p):
       return "TODO"

    def v2h(self, v):
        p_h = F.sigmoid(v @ self.fc + self.bias_h)
        return p_h, self.sample_p(p_h)

    def h2v(self, h):
        p_v = "TODO"
        return p_v, self.sample_p(p_v)

    def gibbs_h2v2h(self, h):
        p_v, a_v = self.h2v(h)
        p_h, a_h = self.v2h(p_v)
        return p_v, a_v, p_h, a_h

    def contrastive_divergence(self, x, lr):
        pos_p_h, pos_a_h = self.v2h(x)

        a_h = pos_a_h
        for _ in range(self.k):
            p_v, a_v, p_h, a_h = self.gibbs_h2v2h(a_h)

        self.fc += "TODO"
        self.bias_v += "TODO"
        self.bias_h += "TODO"

    def v2h2v(self, x):
        h, _ = self.v2h(x)
        v, _ = self.h2v(h)
        return v
```

+ 在 `__init__` 函数中, 请初始化 RBM 的连接权 weight 和阈值 bias.
+ 在 `sample_p` 函数中, 请实现 Gibbs 采样.
+ 在 `h2v` 函数中, 请仿照 `v2h` 函数考虑变量应有的赋值.
+ 在 `contrastive_divergence` 函数中, 请基于对比散度算法对连接权和阈值进行更新.

> 注意到我们实现了管理许多 RBM 模型的类, 该类在 `main.py` 中以下列形式被调用:
>
> ```python
>     train_loader, test_loader = prepare_handle.prepare_dataloader(args.dataset)
> ```
>
> 阅读该类的实现, 它集成了对许多 RBM 模型的统一 `propup` 和 `propdown` 操作, 并伪装成一个迭代器.
>
> ```python
> class RBMHandle():
>     def __init__(self):
>         self.models = []
> 
>     def v2h(self, x):
>         for prev_m in self.models:
>             x, _ = prev_m.v2h(x)
>         return x
> 
>     def h2v(self, h):
>         for prev_m in self.models[::-1]:
>             h, _ = prev_m.h2v(h)
>         return h
> 
>     def v2h2v(self, x):
>         return self.h2v(self.v2h(x))
> 
>     def append(self, m):
>         self.models.append(m)
> 
>     def __len__(self):
>         return len(self.models)
> 
>     def __getitem__(self, i):
>         return self.models[i]
> ```
>
> 遵照上述传参和返回值的形式是被鼓励的.

+ 如上, 我们准备好 RBM 模型后, 请在 `train_rbm` 函数中补全 criterion 中缺失的重构输入.

  ```python
  def train_rbm(model, train_loader, rbm_models, criterion, args):
      print("Begin training..")
      for epoch in range(args.max_epoch):
          epoch_loss = 0
          for idx, (x, _) in enumerate(train_loader):
              x = x.view(x.shape[0], -1).to(torch.device('cuda'))
  
              model.contrastive_divergence(rbm_models.v2h(x), args.lr_rbm)
              loss = criterion("TODO", x)
              epoch_loss += loss
          print(f'Epoch {epoch} Loss: {epoch_loss:.4f}.')
      print("Completed.")
  ```

  提示: 此时我们该如何正确构造与 `rbm_models` 相关的重构输入.

&nbsp;

#### 3.3 实现 Autoencoder 模型 (`30%`)

至此, 我们可以训练一系列的 RBM 模型了, 如下:

```python
    ae_dims = [784, 2000, 1000, 500, 30]
    rbm_models = prepare_handle.prepare_model('rbm_handle')
    criterion = prepare_handle.prepare_loss_fn()
    if args.do_train_rbm:
        for in_features, out_features in zip(ae_dims[:-1], ae_dims[1:]):
            cur_model = prepare_handle.prepare_model('rbm', [in_features, out_features])
            train_rbm(cur_model, train_loader, rbm_models, criterion, args)
            rbm_models.append(cur_model)
```

接下来我们实现 `class Autoencoder`. 该部分占 `30%` 的分数.

```python
def basic_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Sigmoid()
    )

class Autoencoder(nn.Module):
    def __init__(self, layers=[784, 2000, 1000, 500, 30]):
        super(Autoencoder, self).__init__()
        self.encoder = "TODO"
        self.decoder = "TODO"

    def forward(self, x):
        hidden = "TODO"
        reconstructed = self.decoder(hidden)
        return reconstructed
```

+ 在 `__init__` 函数中, 请利用 `basic_block` 和 `layers` 构造 encoder 与 decoder.
+ 在 `forward` 函数中, 请补全代码.

接下来我们将实现 `load_rbm_pretrained_models` 函数, 该方法将 `rbm_models` 内存储的 RBM 模型权重加载进 `ae_model`, 请注意加载完成后 `ae_model` 的encoder和decoder连接权weight互为转置关系, 阈值bias为对应 RBM 的. 例如第一层的encoder和最后一层的decoder对应同一个 RBM, 与原论文一致.

```python
class PrepareFunc(object):
    def __init__(self, args):
        self.args = args

    def load_rbm_pretrained_models(self, ae_model, rbm_models):
        rbm_model_length = len(rbm_models)
        if rbm_model_length == 0:
            return

        model_dict = ae_model.state_dict()
        pretrained_dict = {}
        for i, cur_model in enumerate(rbm_models):
            encoder_id = i
            decoder_id = rbm_model_length - 1 - i
            pretrained_dict[f'encoder.{encoder_id}.0.weight'] = "TODO"
            pretrained_dict["TODO"] = cur_model.bias_h
            pretrained_dict[f'decoder.{decoder_id}.0.weight'] = "TODO"
            pretrained_dict["TODO"] = "TODO"

        model_dict.update(pretrained_dict)
        ae_model.load_state_dict(model_dict)
```

+ 在 `load_rbm_pretrained_models` 函数中, 请补全代码, 以加载对应的参数.

接下来我们将完成 Autoencoder 的训练部分, 该部分的训练与常规的PyTorch训练一致.

```python
def train_ae(model, train_loader, criterion, optimizer, args):
    for epoch in range(args.max_epoch):
        epoch_loss = 0
        for idx, (x, _) in enumerate(train_loader):
            x = x.view(x.shape[0], -1).to(torch.device('cuda'))

            loss = "TODO (这里大约要写五行)"

            epoch_loss += loss
        print(f'Epoch {epoch} Loss: {epoch_loss:.4f}.')
```

&nbsp;

#### 3.4 结果分析和讨论 (`20%`)

如下, 在 `val_ae` 中, 我们实现了一个简单的分类器, 来验证 Autoencoder 的有效性.

```python
def val_ae(model, test_loader, save_result=None, is_raw=False):
    from sklearn.linear_model import LogisticRegression
    hidden, label = [], []
    for idx, (x, y) in enumerate(test_loader):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if isinstance(y, list):
            y = y[0]
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        if is_raw:
            hidden.append(x.detach().cpu())
        else:
            hidden.append(model.encoder(x).detach().cpu())
        label.append(y.cpu())
    hidden_np = torch.cat(hidden).numpy()
    label_np = torch.cat(label).numpy()
    clf = LogisticRegression()
    clf.fit(hidden_np, label_np)
    test_acc = clf.score(hidden_np, label_np)
    print(f'Test Accuracy: {test_acc}.')
    if save_result is not None:
        with open(f'{YOUR_STUDENT_ID}.csv', 'w') as f:
            f.write(f'{save_result},{test_acc}\n')
```

同时, 我们还需要实现一个简单的t-SNE, 将 Autoencoder 中间的30维打印出来, 如下:

```python
def tsne_ae(model, cur_loader, file_name=''):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import manifold
    Axes3D

    sampled_num = 10 * 200
    hidden, label = [], []
    for idx, (x, y) in enumerate(cur_loader):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        if isinstance(y, list):
            y = y[0]
        x = x.view(x.shape[0], -1).to(torch.device('cuda'))
        hidden.append(model.encoder(x).detach().cpu())
        label.append(y.cpu())
    hidden_np = torch.cat(hidden).numpy()
    label_np = torch.cat(label).numpy()
    sampled_idx = np.random.choice(hidden_np.shape[0], sampled_num, replace=False)
    X, y = hidden_np[sampled_idx], label_np[sampled_idx]
    t_SNE_method = manifold.TSNE(n_components=2, init='pca', random_state=929)
    trans_X = t_SNE_method.fit_transform(X)
    plt.scatter(trans_X[:, 0], trans_X[:, 1], s=15, c=y, alpha=.4)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'saves/t-sne-{file_name}.png')
    plt.clf()
```

+ 请在 `main.py` 中适当的位置插入如上两个函数, 分别输出:
  1. 原始图像的 `val_ae` 分类结果, t-SNE 图像.
  2. RBM pretrain后, Autoencoder finetuning 的降维后 `val_ae` 分类结果 (将写入 `f'{YOUR_STUDENT_ID}.csv'` 文件, t-SNE 图像.
  3. 未加 RBM pretrain, 直接训练 Autoencoder 后的降维 `val_ae` 分类结果, t-SNE 图像.
  4. 考虑分析 (1) RBM pretrain, (2) Autoencoder降维后的带来的性能提升.
+ 对比它们的不同, 写在报告里.
+ 将整个实验中你的想法写在报告里, 写最关键的即可.


### 附加: Debias任务与ColoredMNIST数据集

在代码中开启 `--bonus` 选项, 可以发现刚刚训练的模型在一个新的数据集 ColoredMNIST 上进行了测试:

```python
    if args.bonus:
        debias_train_loader, debias_test_loader = prepare_handle.prepare_dataloader('ColoredMNIST')
        val_ae(ae_model, debias_test_loader)
        tsne_ae(ae_model, debias_test_loader, 'colored-mnist-test')

        val_extract_bias_conflicting(ae_model, debias_train_loader)
```

+ ColoredMNIST 数据集:

  该数据集将 MNIST 数据集的数字染色, 背景保持为黑色, 但是颜色在训练时是有偏的, 举个例子:

  + `0` 这个类别的数字, 被染成红色的有5400个, 其余600个被染成其余9个不同的颜色.

  实际上, 颜色 和 数字都叫做数据集的bias, 我们期望模型在ColoredMNIST数据集上学到的分类bias应是数字, 因为这本来就是一个数字分类任务, 但是实验发现, 模型更倾向于学习更简单的bias: 颜色, 对于上述存在大量 bias-aligned 样本, 即颜色和数字相关的样本数量大时, 模型会完全失效, 可以参考代码里对该数据集的`val_ae`.

+ Debias 任务所叙述的就是, 如何在这样的bias数据集上识别差的bias并学习到一个好的表示.

  Debias 问题是有难度的, 我们将该问题退化:

  + 考虑上述例子, `0` 这个类别的数字, 被染成红色的有5400个, 其余600个被染成其余9个不同的颜色. 如何在不知道颜色bias的label(即不知道每个样本的颜色)的情况下, 学一个颜色bias的分类器.

    具体来说, 模型能得到的有: ColoredMNIST数据 *x*, 数字的label *y* (例如 `0`, `1`, ..., `9`). 我们需要训练得到一个能将**所有类别中**上述 600 个被染成其余9个不同的颜色的样本识别出来.

+ 分析:

  + 我们进一步将问题简化, 仅考虑在一个类别内完成这件事, 即:

    考虑 `0` 这个类别的数字, 将 600 个被染成其余9个不同的颜色的样本识别出来.

    形式上这是可行的, 毕竟90%的样本都是红色, 模型应能发现不同, 只不过:

    ```bash
    该问题变成了一个从无监督表示中抽取"异常"样本的问题.
    ```

    即 Autoencoder 能给出类内相对于颜色label的无监督表示, "异常"样本对应10%其余9个不同颜色的样本.

    该问题随即被分为两个步骤:

    + 第一即为Autoencoder抽取表示的有效程度, 该表示能否正确表达"数字形状"这一bias的特征? 该表示是否会被"颜色"这一bias冲掉并造成灾难性的后果?
    + 第二是在获取了无监督表示后, 下游分类器能否考虑到表示的bias来进行分类(提取)? 例如这里 600 个被染成其余9个不同的颜色的样本应被看做是"异常"样本还是分类边界周围的关键样本?

    在这两个步骤中, 颜色label都是不能使用的, 也就是说, 对于提取其余9个不同的颜色的样本, 这就是一个无监督问题, 不能使用颜色label, 也是该部分的唯一要求, 其余方法、模型都可以使用, 可以魔改框架代码.

