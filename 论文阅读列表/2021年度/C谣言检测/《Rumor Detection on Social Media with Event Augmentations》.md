### 《Rumor Detection on Social Media with Event Augmentations》

​		本文主要是面向谣言检测，做了三种数据增强方式，每种增强方式由不同的问题考虑。同时使用了对比自监督学习来克服对labeled数据的依赖。



代码连接：https://github.com/hzy-hzy/RDEA
#### abstract

网络数据的快速发展，谣言检测很重要；

深度学习的高层特征表示提取能力很强，但是需要大量的标注数据用于训练，**意味着时间消耗和数据不高效。**

数据增强的方式：三种通过修改响应特征和时间结构以提取传播模式，同时学习用户（root用户）参与的内在表示。

使用对比自监督学习方式以实现事件增强的高效执行，同时缓解数据有限的问题。

#### 引言

传播渠道的广泛应用的同时，也帮助了谣言的传播；

谣言检测的常见方法是尝试利用**谣言的内容和传播结构**，因为通常认为这两者能够**反映谣言本身的特征和传播模式**；

当前方法的不足是：严重依赖监督学习，这就意味着对标注数据的依赖很严重。

同时，【4】提出了一个数据增强技术：通过使用语境词汇表示ELMo，但是**它却忽视了事件的传播特征**。

**Motivated by this（对标注数据的依赖和数据增强技术的可行性），**提出了自监督学习框架RDEA：final predictation。

- 三个事件增强策略：节点掩码，子图和图边丢失；
- permute内容的特征和传播结构以生成事件的正例；**由此，数据的内在联系被用来生成自监督学习的signals，同时通过增强事件的对比预训练以增强事件表示event representation。**
- 使用label以精调模型得到最终的prediction。



#### 方法论

一个数据集由多个事件构成，一个事件由这个事件的所有post和post之间的图结构联系。而每个post的表示由一个one-hot向量构成。

#### 事件增强

谣言事件的一个特征是：一个谣言的post和comments是内在联系，同时彼此独立。**因此，数据增强操作应该为其量身定制。而谣言的特点是：脆弱的事件结构和结构表示；同时 malicious users and naive users恶意用户和幼稚用户倾向于提升其传播范围，而造成了 the echo chamber effect 回声室效应在社交媒体中频繁发生。**

为此，提出事件增强策略：**它们是通过修改图结构和节点属性，虽然如此，但也保留了事件的关键信息**。

**节点掩码**

通过分析谣言传播的两个传播者 恶意传播者和幼稚传播者的传播目的的不同，**由此得出结论如果分析仅仅专注于谣言的参与者，可能会起到坏的结果。**

为了解决这个问题，**在每次epoch中随机mask图中除了root节点以外的其他节点的特征。**

**子图**

观察发现，当rumor的整个传播链条被考虑的时候，大多数的人会支持真实事件，同时否定错误的谣言。与此同时，如果仅仅观察rumor的早期response的情况，会发现用户对谣言的支持，却无关于其真实性的趋势这一现象。**因此如果对谣言的整个发展链条的事件都关注的话，训练的时候，可能会阻碍infer的早期阶段检测rumor的能力。**

为了解决这个问题，在整个事件图中使用随机游走，从root post开始。 The walk parallel and iteratively travels to its neighborhood with a probability 平行行走并以一定的概率迭代到其邻域。

**边dropping**

【14】EdgeDropping 对于GCN模型而言能够有效缓和过拟合和过平滑等问题。

具体的，它在每次epoch训练时，随机移除图中的边。这将有助于数据增强和减少信息传递。

另外，在该rumor场景中，EdgeDrop将同时减弱回声室效应（潜在影响用户的立场和观点，同时增加社会两极分化和极端主义）

#### Contrastive Pre-training

<img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907112307816.png" alt="image-20210907112307816" style="zoom:80%;" />

（TODO：比较好奇 MI的计算公式）

GCL是图卷积编码层，一个节点的特征向量经过它将得到下一层该节点的特征向量。**可以是不同的卷积层。**

CONCAT行为是拼接加和行为，将每个节点在各层中的特征向量加和。

READOUT行为是将所有节点在各层中特征的融合向量经过某种方式得到融合，获取到事件图的全局表示。**可以是不同的池化方法**

**本文中使用的卷积层为GIN，而readout池化层选择均值。**

对比预训练的目标是 **最大化谣言传播图数据集的互信息值。**

计算方式是：

![image-20210906153947645](/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210906153947645.png)

其中：

- $\psi$  表示 神经网络的参数集合；
- I表示互信息评估器，T表示判别器；G表示一个输入的事件图样本，Gpos表示G的正例，Gneg表示G的负例；
- sp 表示一个softplus函数。**它是relu函数的平滑版本。**（TODO：这里之所以使用它，是有什么考虑吗？）
- 正例事件：使用输入事件图和生成图的样本的local patch representations；负例事件：使用一个batch中其他事件图的local patch representation。

#### **更好地预测谣言的真实性**

融合了对比预训练得到的互信息，想要强调的source post和textual features文字特征，以得到事件图的representations。

其中textual features 是本事件的所有post的features的均值得到的一个特征向量。

![image-20210907095348094](/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907095348094.png)

![image-20210907095402675](/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907095402675.png)



#### Fine tuning

使用预训练的参数作为该阶段模型的初始化参数，接着用labeled data训练模型。

预测的输出通过多个全连接层和一个softmax层；通过交叉熵损失函数和L2正则项得到损失值；



#### 实验细节

**数据集：**Twitter15/16两个数据集中，节点表示用户，边表示响应关系。其中特征根据TF-IDF值排序选择前5000个word。其中每个source post 被标注为四个类别：Nonrumor (N), False rumor (F), True rumor (T), and Unverified rumor(U),（**谣言不一定是假的，是中性词。**）

**Baseline：**

- DTC：使用了决策树；
- SVM-TS：基于SVM的线性时间序列模型，使用手工特征做预测；
- RvNN：结合了GRU单元的递归树结构模型，通过树结构学习谣言表示；
- PPC_RNN+CNN：结合了RNN和CNN的谣言检测模型，特别面向早期的谣言检测；
- BI-GCN：直接使用GCN，通过双向传播结构学习谣言特征表示；

**Metrics**：acc和F1；

parameters settings：

- 数据集分成五份，进行五折交叉验证；
- SGD+Adam
- 隐藏层特征向量的维度是64；
- 掩码率0.2，子图率0.4，drop率0.4.
- 自监督预训练epoch25；监督fine-tuning100epcoh；同时使用早停法，如果验证集的acc**停止上升连续10次**，将停止训练。

#### 实验结果与分析

**两个数据集上的acc和F1结果：**

<img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907110002149.png" alt="image-20210907110002149" style="zoom:80%;" />

- 深度学习方法效果都好于手工fueatures；
- proposed method 优于其他的深度学习方法；

其他模型的不足的分析：

- RvNN：仅仅使用了所有叶子节点的特征向量，受传播链越往后的post的影响越大，丢失了对former post的信息。
- PPC_RNN+CNN：将传播结构视为平坦的时间序列，而丢失了较多的结构性信息；
- Bi-GCN：特征表示比较容易受到噪声的干扰，同时需要大量的标注数据用于训练。

RDEA模型：

- 通过对比预训练得到最大化的互信息值，使得模型能够捕捉到谣言传播过程中的内在联系；
- 通过强调root post，模型能够更重视root post中的信息；

#### 消融实验

- root feature enhancement：indispensible 必不可少。
- textual graph：
- event augmentation：
- mutual information： 

<img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907111212899.png" alt="image-20210907111212899" style="zoom:80%;" />



#### 有限标注数据

标注越少，提升提升越大，由此呈现出模型的鲁棒性；

<img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907111417832.png" alt="image-20210907111417832" style="zoom:80%;" />

（TODO：这一布是怎么做的？）

#### 提早进行谣言检测的价值

在谣言出现的早期进行谣言检测，能有效地阻止谣言的传播和影响。

实验：选择一系列的检测deadlines，只使用deadlines之前的post用于测试实验的acc；

![image-20210907112117942](/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907112117942.png)

结果显示：We can also observe that the performance of all three methods is almost fixed at an early time (slightly improves over time).

在早期的时候，acc就已经基本上不变化了，这种随着时间伴随的轻微变动，**验证了早期检测的有效性和意义**。



#### 模型结构

- Encoder：len(num_gc_layers)个子模块：linear > ReLU > Linear > GINConv；

    计算过程中采集每一个GCL层的输出。

- FF：feed forward Layer： **block**(Linear>ReLU>Linear>ReLU>Linear>ReLU) + **linear_shortcut**(Linear)

- Classifier：**1**(Linear>dropout>prelu) > **2**(Linear>dropout>prelu) > **3** (Linear>dropout>prelu) > Linear > softmax

整体流程图：

<img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907154924067.png" alt="image-20210907154924067" style="zoom:80%;" />





### 《Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks》

​		本文介绍了propagation 和 dispersion 是 谣言的两个重要特征，本文引入了一种新的双向图模型，Bi-GCN。它通过上下传播和下上传播路经以捕捉这两个方面的特征。同时该方法注重source post的重要性。（这里对于propagation和dispersion的区别是，前者的描述是深浅，后者的描述是宽窄）

#### **引言**

​		以往的深度学习方法以及传统方法对于谣言的检测，大都只停留在对propagation特征的学习上，而忽视谣言的扩散特征，即dispersion。而dispersion特征并非是CNN可以学习到的，**适合的方法应该是GCN，至少它面向的是这种非欧几里德空结构的数据**。

但是如果简单将数据套用到GCN中，虽然会获取到相关节点之间的关系特征，但是却无法获取到节点之间的顺序特征。

TODO：**地铁站数据是否可以使用这种GCN网络，同时考虑到节点之间的联系以及顺序特征；其次使用掩码行为，以克服过分专注于某个节点的信息。获取到比较平稳的正常的非异常的信息特征。**

<img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/2021年度/C谣言检测/《Rumor Detection on Social Media with Event Augmentations》.assets/image-20210907210638739.png" alt="image-20210907210638739" style="zoom:67%;" />

​		由此，作者提出了双向GCN，TD-GCN用于构建propagation特征，后者用于表示dispersion特征。最后两个特征通过全连接层得到融合。**期间**，为了充分利用root post的信息，克服回声室效应，作者将root post的特征和每一个GCN层的隐藏层特征进行concatenate融合。训练过程中，为了避免模型过拟合，使用了DropEdge。**这就是作者所有的创新点。从这一篇文章看出，RDEA框架的一些想法在本文中已经有提及，而RDEA模型的最大优点在于其克服标注数据的对比自监督预训练行为。**



#### 相关工作

作者简述了谣言检测经历了传统方法、RNN、CNN和GAN等方法，并且介绍了一些使用trick，比如SVM中的随机游走核、融合了注意力机制的RNN和增加了额外的特征（比如propagation 结构和文字内容的融合），最后引出了GCN，不过这里的GCN使用的是一阶切比雪夫网络。



#### Preliminaries 预知

这里提及事件event的代表是由三部分组成：text contents 文字内容、user information 和propagation structure.

dropedge方法是一种GCN模型中缓解过拟合的方法2019，它可以增加数输入数据的随机性和丰富性；就像旋转、水平翻转。

#### Bi-GCN Rumor Detection Model

​		作者提及







### 《A Survey of Information Cascade Analysis: Models, Predictions, and Recent Advances》


