## Ops Models  TSN

### prepare baseModel

basemodel有三种选择：ResNet系列、mobileNetV2和BNInception三类。

#### ResNet系列

- 通过torchvision.models预加载残差网络：https://blog.csdn.net/u014380165/article/details/79119664。

- 是否进行时序移动Temporal shift。

  确定进行时序移动之后，残差网络的四个层每一层或者每一层的卷积层进行block或者blockres类型的时序偏移风格。

- 是否进行非局域化non_local

  确定进行非局域化操作之后，basemodel如果是残差网络，将对指定的层进行非局域化操作。**该操作也仅面向残差网络。**

- 修改basemodel最后一层的名字——**可以吗？**

- 修改basemodel最后的平均池化层的定义。默认是`self.avgpool = nn.AdaptiveAvgPool2d((1, 1))`。如果需要修改参数，需要重新定义再赋给basemodel`self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)`。

- 指定输入数据的像素值的均值和方差，以及输入图片的尺寸。

#### mobileNetV2和BNInception

除了没有局域化操作以外，其他基本与ResNet相同。

确定的参数有：`base_model、input_size、input_mean、input_std`



### prepare TSN

#### Dropout==0

最后一层为全连接层；同时不创建新的层；

为最后的全连接层初始化权重参数。

#### Dropout！=0

最后一层仍然为全连接层，不过在此之前添加一个丢失层。

为全连接层初始化权重参数。



### 构建光流模型construct_flow_model

### 构建光流模型construct_diff_model







