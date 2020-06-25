

#  End-to-End Object Detection with Transformers  -- from Facebook AI


## 背景

目前业界主流的目标检测任务，在流程中使用了很多手工设计的先验知识，比如 “rpn”，“anchor”，“nms”等等，这些知识使得网络结构不够简洁，类似于一种**曲线救国**的思路。那我们可不可以换个角度，重新思考一下目标检测任务的建模呢？

本文从序列预测的角度入手，将目标检测任务看做是一种“**从图像到集合**”的问题。给定一张图像，输出所有目标的**无序集合**，每个目标用类别表示并且包含一个边界框。

作者认为这种问题很适合用 transeformer 来解决。

作者给这个方案起名叫 **DETR**（DEtection TRansformer）

![](/images/对比.png)

## 一些相关工作（或者叫需要解决的几个难点）

### 集合预测
做集合预测第一个要解决的问题就是重复预测，在以前的目标检测任务中，大家用 nms 来做后处理过滤掉分数低的预测目标，保留一个唯一结果。

在DETR中，使用二分匹配来引导模型训练，让模型天然的学会给出唯一的预测结果，具体流程后面说。

### 并行预测 （避免自回归）
在nlp或者语音任务中，transeformer的预测是非并行的，用前一时刻的输出结果做下一时刻的输入，然后再decode产生下一时刻的输出，一直到结束。这样可以说是合理的，因为语言或者文本确实有顺序上的相关。

但是做无序的目标集合的预测，确实不需要用所谓“前一时刻”来帮助预测“下一时刻”。为了避免自回归，DETR 对原始的transeformer 解码部分做了一些改进。

* 提前确定输入有多少个input tokens
* decoder的输入embedding（object queries）也是学出来的，代码中最开始初始化用rand做初值

## 网络结构

非常简单，作者说，你不需要额外写任何特殊的层实现，只要你使用的深度学习框架能支持基本的cnn和transeformer， 你直接调包就能搭建出DETR。（使用pytorch 完成推理代码只需要50多行）


![](/images/网络结构1.png)

### backbone
* resnet50 或者 resnet101
* image features: 

![](/images/input.png)
![](/images/output.png)

*  Typical values we use are C = 2048 and H=H0/32, W=W0/32

### transeformer 

![](/images/transeformer.png)

* 因为transeformer的attentione中无位置信息，需要在encoder 的输入加上positional encoding， 具体计算方式采用sin编码。（参考: [如何理解Transformer论文中的positional encoding，和三角函数有什么关系？](https://www.zhihu.com/question/347678607/answer/864217252)）
* 前面提到的，预先假设一张图片中最多有N个目标（作者取 N = 100），初始化生成随机N个object queries 作为decoder的输入。这N个object queries设计在这里，是用来学习positional encodings的，他们将作为positional encodings 加在decoder的各个部分。
* 同样，decoder的输出也就是N个结果，在这里对所有目标集合中引入一个空目标集作为padding, 所以推理时，N个输出的结果，需要去掉预测为空的output.
* decoder 结束后，通过一个ffn结构，来得到集合中最终每个目标的类别和bbox

### loss
decoder结束后，会得到一个数量为N的集合（其中一些结果为空，一些结果为预测的目标），那这个集合如何与ground truth 对应起来，来计算loss呢？

DETR中使用了最小二分匹配loss来解决这个问题

* 将ground truth 加空目标做padding,同样补充到N这个数量
* 此时，我们就得到了2个待匹配的组，我们的目标是对2个组做一一映射，得到最优的对应关系。
* 一个二分图，做匹配的可能性有很多种，相应的，每种匹配方式都可以计算出匹配上的预测-gt之间的bboxloss,分类loss(跟传统目标检测相同)，各个匹配方式相加，就能得到该匹配方式下的total loss。 在这么多组total loss中，有一组匹配关系计算得到的total loss值最小，那么我们就需要找到这个最小的匹配关系，然后计算它的loss, 做梯度回传去优化网络。
* 寻找这个最小匹配关系的方法，使用了**匈牙利算法**，[此处可参考链接](https://www.renfei.org/blog/bipartite-matching.html)

相关公式：

![](/images/公式1.png)

![](/images/公式2.png)

![](/images/公式3.png)

![](/images/公式4.png)

* 补充：在DETR中，bbox loss的计算中l1 之间计算预测值与gt的差，而不是像anchor那样计算基于某个中间值的差

## 实验

* 对比faster-rcnn

* 在大目标检测上优于faster-rcnn，小目标还不行

* 性能更快，参数更少

  ![](/images/对比实验.png)

## 消融性实验
### encoder作用
* 结论1：encoder层数增加可以提升效果。 
* 结论2: 通过可视化，发现在attention层就已经把各个目标区分开了
![](/images/层数.png)

![](/images/encoder可视化.png)
### decoder作用
* 结论1: decoder初期还不能很好解决重复预测问题，但是到后几层后就可以达到与使用nms差不多的效果
* 结论2: 通过可视化，发现在decoder阶段，为了更好的预测边框，模型开始注意到目标的一些跟边界有关的特征区域，并且这种注意，对于重叠目标无影响（对比faster系列的方法）
![](/images/decoder结论.png)
![](/images/decoder可视化.png)

### FFN的重要性
* 结论：去掉ffn层，会使AP下降2.3个点

### positional encodings的实现方式
* 结论：本来可以有多种方式，当前的组合取得了最佳效果
![](/images/sin.png)

### l1 loss 和GIoU loss对比
* 结论：单独用效果都一般，联合到一起最佳
![](/images/loss.png)

### 预测结果的分布
* 结论：all slots 都有预测大型水平目标的能力
![](/images/box.png)

## 未来展望
* 使用fpn来解决小目标问题
* 在图像+文本双模态任务上的探索


## 其他补充说明
### 训练情况
* 时间比较长：需要300个epoch(一轮epoch代表数据集过一遍)，需要6天
* 硬件：batchsize=64, 需要16块v100（32g）
### 其他视觉任务上的应用
* 全景图像分割

### 论文地址：
* [https://arxiv.org/abs/2005.12872v2 ](https://arxiv.org/abs/2005.12872v2 ) 
* 注意有3个版本

### 代码地址
* [https://github.com/facebookresearch/detr]() 

### demo
* [地址](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)
