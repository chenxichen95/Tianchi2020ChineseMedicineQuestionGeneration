# 1.所使用的依赖

1. **操作系统版本:** Ubuntu 16.04 LTS

2. **Python的版本**  3.6.12

3. **需要安装的Python package**
    + tensorflow-gpu       1.14.0

    + Keras                2.3.1
    + scikit-learn         0.23.2
    + tqdm                 4.50.0

4. **CUDA 版本**  10.0

5. **cuDNN 版本** 7.6.0

    


# 2. 解决方案

## 2.1 模型思路：
针对“中医文献问题生成”这一问题，我们队伍的解决方案如下：

+ **训练阶段：** (1) 将 UniLM-MASK 和 BERT 类型的预训练模型 作为 baseline，采用“文档+答案+问题”的句子对作为输入。(2) 结合使
  用标签平滑和基于 Embedding 层的对抗扰动来防止过拟合。(3)利用知识蒸馏技术来提高单模型的泛化性能。
+ **生成阶段：** (1) 使用 beam search 策略来进行问题生成。(2) 在每个时间步的单词预测阶段，使用基于 **WoBERT** 和 **WoNEZHA** 进行集成投票预测。

本解决方案在复赛榜单排名第四，**Rouge-L** 得分是：0.6278。
	

## 2.2 数据预处理模块：
使用 main.py 中的 load_train_json 函数，从原始数据 round1_train_0907.json 中提出去 QA pair。  
我们尝试过剔除原始文本中的一些 非法字符 ，对一些字符进行替换。但模型的效果没有提升，所以，数据预处理的模块，暂时只起到数据提取和编码的作用。    
我们将 text 和 answer 作为 Bert 的第一个句子，question 作为 bert 的第二个句子。具体编码格式如下：  
~~~
    token_ids: [CLS] + text + [SEP] + answer + [SEP] + question + [SEP]
    segment_ids:  0 + 0 + 0 + 0 + 0 + 1 + 1	
~~~
数据集划分：
	我们使用 sklearn 的 train_test_split 函数对训练数据进行划分，90% 做为训练集， 10% 作为验证集。通过设置随机种子为 42 ，让实验具有可复现性。

## 2.3 模型建立：
### 2.3.1 预训练文件

基于已做的实验结果，我们选择性能表现最好的两种 bert 模型作为 ensemble 的对象：   

+ 第一个是 **WoBERT** ，其预训练权重文件来自于**追一科技**开源的 [chinese_wobert_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/WoBERT)  。
+ 第二个是 **WoNEZHA** ，其预训练文件来自于**追一科技**开源的 [chinese_wonezha_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/WoBERT) 。

### 2.3.2 UniLM-MASK

为了让 BERT 模型具有 seq2seq 的能力，使其能够处理 NLG 任务，例如：问题生成 。
基于 2019 年发表在 NIPS 上的 [《Unified Language Model Pre-training for Natural Language Understanding and Generation》](http://papers.nips.cc/paper/9464-unified-language-model-pre-training-for-natural-language-understanding-and-generation.pdf) ，我们使用了论文中提到的 Seq2seq Mask 来替换原 Bert  Multi-head attention 中的 attention mask ，让 Bert 在训练的过程中，具有以下的特性： 

1. 第一个句子 (text + answer) 只能看到自己本身的 token，而看不到第二个句子 (question) 的 token 。
2. 第二个句子 (question) 只能看到前面的 token ，包括第一个句子 (text + answer) 中含有的 token 。   

这两个特性 让 Bert 具有了 seq2seq 能力。

## 2.3.2 标签平滑

标签平滑是一种正则化方法，通常用于分类问题，目的是防止模型在训练时过于自信地预测标签，改善泛化能力差的问题。

将真实的标签进行 label smoothing
$$
\hat{y}_i = y_i*(1 - \alpha) + \frac{\alpha}{K}
$$
$y_i$ 是第 i 个样本的 one-hot 标签向量，维度是词表的大小

$\alpha$ 是平滑因子，通常是 0.1 ， $K$ 是类别个数， $\hat{y}_i$ 平滑后的标签向量

### 2.3.3 针对 Embedding 层的对抗扰动

对抗扰动本质上就是对抗训练，就是构造了一些对抗样本加入到原数据集中，增强模型对对抗样本的鲁棒性，同时提高模型的表现。但 NLP 的输入是文本，本质上就是 one-hot 向量，因此不存在所谓的小扰动。因此，我们可以从 Embedding 层进行对抗扰动。在我们的方案中，我们是直接对 Embedding 层的权重进行扰动，让 look-up 后的词向量发生变化。

对抗扰动的公式：
$$
\mathop{min}\limits_{\theta} \mathbb{E}_{(x,y) \in D}[\mathop{max}\limits_{\Delta x\in \Omega} Loss(x+\Delta x, y; \theta)]
$$
$\theta$ 是参数模型，$L(x,y;\theta)$ 单个模型的loss，$\Delta x$ 是对抗扰动，$\Omega$ 是扰动空间。

1. 对 $x$ 加入对抗扰动 $\Delta x$ ，目的是让 Loss 越大越好，即尽量让模型预测错误
2. 当然 $\Delta x$ 不是越大越好，所以他会有一个约束空间 $\Omega$
3. 每个样本构造出来对抗样本 $x + \Delta x$ 后，用它作为模型的输入，来最小化 loss，更新模型的参数

**使用 FGM 计算 ** $\Delta x$

因为目的是为了增大 loss ，loss 减少的方法是梯度下降，那么 loss 增大的方法，我们就可以使用梯度上升

所以，可以这样取：
$$
\Delta x = \epsilon \triangledown_x Loss(x, y; \theta)
$$
$\epsilon$ 是一个超参数，一般取 0.1

为了防止计算出来的梯度过大，我们对梯度进行标准化
$$
\Delta x = \epsilon \frac{\triangledown_x Loss(x, y; \theta)}{||\triangledown_x Loss(x, y; \theta)||}
$$

针对 Embedding Weights 进行对抗扰动，维度情况：

$x \in \mathbb{R}^{vocab\_size, dim}$  是词嵌入层的权重

$\triangledown_x Loss(x, y; \theta) \in \mathbb{R}^{vocab\_size, dim}$ 是词嵌入层的梯度

### 2.3.4 知识蒸馏

知识蒸馏就是通过引入与 Teacher Model  相关的软目标（soft-target）作为 total loss 的一部分，以指导 Student Model 的训练，实现知识迁移（knowledge transfer）。

在我们的方案中，Teacher Model  和 Student Model 是结构一样的 BERT 模型。

**知识蒸馏的实现步骤：**

1. 训练一个 Teacher Model
2. 在 Student Model  的训练过程中，加入对 Teacher Model 输出的标签概率 (soft target) ，并计算与其的 softmax loss ，
3. 再与真实标签 (hard target) 的 softmax loss  进行叠加，作为一个总的 loss 

**引入温度系数：**

直接使用训练好的 teacher model 输出的预测概率，可能不太合适。

因为，一个网络训练好后，对正标签有很高的置信度，负标签的值都很接近0，对损失函数的贡献非常小，小到可以忽略不计。

所以，可以引入一个温度变量，来让概率分布更加平滑：
$$
\hat{t}_i = softmax(t_i/T)
$$
$t_i$ 是 teacher 模型进行 $softmax$ 之前的概率向量。 

$T$ 是缩放因子。

当  $T$ 越高，$softmax$ 的输出概率越平滑，其分布的熵越大，负标签携带的信息会被相对地放大，模型训练将更加关注负标签。

放大负标签概率还有一个好处： 就是可以让 Student 模型学习到不同负标签与正标签之间的关系。比如一只狗，在猫这个类别下的概率值可能是0.001，而在汽车这个类别下的概率值可能就只有0.0000001不到，这能够反映狗和猫比狗和汽车更为相似，这就是大规模神经网络能够得到的更为丰富的数据结构间的相似信息。

### 2.3.5 模型的损失函数

1. Teacher Model: 训练所使用的 Loss function 是 “Bert 预测出来的全部 question 单词” 与 “原全部 question 单词”  （标签平滑后）的 KL散度 。 

2. Student Model: 训练所使用的 Loss function 是 “Bert 预测出来的全部 question 单词概率” 与 “原全部 question 单词概率”  （标签平滑后）的 KL散度 以及   “Bert 预测出来的全部 question 单词概率” 与  “Teacher 预测出来的全部 question 单词概率”的KL散度。 

### 2.3.6 训练模型的保存：

在每一个 epoch 结束后，会计算模型在验证集上的 Rouge-L 分数，如果  Rouge-L 高于之前最优的  Rouge-L ，则保存最新的模型。我们使用 greedy search 来对验证集进行问题生成。  

## 2.4 参数设置：
1. 文本长度设置（主要基于文本长度的分布）：
    + text 的最大长度 (max_t_len) 为 384
    + answer 的最大长度 (max_a_len) 为 96
    + question 的最大长度 (max_q_len) 为 32

2. 训练参数（主要基于大量的调参实验）：
    + batch_size : 4
    
    + 梯度累积步数 (gradient_accumulation_steps) : 8
    
    + 迭代次数(EPOCHS) : 5  
    
      大部分实验在第 5 次迭代训练结束后，模型性能达到最优
    
    + 标签平滑的平滑因子 (label_weight) : 0.1
    
    + 对抗训练的 $\epsilon$  (ADV_epsilon) : WoBERT 为 0.3， WoNEZHA 为 0.1
    
    + Teache model 在 Student loss 中所占的权重 (teacher_rate) : 0.5
    
    + 温度系数 (temperature) : 10
    
3. 优化器设置：
    + 使用 Adam 优化器
    + 初始学习率为 3e-5
    + 使用学习率线性衰减函数，让学习率从第 1 个 step 到最后一个 step ，线性衰减到初始学习率 的 50% 。

## 2.5 模型预测：
加载训练好的 **WoBERT** 和 **WoNEZHA** 的 Student Model   

在进行问题生成时，对于每个 token 的预测，将两个模型预测的 logits 进行平均加权求和，让预测出来的 token 尽量在 **WoBERT** 和 **WoNEZHA** 中得分靠前。  

我们采用 beam search 的方式，来对测试集进行问题生成， beam 的个数为 5 。	