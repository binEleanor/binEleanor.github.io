---
layout: post
title:  "浅析Transformer模型"
date:   2019-08-11
excerpt: "介绍Attention机制， Transformer模型的整体架构"
tag:
- Attention
- Transformer
comments: true
---


## 浅析Transformer模型

**Transformer模型**起初被提出于谷歌《Attention Is All you Need》这篇论文。



其完全抛弃了CNN，RNN等结构模式，仅仅通过注意力机制（self-attention）和前向神经网络（Feed Forward Neural Network），不需要使用序列对齐的循环架构就实现了较好的performance 。

（1）摒弃了RNN的网络结构模式，其能够很好的并行运算；

（2）其注意力机制能够帮助当前词获取较好的上下文信息。



论文发布没多久，就有人用tensorflow复现该论文，代码可查阅：<a href = "https://github.com/Kyubyong/transformer">https://github.com/Kyubyong/transformer</a>，官方实现版本：<a href= "https://github.com/tensorflow/tensor2tensor">https://github.com/tensorflow/tensor2tensor</a>

较好的一篇详解Transformer模型的一篇博客：http://jalammar.github.io/illustrated-transformer/



以下将分成三个部分浅析一下Transformer模型的架构：

- 注意力机制；

- 位置编码；

- Transformer：Encoder-Decoder模型；



### 1. 注意力机制（Attention）

Attention起初被提起是在计算机视觉当中，用于捕捉图像中的感受野，直观解释就是我们看一张图片的时候，第一眼看上去的往往是一些比较明显的图片特征，将注意力集中到图像的特定部分，或者当我们带着某种目的去查看一张图片的时候，往往我们更加关注我们想要关注的内容，我们就会对整张图片的不同像素点有不同的注意力权重。

<a href="https://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf">《Recurrent Models of Visual Attention》</a>在RNN模型上使用Attention机制进行图像分类；


**Attention-机器翻译**

Attention用于自然语言处理可以被理解成，对于给定的文本段落，哪些token对整篇文本的语义影响程度较大，哪些影响程度较小。

<a href = "https://arxiv.org/pdf/1409.0473.pdf">《Neural Machine Translation by Jointly Learning to Align and Translate》</a>使用Attention机制在机器翻译任务上将翻译和对齐同时进行。


机器翻译模型是一个典型的sequence-to-sequence模型，将源语言翻译到目标语言，不加attention机制的结构模式常常是使用一个RNNs作为编码模型，另外一个RNNs作为解码模型，解码层的输入是编码模型的最后输出。

**不加注意力机制**的解码模型单词生成过程：

$$
p(y_i|y_1, y_2, ..., y_{i-1}, c) = f(y_{i-1}, s_i, c)
$$

其中$$y_i$$表示当前步要翻译出来的单词，$$y_1, y_2,...,y_{i-1}$$表示当前步翻译出来单词的前序已经被翻译完成的单词，$$s_i$$表示通过RNN获取的当前步的一个隐状态信息，$$c$$表示整个目标待翻译句子的上下文信息，$$f$$是非线性激活函数。


**引入注意力机制**之后：

$$
p(y_i|y_1, y_2, ..., y_{i-1}, x) = g(y_{i-1}, s_i, c_i)
$$

注意这个地方的$$s_i$$虽然也是表示通过RNN获取的当前步的一个隐状态信息，但是其计算方式较上文不一样：

$$
s_i= f(s_{i-1}, y_{i-1}, c_i)
$$
其依赖的不再是整体的一视同仁的上下文信息，而是对上下文信息计算注意力机制之后的上下文信息：

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j
$$
$$T_x$$表示输入句子的长度，$$h_j = f(x_j, h_{j-1})$$是通过RNN网络模型计算得到的中间隐状态信息，这里的$$\alpha_{ij}$$表示第$$j$$个字对于第$$i$$个字的权重，直观理解就是第$$j$$个字对于翻译第$$i$$个字的影响因子，那么这个影响因子要怎么去计算呢？

$$
\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})}
$$


看到这个式子，乍一眼不就是softmax求概率吗？计算输入的每个字对于第$$i$$个字的得分，然后计算第$$j$$个字的得分占比，不就是第$$j$$个字对于第$$i$$个字的影响权重了吗？  

现在来解释一下$$e_{ij}$$是什么？根据上边的描述，我们知道$$e_{ij}$$其实就是在计算第$$j$$个字对于第$$i$$个字而言的一个影响得分函数，这个得分函数就是注意力机制的计算:

$$
e_{ij} = a(s_{i-1}, h_j)=v_a^Ttanh(W_a[s_{i-1};h_j])
$$


$$h_j$$是第$$j$$个字的隐状态信息（记住这里获取隐状态信息就是encoder模型编码获取的隐状态信息），$$s_{i-1}$$表示的是依赖于上文解码出来的隐状态信息，其信息计算的目的就是为了生成下一个词。


上文介绍的Attention机制在后续的演化当中，可以称之为global attention，即在每一次计算注意力权重矩阵的时候，需要计算所有目标待翻译词与将要翻译出来的词的权重矩阵，可想而知，如果我们的目标待翻译的句子是一篇过长文本的话，计算全局的注意力机制将是耗时耗力的，而且对于过长文本，可能在一定的滑动窗口外，第一个词对后续的词的影响几乎为0.

基于上述局限性，衍生出局部注意力机制的概念，其大体思想还是与上述文章中一致的，只是计算注意力权重矩阵的时候，有些差异，下边我们来介绍一下这篇文章<a href="https://arxiv.org/pdf/1508.04025.pdf">《 Effective Approaches to Attention-based Neural Machine Translation》</a>是如何计算注意力权重矩阵。

首先我们简化一下隐层状态信息的计算：

$$
\tilde{h}_t = tanh(W_c[c_t, h_t]), c_t表示预测目标词y_t解码层获取的隐层状态信息
$$
$$h_t = f(h_{t-1}, c)$$，这里的$$c$$表示源句子全部的上下文下信息。



从这个地方可以看出上述论文与本论文计算隐状态信息及注意力权重的方式是有些不一致的：

Bahdanau等人通过上一个隐层状态的信息来计算注意力权重：

$$
s_{t-1} \rightarrow a_t  \rightarrow c_t  \rightarrow s_{t}
$$
最后求解的概率函数为：

$$
g(y_{t-1}, s_t, c_t)
$$
这个部分的$$s_t$$部分等价于$$\tilde {h}_{t}$$.



而本文直接从当前状态出发计算：

$$
h_{t} \rightarrow a_t  \rightarrow c_t  \rightarrow \tilde h_{t}
$$
将两者结合起来看的话，本文最后的求解概率函数应该为：

$$
g(y_{t-1}, \tilde h_t, c_t)
$$


预测下一个字生成的概率计算公式为：

$$
p(y_t|y_{<t}, x) = softmax(W_s \tilde{h}_t)
$$
根据注意力机制的核心思想，我们知道注意力机制的引入，主要在于对$$c_t$$的计算，原始不引入注意力机制的模型当中，直接将编码层的输出结果作为$$c_t$$，上篇论文通过计算输入每个单词对于当前预测词的权重来获取$$c_t$$，那么本文是如何计算$$c_t$$的呢？



**Global Attention类似于soft-attention**

$$
a_t = align(h_t, \overline{h}_s) = \frac{exp(socre(h_t, \overline{h}_s))}{\sum_j exp(socre(h_t, \overline{h}_j))}
$$

其中$$h_t$$表示目标词获取的隐状态信息， $$\overline{h}_{s}$$表示源句子的隐状态信息，计算源句子与木笔哦啊句子的计算方式可以有多种，如内积，加权乘，或者使用激活函数作用之后再计算。

$$
最开始的location-based注意力机制的计算表达式：a_t = softmax(W_ah_t)
$$

Global注意力机制的思想与Bahdanau等人提出的那篇文章基本思想是完全一致的，但是本文引出了多种计算注意力得分的函数，可以借鉴一下。



**Local Attention**

Local Attention提出的目的旨在缓解Global Attention中计算输入每个词对当前预测词的一个权重信息时的消耗过大，Local Attention选择与预测目标词相关的词，即subset of source sentences positions来计算他们对目标词的权重。

该模型受启发于图像领域的soft-attention（对应于global attention）与hard-attention，hard-attention的主要原理是每次在计算注意力权重的时候，选择图像中的某一个小块（one patch）计算注意力权重，但是与此同时我们可以发现，当只使用一小块进行计算的时候，其目标函数往往是不可导的，反响传播是不可行的，为了计算其梯度信息，往往需要比较复杂的技巧，如强化学习或者蒙特卡洛采样等方法来估算其梯度。

hard-attention 是一个随机采样的过程，采样集合是输入向量的集合，输出是某一个特定的输入向量，采样概率是由对齐函数计算的权重信息来获取的；soft-attention是一个带权求和的过程，求和的集合是输入向量的集合，soft-attention较hard-attention更常见，因其可以使用反向传播对参数进行求解。

该论文提出的local attention着重于使用一个小的上下文窗口（部分输入向量-缩小求解注意力关注的范围）使其不仅能够在关注与目标词相关的词语，而且最后刻画出来的目标函数是可导的。

首先，对于在$$t$$时刻的目标词生成其对齐位置$$p_t$$，$$c_t$$的计算就不再依赖于全部的输入句子，而依赖于给定的窗口内的词，根据对齐位置选择窗口区间$$[p_t-D, p_t+D]$$，D是根据经验来选择的，那么要计算的窗口词的数目就变成了$$2D+1$$。



**局部注意力机制存在两种可能的变体**

（1）假设对齐位置是monotonic的，目标句子与原句子位置完全一一对应，此时可以直接简单地设置$$p_t=t$$；

（2）不对对齐位置进行假设，即存在非monotonic的情况，此时需要去预测对齐位置到底是哪个位置，而不能像上述一样简单的设置成$$t$$，该论文给出了$$p_t$$的预测公式：

$$
p_t = S \cdot sigmoid(v_p^Ttanh(W_ph_t))
$$

其中$$W_p, v_p$$都是模型预测对齐位置信息时的参数信息，$$S$$表示原始句子的长度，由于sigmoid函数的值域为(0,1]，所以预测出的$$p_t$$的范围在(0, S)，最后在计算权值的时候乘以一个关于对齐位置信息的高斯分布：

$$
a_t(s) = align(h_t, \tilde{h}_s)exp(-\frac{(s-p_t)^2}{2\sigma^2})
$$

CNN+Attention可以参考该论文 <a href="https://arxiv.org/pdf/1512.05193.pdf">《ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs》</a>

综上所述，我们可以知道，其实attention机制的主要建模在于对得分函数（score function）如何计算，对齐函数（align function）如何计算，以及如何生成上下文的向量函数（context vector）等。

前边描述的global attention相较于Bahdanau等人提出的那篇论文提出了几种不同的**得分函数**的计算方法。还有就是上下文向量的计算方法，local attention通过对当前词建模不同的上下文向量来进行注意力机制的建模。



**Self-Attention**   


自注意力机制通常也被称作是**intra-Attention**，与上述描述的机器翻译注意力机制不同之处在于计算的注意力是其target与source是相等，但是其内部的计算过程还是跟原来一致的，只是计算的对象发生了改变而已。某个句子自己和自己计算注意力机制，可以捕获同一个句子中单词之间的一些语义特征。

  

解释到这里，有没有一种感觉，self-attention似乎把RNN能做的事情也能做了，RNN可以获取句子间的长期依赖，self-attention也能获取句子间的依赖信息，并且还能够并行运算，摒弃了RNN的递归结构，RNN无法获取序列的层次结构信息，self-attention能够在某种程度上获取句子的层次结构信息。

  

回过头来看一眼，卷积神经网络CNN，卷积过程是通过不同的卷积核计算在卷积窗口内词与词之间的语义特征信息，卷积核-卷积窗口，似乎脑海中第一下蹦出来的就是n-gram语言模型，个人理解，CNN在自然语言处理当中扮演的角色模型就是类似n-gram的作用，所以其能够在分类上达到比较好的效果，但是同时也继承了n-gram的局限性：无法获取句子之间的长期依赖性。


  
注意到这里，似乎觉得self-attention是不是无法获取字与字之间的组合特征呢？

再想一想self-attention的本质，计算输入句子每个字对目标字的权重，输入句子的**每个字**，这不就是把卷积核的窗口设成了类似整个句子长度吗？是的，个人理解就是这个样子的，此时就能够获取词与词之间的组合特征啦......所以self-attention的厉害之处在于其似乎把CNN/RNN的一些功能特性都继承了,《Attention is all you need》，这篇论文的名字是不是也就顺利成章的成立了，名副其实，self-attention可以帮助我们implement所有需要实现的功能。
  
  
  
Oh， 再提一句 


**Multi-head Attention**，多头注意力机制，个人认为其就是简单的将计算注意力机制的方法计算了多次，然后将计算得到的结果进行concat，计算多次注意力权重的直观解释是：每次可以捕捉不同层面的语义特征信息，即我们希望每个头计算出来的权重都能够帮助我们去捕捉不同层面不同级别的信息特征（简单理解：假设是一个多分类问题，我们可以使用一个头去计算获取的属于第一个类的特征，第二个头获取属于第二个类别的特征，依次下去，....，可能这个理解有点简单粗暴，但是可以通俗的这么去理解这个多头的含义）

接着上边的卷积神经网络与多头注意力机制一起联想一下，多头注意力是不是有点类似于卷积操作中的多个channel进行卷积？其目的都是一样的，获取不同层次的特征信息。

Attention的本质在于不再是通过一些知识的全局/部分记忆，叠加或者是怎样去建模整个信息的获取过程，而是有选择性（权重）的对全文内容进行focus。



### 2.位置编码

Transformer模型使用位置编码的主要原因是：在没有使用任何的递归和卷积结构的情况下，为了能够更加充分地利用句子顺序这个特征，必须引入token在整个句子中的相对信息或者绝对信息来记录其位置信息。

Transformer模型直接在input embedding之后加入位置信息，其位置信息使用的是余弦函数和正弦函数来计算：  

$$
PE(pos, 2i) = sin(pos / 10000^{\frac{2i}{d_{model}}})
$$


$$
PE(pos, 2i+1) = cos(pos / 10000^{\frac{2i}{d_{model}}})
$$
  
  
$$pos$$表示位置，$$i$$表示维度，还不太理解位置函数的为什么这么设置。



### 3. Transformer：Encoder-Decoder模型



**Encoder-Decoder模型架构**   


- encoder可被看作将输入句子$$x=(x_1, x_2, ..., x_n)$$映射的连续的表达$$z=(z_1, z_2, ..., z_n)$$的操作；  

- decoder可被看作将对于给定的连续表达$$z=(z_1, z_2, ..., z_n)$$将其解析到输出句子$$y=(y_1, y_2, ..., y_m)$$的操作；  





#### Encoder模块 

Transformer中，encoder模块是由六个相同的层次组成的，每一层都具有两个子层，一个是多头注意力机制，一个是简单的前向全连接，每两个子层之间进行残差连接，之后接一层normalization的操作，所以每一个则层的输出应该是：

$$
LayerNorm(x+SubLayer(x))
$$

layer_normaliztion操作可使用简单的：

$$
\hat{x} = \frac{x- \mu}{\sigma}
$$
  

#### Decoder模块

Transformer中，decoder模块同样是由六个相同的层次组成的，每一层都具有与encoder中相同的两个子层之外，还具有第三个子层，用于计算encoder块对于输出的多头注意力，同样在每两个子层之间进行残差连接，然后接一层normalization的操作。



**Multi-head Attention**   


文中定义$$(Q, K, V)$$，计算Attention的操作如下：

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
在self-attention中，$$Q, K, V$$是相同的。

整个attention的计算过程：

1. 通过输入句子和三个权重矩阵$$W^Q, W^K, W^V$$进行乘法运算，得到三个相对于输入维度较小的向量矩阵$$q, k, v$$

2. 计算Query Vector与Key Vector之间的注意力：$$score = q * k$$

3. 得到注意力得分之后将其除以8（为啥除这个数呢，其实不是很明白），再取softmax.   
4. softmax之后得到权重，将权重值乘以原来的Value Vector，就可以输入到前向神经网络单元。  


$$(Q, K, V)$$其实一直没太明白具体指的是什么？

但是似乎可以简单理解成是：Q就表示对当前词的嵌入层表达，K表示整个上下文的全部表达，V也是整个上下文的全部表达，不同之处在于K要用于和Q进行乘法的运算得到要被进行softmax计算的值，而V只有在词嵌入层获取特征表达的时候才需要进行改变，在得到注意力权重之后一直保持不变的状态。

多头注意力权重就是把多次计算的结果进行concat的操作：    

$$
MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O
$$  

其中：   

$$
 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
$$   


#### 前向神经网络

前向神经网络的计算公式如下：  

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$   

从公式可以看出此处使用了两个线性转换和ReLU激活函数。





参考资料：

[1]. http://jalammar.github.io/illustrated-transformer/

[2]. https://blog.csdn.net/yimingsilence/article/details/79208092

[3]. Bahdanau, D., Cho, K. & Bengio, Y. Neural Machine Translation by Jointly Learning to Align and Translate. Iclr 2015 1–15 (2014).

[4]. Luong, M. & Manning, C. D. Effective Approaches to Attention-based Neural Machine Translation. 1412–1421 (2015).

[5]. Rush, A. M. & Weston, J. A Neural Attention Model for Abstractive Sentence Summarization. EMNLP (2015).

[6]. Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need [C]//Advances in Neural Information Processing Systems. 2017: 5998-6008.







