本项目采用tensorflow_hub对英语电影评论数据集（IMDB）进行文本二分类。

### 维护者

- jclian91

### 环境

```
python: 3.7.9
cuda: 10.2
cudnn: 7.6.5
cudatoolkit: 10.1
tensorflow: 2.3.0
```

其余第三方Python模块见requirements.txt文件。

### 模型结构

模型结构如下：

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
text (InputLayer)               [(None,)]            0                                            
__________________________________________________________________________________________________
preprocessing (KerasLayer)      {'input_word_ids': ( 0           text[0][0]                       
__________________________________________________________________________________________________
BERT_encoder (KerasLayer)       {'sequence_output':  109482241   preprocessing[0][0]              
                                                                 preprocessing[0][1]              
                                                                 preprocessing[0][2]              
__________________________________________________________________________________________________
dropout (Dropout)               (None, 768)          0           BERT_encoder[0][13]              
__________________________________________________________________________________________________
classifier (Dense)              (None, 1)            769         dropout[0][0]                    
==================================================================================================
Total params: 109,483,010
Trainable params: 109,483,009
Non-trainable params: 1
__________________________________________________________________________________________________
```

训练完后的模型文件保存格式为pb格式（SavedModel 格式）。

### 模型评估

|模型参数|dev评估指标|test评估指标|
|---|---|---|
|长度128, BATCH_SIZE=32, EPOCH=5|0.8850|0.8850|
|长度300, BATCH_SIZE=32, EPOCH=5|||


### 模型预测

对新样本进行预测，结果如下：

```
input: this is such an amazing movie! : score: 0.998060
input: The movie was great!           : score: 0.994865
input: The movie was meh.             : score: 0.319709
input: The movie was okish.           : score: 0.759805
input: The movie was terrible...      : score: 0.003196
```

### 参考网址

1. 利用TensorFlow Hub中的预处理模型简化BERT: https://mp.weixin.qq.com/s/8NOCheero3R6gqi9xjrCng
2. Classify text with BERT: https://tensorflow.google.cn/tutorials/text/classify_text_with_bert
3. Tensorflow Build from source: https://tensorflow.google.cn/install/source?hl=en#gpu_support_3
4. 【tensorflow】缺少libcudart.so.11.0和libcudnn.so.8解决方法: https://blog.csdn.net/qq_44703886/article/details/112393149
5. 入门必看！TensorFlow 2 安装指南: https://zhuanlan.zhihu.com/p/245525398?utm_source=wechat_session