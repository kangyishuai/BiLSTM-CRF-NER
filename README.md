# BiLSTM-CRF命名实体识别模型

## 概述

使用TensorFlow2.1.0实现BiLSTM-CRF命名实体识别模型。

其中，CRF通过继承`tf.keras.layers.Layer`来实现，使用方法与其他Keras网络层类似。

另外，还实现了对应的度量指标F1-Score，即`metrics.py`文件下的`IOBESF1`类，使用方法与Keras中其他度量指标类似。

## 其他

数据集：msra_ner，完整数据集[下载](http://file.hankcs.com/corpus/msra_ner.zip)

CRF的基本原理参考：
* [最通俗易懂的BiLSTM-CRF模型中的CRF层介绍](https://zhuanlan.zhihu.com/p/44042528)
* [NLP硬核入门-条件随机场CRF](https://zhuanlan.zhihu.com/p/87638630)
