import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import data

from crf_layer import CRFLoss
from model import BiLSTMCRF
from utils import build_vocab, read_vocab, tokenize, format_result

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'    # 只显示 Error

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train(config, params):
    """模型训练。"""
    # 构建词典
    if not (os.path.exists(config["vocab_file"]) and
            os.path.exists(config["tag_file"])):
        build_vocab(
            config["train_path"], config["vocab_file"], config["tag_file"])

    # 读取词典
    vocab2id, id2vocab = read_vocab(config["vocab_file"])
    tag2id, id2tag = read_vocab(config["tag_file"])
    # 数据预处理
    train_text, train_label = tokenize(config["train_path"], vocab2id, tag2id)
    dev_text, dev_label = tokenize(config["dev_path"], vocab2id, tag2id)

    # 将数据转换为tf.data.Dataset
    train_dataset = data.Dataset.from_tensor_slices((train_text, train_label))
    train_dataset = train_dataset.shuffle(len(train_text)).batch(
        params["batch_size"], drop_remainder=True)

    dev_dataset = data.Dataset.from_tensor_slices((dev_text, dev_label))
    dev_dataset = dev_dataset.batch(params["batch_size"], drop_remainder=True)

    print(f"hidden_num:{params['hidden_num']}, vocab_size:{len(vocab2id)}, "
          f"label_size:{len(tag2id)}")

    # 构建模型
    model = BiLSTMCRF(
        hidden_num=params["hidden_num"], vocab_size=len(vocab2id),
        label_size=len(tag2id), embedding_size=params["embedding_size"])

    # 编译模型
    model.compile(
        loss=CRFLoss(model.crf, model.dtype),
        optimizer=tf.keras.optimizers.Adam(params["lr"]),
        metrics=[model.crf.viterbi_accuracy],
        run_eagerly=True)
    model.build((None, train_text.shape[-1]))
    model.summary()

    # 设置回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config["ckpt_path"],
            save_weights_only=True,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"),
    ]

    # 训练（拟合）模型
    model.fit(
        train_dataset,
        epochs=params["epochs"],
        callbacks=callbacks,
        validation_data=dev_dataset)


def predict(text, config, params):
    # 读取词典
    vocab2id, id2vocab = read_vocab(config["vocab_file"])
    tag2id, id2tag = read_vocab(config["tag_file"])

    # 构建模型
    model = BiLSTMCRF(
        hidden_num=params["hidden_num"], vocab_size=len(vocab2id),
        label_size=len(tag2id), embedding_size=params["embedding_size"])
    model.load_weights(config["ckpt_path"])

    # 数据预处理
    dataset = tf.keras.preprocessing.sequence.pad_sequences(
        [[vocab2id.get(char, 0) for char in text]], padding='post')

    # 模型预测
    result = model.predict(dataset)[0]
    result = np.argmax(result, axis=-1)
    result = [id2tag[i] for i in result]
    print(result)
    # 结果处理
    entities_result = format_result(list(text), result)
    print(json.dumps(entities_result, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    config = {
        "train_path": "data/train.tsv",
        "dev_path": "data/dev.tsv",
        "vocab_file": "data/vocab.txt",
        "tag_file": "data/tags.txt",
        "ckpt_path": "checkpoints/ckpt_best"
    }

    params = {
        "batch_size": 256,
        "hidden_num": 128,
        "embedding_size": 128,
        "lr": 1e-3,
        "epochs": 3
    }

    text = "中共中央致中国致公党十一大的贺词"

    # train(config, params)
    predict(text, config, params)
