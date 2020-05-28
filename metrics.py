from abc import ABC, abstractmethod

import tensorflow as tf

from conlleval import CoNLLEval


class ChunkingF1(tf.keras.metrics.Metric, ABC):
    """分块F1。"""
    def __init__(self, tag_vocab, from_logits=True, name="f1", dtype=None,
                 **kwargs):
        super().__init__(name, dtype, dynamic=True, **kwargs)
        self.tag_vocab = tag_vocab
        self.from_logits = from_logits

    def update_the_state(self, y_true, y_pred, sample_weight=None, **kwargs):
        mask = y_pred._keras_mask if sample_weight is None else sample_weight
        assert mask is not None, "ChunkingF1 requires masking, " \
                                 "check your _keras_mask or compute_mask"
        if self.from_logits:
            y_true = tf.argmax(y_true, axis=-1)
            y_pred = tf.argmax(y_pred, axis=-1)
        y_true = self.to_tags(y_true, mask)
        y_pred = self.to_tags(y_pred, mask)
        return self.update_tags(y_true, y_pred)

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        return self.update_the_state(y_true, y_pred, sample_weight)

    def update_state(self, y_true, y_pred, sample_weight=None, **kwargs):
        pass

    def to_tags(self, y, sample_weight):
        batch = list()
        y = y.numpy()
        sample_weight = sample_weight.numpy()
        for sent, mask in zip(y, sample_weight):
            tags = list()
            for tag, m in zip(sent, mask):
                if not m:
                    continue
                tags.append(self.tag_vocab[tag])
            batch.append(tags)
        return batch

    @abstractmethod
    def update_tags(self, true_tags, pred_tags):
        pass

    @abstractmethod
    def result(self):
        pass


class IOBESF1(ChunkingF1):
    """IOBES标注标准下的F1。"""
    def __init__(self, tag_vocab, from_logits=True, name="f1", dtype=None,
                 **kwargs):
        super().__init__(tag_vocab, from_logits, name, dtype, **kwargs)
        self.state = CoNLLEval()

    def update_tags(self, true_tags, pred_tags):
        for gold, pred in zip(true_tags, pred_tags):
            self.state.update_state(gold, pred)
        return self.result()

    def result(self):
        return self.state.result(full=False, verbose=False).fscore

    def reset_states(self):
        self.state.reset_state()
