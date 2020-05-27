import tensorflow as tf

from crf import crf_decode, crf_log_likelihood


class CRF(tf.keras.layers.Layer):
    """
    条件随机场层 (tf.keras)
    CRF可以用作网络的最后一层（作为分类器使用）。
    输入形状（特征）必须等于CRF可以预测的类数（建议在线性层后接CRF层）。

    Args:
        num_classes (int): 标签（类别）的数量。

    Input shape:
        (batch_size, sentence length, num_classes)。

    Output shape:
        (batch_size, sentence length, num_classes)。
    """

    def __init__(self, num_classes, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.transitions = None
        self.output_dim = int(num_classes)  # 输出维度
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3)
        self.supports_masking = False

    def get_config(self):
        config = {
            "output_dim": self.output_dim,
            "supports_masking": self.supports_masking,
            "transitions": tf.keras.backend.eval(self.transitions)
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) == 3
        f_shape = tf.TensorShape(input_shape)
        input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: f_shape[-1]})

        if f_shape[-1] is None:
            raise ValueError(
                "The last dimension of the inputs to `CRF` "
                "should be defined. Found `None`.")
        if f_shape[-1] != self.output_dim:
            raise ValueError(
                "The last dimension of the input shape must be equal to output"
                " shape. Use a linear layer if needed.")
        self.input_spec = input_spec
        self.transitions = self.add_weight(
            name="transitions",
            shape=[self.output_dim, self.output_dim],
            initializer="glorot_uniform",
            trainable=True)
        self.built = True

    def compute_mask(self, inputs, mask=None):
        # 只需将接收到的mask从上一层传递到下一层，或者在该层更改输入的形状时对其进行操作
        return mask

    def call(self, inputs, sequence_lengths=None, mask=None, training=None,
             **kwargs):
        sequences = tf.convert_to_tensor(inputs, dtype=self.dtype)
        if sequence_lengths is not None:
            assert len(sequence_lengths.shape) == 2
            assert tf.convert_to_tensor(sequence_lengths).dtype == "int32"
            seq_len_shape = tf.convert_to_tensor(
                sequence_lengths).get_shape().as_list()
            assert seq_len_shape[1] == 1
            sequence_lengths = tf.keras.backend.flatten(sequence_lengths)
        else:
            sequence_lengths = tf.math.count_nonzero(mask, axis=1)

        viterbi_sequence, _ = crf_decode(
            sequences, self.transitions, sequence_lengths)
        output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
        return tf.keras.backend.in_train_phase(sequences, output)

    def compute_output_shape(self, input_shape):
        tf.TensorShape(input_shape).assert_has_rank(3)
        return input_shape[:2] + (self.output_dim,)

    @property
    def viterbi_accuracy(self):
        def accuracy(y_true, y_pred):
            shape = tf.shape(y_pred)
            sequence_lengths = tf.ones(shape[0], dtype=tf.int32) * (shape[1])
            viterbi_sequence, _ = crf_decode(
                y_pred, self.transitions, sequence_lengths)
            output = tf.keras.backend.one_hot(viterbi_sequence, self.output_dim)
            return tf.keras.metrics.categorical_accuracy(y_true, output)

        accuracy.func_name = "viterbi_accuracy"
        return accuracy


class CRFLoss(object):
    """CRF损失函数。"""
    def __init__(self, crf: CRF, dtype) -> None:
        super().__init__()
        self.crf = crf
        self.dtype = dtype

    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        assert sample_weight is not None, "your model has to support masking"
        if len(y_true.shape) == 3:
            y_true = tf.argmax(y_true, axis=-1)
        sequence_lengths = tf.math.count_nonzero(sample_weight, axis=1)
        y_pred = tf.convert_to_tensor(y_pred, dtype=self.dtype)
        log_likelihood, self.crf.transitions = crf_log_likelihood(
            y_pred,
            tf.cast(y_true, dtype=tf.int32),
            sequence_lengths,
            transition_params=self.crf.transitions)
        return tf.reduce_mean(-log_likelihood)
