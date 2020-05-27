import tensorflow as tf

from crf_layer import CRF


class BiLSTMCRF(tf.keras.Model):
    """BiLSTM+CRF模型。"""
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size):
        super(BiLSTMCRF, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.transition_params = None

        # layers
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_size, mask_zero=True)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_num, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)
        self.crf = CRF(label_size)

    # @tf.function
    def call(self, text, training=None):
        inputs = self.embedding(text)       # [B, seq_len, embed_size]
        inputs = self.dropout(inputs, training)     # [B, seq_len, embed_size]
        logits = self.dense(self.biLSTM(inputs))    # [B, seq_len, label_size]
        viterbi_output = self.crf(logits)   # [B, seq_len, label_size]

        return viterbi_output
