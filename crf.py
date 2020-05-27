import tensorflow as tf


def crf_decode(potentials, transition_params, sequence_length):
    """解码TensorFlow中标记的最高评分序列。

    这是张量的函数。

    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.

    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """
    sequence_length = tf.cast(sequence_length, dtype=tf.int32)

    # 如果max_seq_len为1，则跳过算法，只返回argmax标记和max activation。
    def _single_seq_fn():
        squeezed_potentials = tf.squeeze(potentials, [1])
        decode_tags = tf.expand_dims(tf.argmax(squeezed_potentials, axis=1), 1)
        best_score = tf.reduce_max(squeezed_potentials, axis=1)
        return tf.cast(decode_tags, dtype=tf.int32), best_score

    def _multi_seq_fn():
        """最高得分序列的解码。"""
        # Computes forward decoding. Get last score and backpointers.
        initial_state = tf.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = tf.squeeze(initial_state, axis=[1])
        inputs = tf.slice(potentials, [0, 1, 0], [-1, -1, -1])

        sequence_length_less_one = tf.maximum(
            tf.constant(0, dtype=sequence_length.dtype), sequence_length - 1)

        backpointers, last_score = crf_decode_forward(
            inputs, initial_state, transition_params, sequence_length_less_one)

        backpointers = tf.reverse_sequence(
            backpointers, sequence_length_less_one, seq_axis=1)

        initial_state = tf.cast(tf.argmax(last_score, axis=1), dtype=tf.int32)
        initial_state = tf.expand_dims(initial_state, axis=-1)

        decode_tags = crf_decode_backward(backpointers, initial_state)
        decode_tags = tf.squeeze(decode_tags, axis=[2])
        decode_tags = tf.concat([initial_state, decode_tags], axis=1)
        decode_tags = tf.reverse_sequence(
            decode_tags, sequence_length, seq_axis=1)

        best_score = tf.reduce_max(last_score, axis=1)
        return decode_tags, best_score

    if potentials.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


def crf_decode_forward(inputs, state, transition_params, sequence_lengths):
    """计算线性链CRF中的正向解码。

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous step's
            score values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
      sequence_lengths: A [batch_size] vector of true sequence lengths.

    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    mask = tf.sequence_mask(sequence_lengths, tf.shape(inputs)[1])
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
    crf_fwd_layer = tf.keras.layers.RNN(
        crf_fwd_cell, return_sequences=True, return_state=True)
    return crf_fwd_layer(inputs, state, mask=mask)


def crf_decode_backward(inputs, state):
    """计算线性链CRF中的反向解码。

    Args:
      inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
      state: A [batch_size, 1] matrix of tag index of next step.

    Returns:
      new_tags: A [batch_size, num_tags] tensor containing the new tag indices.
    """
    inputs = tf.transpose(inputs, [1, 0, 2])

    def _scan_fn(state, inputs):
        state = tf.squeeze(state, axis=[1])
        idxs = tf.stack([tf.range(tf.shape(inputs)[0]), state], axis=1)
        new_tags = tf.expand_dims(tf.gather_nd(inputs, idxs), axis=-1)
        return new_tags

    return tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])


class CrfDecodeForwardRnnCell(tf.keras.layers.AbstractRNNCell):
    """计算线性链CRF中的正向解码。"""

    def __init__(self, transition_params, **kwargs):
        """初始化CrfDecodeForwardRnnCell。

        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. 这个矩阵将被扩展为[1, num_tags, num_tags]
            以在下面的cell中进行广播求和。
        """
        super(CrfDecodeForwardRnnCell, self).__init__(**kwargs)
        self._transition_params = tf.expand_dims(transition_params, 0)
        self._num_tags = transition_params.shape[0]

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def build(self, input_shape):
        super(CrfDecodeForwardRnnCell, self).build(input_shape)

    def call(self, inputs, state):
        """构建CrfDecodeForwardRnnCell。

        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.

        Returns:
          backpointers: A [batch_size, num_tags] matrix of backpointers.
          new_state: A [batch_size, num_tags] matrix of new score values.
        """
        state = tf.expand_dims(state[0], 2)
        transition_scores = state + self._transition_params
        new_state = inputs + tf.reduce_max(transition_scores, [1])
        backpointers = tf.argmax(transition_scores, 1)
        backpointers = tf.cast(backpointers, dtype=tf.int32)
        return backpointers, new_state


def crf_log_likelihood(inputs,
                       tag_indices,
                       sequence_lengths,
                       transition_params=None):
    """计算CRF中标记序列的对数似然。
    通过crf_sequence_score计算状态序列可能性分数，通过crf_log_norm计算归一化项。
    最后返回log_likelihood对数似然。

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the log-likelihood.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix,
          if available.
    Returns:
      log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
        each example, given the sequence of tag indices.
      transition_params: A [num_tags, num_tags] transition matrix. This is
          either provided by the caller or created in this function.
    """
    num_tags = inputs.shape[2]

    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    if transition_params is None:
        initializer = tf.keras.initializers.GlorotUniform()
        transition_params = tf.Variable(
            initializer([num_tags, num_tags]), "transitions")

    sequence_scores = crf_sequence_score(
        inputs, tag_indices, sequence_lengths, transition_params)
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # Normalize the scores to get the log-likelihood per example.
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    """计算标记序列的非标准化分数。
    通过crf_unary_score计算状态特征分数，通过crf_binary_score计算转移特征分数。

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which
          we compute the unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    # 如果max_seq_len为1，则跳过分数计算，只收集单个标记的unary potentials。
    def _single_seq_fn():
        batch_size = tf.shape(inputs, out_type=tag_indices.dtype)[0]

        example_inds = tf.reshape(
            tf.range(batch_size, dtype=tag_indices.dtype), [-1, 1])
        sequence_scores = tf.gather_nd(
            tf.squeeze(inputs, [1]),
            tf.concat([example_inds, tag_indices], axis=1))
        sequence_scores = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(sequence_scores),
            sequence_scores)
        return sequence_scores

    def _multi_seq_fn():
        # 计算给定标记序列的分数。
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(
            tag_indices, sequence_lengths, transition_params)
        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    if inputs.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


def crf_unary_score(tag_indices, sequence_lengths, inputs):
    """计算标记序列的状态特征分数。
    利用掩码的方式，计算得出一个类似交叉熵的值。

    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
    Returns:
      unary_scores: A [batch_size] vector of unary scores.
    """
    assert len(tag_indices.shape) == 2, "tag_indices: A [batch_size, max_seq_len] matrix of tag indices."
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    batch_size = tf.shape(inputs)[0]
    max_seq_len = tf.shape(inputs)[1]
    num_tags = tf.shape(inputs)[2]

    flattened_inputs = tf.reshape(inputs, [-1])

    offsets = tf.expand_dims(tf.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += tf.expand_dims(tf.range(max_seq_len) * num_tags, 0)
    # 根据标记索引的数据类型使用int32或int64。
    if tag_indices.dtype == tf.int64:
        offsets = tf.cast(offsets, tf.int64)
    flattened_tag_indices = tf.reshape(offsets + tag_indices, [-1])

    unary_scores = tf.reshape(
        tf.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)

    unary_scores = tf.reduce_sum(unary_scores * masks, 1)
    return unary_scores


def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """计算标记序列的转移特征分数。
    通过转移矩阵返回转移特征分数。

    Args:
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      binary_scores: A [batch_size] vector of binary scores.
    """
    tag_indices = tf.cast(tag_indices, dtype=tf.int32)
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    num_tags = tf.shape(transition_params)[0]
    num_transitions = tf.shape(tag_indices)[1] - 1

    # 在序列的每一侧截断一个，得到每个转换的开始和结束索引。
    start_tag_indices = tf.slice(tag_indices, [0, 0], [-1, num_transitions])
    end_tag_indices = tf.slice(tag_indices, [0, 1], [-1, num_transitions])

    # 将索引编码为扁平表示。
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = tf.reshape(transition_params, [-1])

    # 基于扁平化表示得到转移特征分数。
    binary_scores = tf.gather(
        flattened_transition_params, flattened_transition_indices)

    masks = tf.sequence_mask(
        sequence_lengths, maxlen=tf.shape(tag_indices)[1], dtype=tf.float32)
    truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
    binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


def crf_log_norm(inputs, sequence_lengths, transition_params):
    """计算CRF的标准化。

    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)
    # 分割第一个和其余的输入，为正向算法做准备。
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])

    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
    # the "initial state" (the unary potentials).
    def _single_seq_fn():
        log_norm = tf.reduce_logsumexp(first_input, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(log_norm),
            log_norm)
        return log_norm

    def _multi_seq_fn():
        """α值的正向计算。"""
        rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
        # 计算前向算法中的alpha值以得到分割函数。

        alphas = crf_forward(
            rest_of_input, first_input, transition_params, sequence_lengths)
        log_norm = tf.reduce_logsumexp(alphas, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0),
            tf.zeros_like(log_norm),
            log_norm)
        return log_norm

    if inputs.shape[1] == 1:
        return _single_seq_fn()
    else:
        return _multi_seq_fn()


def crf_forward(inputs, state, transition_params, sequence_lengths):
    """计算线性链CRF中的alpha值。

    Args:
      inputs: A [batch_size, num_tags] matrix of unary potentials.
      state: A [batch_size, num_tags] matrix containing the previous alpha
         values.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
          This matrix is expanded into a [1, num_tags, num_tags] in preparation
          for the broadcast summation occurring within the cell.
      sequence_lengths: A [batch_size] vector of true sequence lengths.

    Returns:
      new_alphas: A [batch_size, num_tags] matrix containing the
          new alpha values.
    """
    sequence_lengths = tf.cast(sequence_lengths, dtype=tf.int32)

    sequence_lengths = tf.maximum(
        tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 2)
    inputs = tf.transpose(inputs, [1, 0, 2])
    transition_params = tf.expand_dims(transition_params, 0)

    def _scan_fn(state, inputs):
        state = tf.expand_dims(state, 2)
        transition_scores = state + transition_params
        new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])
        return new_alphas

    all_alphas = tf.transpose(tf.scan(_scan_fn, inputs, state), [1, 0, 2])
    idxs = tf.stack(
        [tf.range(tf.shape(sequence_lengths)[0]), sequence_lengths], axis=1)
    return tf.gather_nd(all_alphas, idxs)
