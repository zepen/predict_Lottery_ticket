# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import tensorflow as tf
from tensorflow_addons.text.crf import crf_decode, crf_log_likelihood

# 关闭eager模式
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class LstmWithCRFModel(object):
    """ lstm + crf解码模型
    """

    def __init__(self, batch_size, n_class, ball_num, w_size, embedding_size, words_size, hidden_size, layer_size):
        self._inputs = tf.keras.layers.Input(
            shape=(w_size, ball_num), batch_size=batch_size, name="inputs"
        )
        self._tag_indices = tf.keras.layers.Input(
            shape=(ball_num, ), batch_size=batch_size, dtype=tf.int32, name="tag_indices"
        )
        self._sequence_length = tf.keras.layers.Input(
            shape=(), batch_size=batch_size, dtype=tf.int32, name="sequence_length"
        )
        # 构建特征抽取
        embedding = tf.keras.layers.Embedding(words_size, embedding_size)(self._inputs)
        first_lstm = tf.convert_to_tensor(
            [tf.keras.layers.LSTM(hidden_size)(embedding[:, :, i, :]) for i in range(ball_num)]
        )
        first_lstm = tf.transpose(first_lstm, perm=[1, 0, 2])
        second_lstm = None
        for _ in range(layer_size):
            second_lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(first_lstm)
        self._outputs = tf.keras.layers.Dense(n_class)(second_lstm)
        # 构建损失函数
        self._log_likelihood, self._transition_params = crf_log_likelihood(
            self._outputs, self._tag_indices, self._sequence_length
        )
        self._loss = tf.reduce_sum(-self._log_likelihood)
        #  构建预测
        self._pred_sequence, self._viterbi_score = crf_decode(
            self._outputs, self._transition_params, self._sequence_length
        )

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def outputs(self):
        return self._outputs

    @property
    def transition_params(self):
        return self._transition_params

    @property
    def loss(self):
        return self._loss

    @property
    def pred_sequence(self):
        return self._pred_sequence


class SignalLstmModel(object):
    """ 单向lstm序列模型
    """

    def __init__(self, batch_size, n_class, w_size, embedding_size, hidden_size, outputs_size, layer_size):
        self._inputs = tf.keras.layers.Input(
            shape=(w_size, ), batch_size=batch_size, dtype=tf.int32, name="inputs"
        )
        self._tag_indices = tf.keras.layers.Input(
            shape=(n_class, ), batch_size=batch_size, dtype=tf.float32, name="tag_indices"
        )
        embedding = tf.keras.layers.Embedding(outputs_size, embedding_size)(self._inputs)
        lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(embedding)
        for _ in range(layer_size):
            lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)(lstm)
        final_lstm = tf.keras.layers.LSTM(hidden_size, recurrent_dropout=0.2)(lstm)
        self._outputs = tf.keras.layers.Dense(outputs_size, activation="softmax")(final_lstm)
        # 构建损失函数
        self._loss = - tf.reduce_sum(self._tag_indices * tf.math.log(self._outputs))
        # 预测结果
        self._pred_label = tf.argmax(self.outputs, axis=1)

    @property
    def inputs(self):
        return self._inputs

    @property
    def tag_indices(self):
        return self._tag_indices

    @property
    def outputs(self):
        return self._outputs

    @property
    def loss(self):
        return self._loss

    @property
    def pred_label(self):
        return self._pred_label
