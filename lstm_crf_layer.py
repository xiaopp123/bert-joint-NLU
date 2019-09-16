import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class BLSTM_CRF(object):
    def __init__(self, config):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """

        self.hidden_unit = config["hidden_unit"]
        self.dropout_rate = config["dropout_rate"]
        self.cell_type = config["cell_type"]
        self.num_layers = config["num_layers"]
        self.embedded_chars = config["embedded_chars"]
        self.initializers = config["initializers"]
        self.seq_length = config["seq_length"]
        self.num_labels = config["num_labels"]
        self.labels = config["labels"]
        self.lengths = config["lengths"]
        self.embedding_dims = self.embedded_chars.shape[-1].value
        self.is_training = config["is_training"]

    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf
        """
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            #blstm
            lstm_output = self.blstm_layer(self.embedded_chars)
            #project
            logits = self.project_bilstm_layer(lstm_output)

        #crf
        loss, trans = self.crf_layer(logits)
        print(self.labels)
        #
        # CRF decode, pred_ids 是一条最大概率的标注路径
        if self.is_training:
            return (loss, logits, trans, None)

        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        
        return (None, logits, None, pred_ids)

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],\
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,\
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,\
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            if self.labels is None or self.is_training == False:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)

                return tf.reduce_mean(-log_likelihood), trans

    def blstm_layer(self, embedding_chars):
        with tf.variable_scope("rnn_layer"):
            cell_fw = rnn.LSTMCell(self.hidden_unit)
            cell_bw = rnn.LSTMCell(self.hidden_unit)
            if self.is_training and self.dropout_rate is not None:
                cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
                cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars, dtype=tf.float32)

            outputs = tf.concat(outputs, axis=2)

        return outputs

    def project_bilstm_layer(self, lstm_outputs, name = None):
        '''

        '''
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.num_labels],\
                        dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,\
                        initializer=tf.zeros_initializer)

                outputs = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])

                pred = tf.nn.xw_plus_b(outputs, W, b)

            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])
            '''
            with tf.variable_scope("hidden"):
                W1 = tf.get_variable("W1", shape=[self.hidden_unit * 2, self.hidden_unit],\
                        dtype=tf.float32, initializer=self.initializers.xavier_initializer())
                b1 = tf.get_variable("b1", shape=[self.hidden_unit], dtype=tf.float32,\
                        initializer=tf.zeros_initializer)
                output = tf.reshape(lstm_outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W1, b1))
            with tf.variable_scope("logits"):
                W2 = tf.get_variable("W2", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b2 = tf.get_variable("b2", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer)

                pred = tf.nn.xw_plus_b(hidden, W2, b2)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])
            '''
