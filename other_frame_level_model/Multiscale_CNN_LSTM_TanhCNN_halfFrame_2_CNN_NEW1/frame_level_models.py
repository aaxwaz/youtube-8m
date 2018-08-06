import math
import models
import tensorflow as tf
import numpy as np
import utils
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS
import video_level_models

flags.DEFINE_string("video_level_classifier_model", "MoeModel_CG",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")

class MultiscaleCnnLstmModel(models.BaseModel):

  def cnn(self, 
          model_input, 
          l2_penalty=1e-8, 
          num_filters = [1024, 1024],
          filter_sizes = [1,3], 
          sub_scope="",
          **unused_params):
    max_frames = model_input.get_shape().as_list()[1]
    num_features = model_input.get_shape().as_list()[2]

    shift_inputs = []
    for i in range(max(filter_sizes)):
      if i == 0:
        shift_inputs.append(model_input)
      else:
        shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

    cnn_outputs = []
    for nf, fs in zip(num_filters, filter_sizes):
      sub_input = tf.concat(shift_inputs[:fs], axis=2)
      sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs, 
                       shape=[num_features*fs, nf], dtype=tf.float32, 
                       initializer=tf.glorot_uniform_initializer(), 
                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
      cnn_outputs.append(tf.nn.l2_normalize(tf.einsum("ijk,kl->ijl", sub_input, sub_filter),2)) # each one has None x max_frames x nf 

    cnn_output = tf.concat(cnn_outputs, axis=2)
    cnn_output = tf.nn.l2_normalize(cnn_output,2)
    cnn_output = tf.nn.tanh(cnn_output)
    return cnn_output

  def rnn(self, model_input, lstm_size, num_frames, 
          sub_scope="", **unused_params):

    cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, 
                                        state_is_tuple=True)
    with tf.variable_scope("RNN-"+sub_scope):
      outputs, state = tf.nn.dynamic_rnn(cell, model_input,
                                         sequence_length=num_frames,
                                         swap_memory=True,
                                         dtype=tf.float32)
    # return final memory
    return tf.concat(state.c, 1)

  def create_model(self, model_input, vocab_size, num_frames, is_training=True,
                   l2_penalty=1e-8, **unused_params):

    num_layers = 3
    lstm_size = 900
    activation_proj_dim = int(lstm_size*1.18)
    pool_size=2
    num_filters=[128,128]
    filter_sizes=[1,3]
    features_size = int(sum(num_filters))
    self.is_training=is_training

    cnn_input = model_input

    cnn_max_frames = model_input.get_shape().as_list()[1]

    lstm_memories = []

    for layer in range(num_layers):

      if layer > 0:
        cnn_output = self.cnn(cnn_input, num_filters=num_filters, filter_sizes=filter_sizes, sub_scope="cnn%d"%(layer+1))
        tf.summary.histogram("cnn_output_{}".format(layer), cnn_output)

        cnn_output = slim.batch_norm(
          cnn_output,
          center=True,
          scale=True,
          is_training=self.is_training,
          scope="cnn_output_bn_layer_"+str(layer))
        tf.summary.histogram("cnn_output_after_bn_before_tanh_{}".format(layer), cnn_output)
      else:
        cnn_output = slim.batch_norm(
          cnn_input,
          center=True,
          scale=True,
          is_training=self.is_training,
          scope="cnn_output_bn_layer_"+str(layer))
        tf.summary.histogram("cnn_output_after_bn_before_tanh_{}".format(layer), cnn_output)

      cnn_output_tanh = tf.nn.tanh(cnn_output)
      tf.summary.histogram("cnn_output_after_bn_after_tanh_{}".format(layer), cnn_output_tanh)

      lstm_memory = self.rnn(cnn_output_tanh, lstm_size, num_frames, sub_scope="rnn%d"%(layer+1)) # None x lstm_size
      tf.summary.histogram("lstm_memory_{}".format(layer), lstm_memory)

      lstm_memory = tf.nn.l2_normalize(lstm_memory,1)
      tf.summary.histogram("lstm_memory_after_l2Norm_{}".format(layer), lstm_memory)

      lstm_memories.append(lstm_memory)

      max_pooled_cnn_output = tf.layers.max_pooling1d(cnn_output_tanh, pool_size=3, strides=2, padding='same')

      # for the next cnn layer
      cnn_input = max_pooled_cnn_output
      num_frames = tf.maximum(num_frames/pool_size, 1)

    concat_lstm_memory = tf.concat(lstm_memories, 1)
    concat_lstm_memory = tf.nn.l2_normalize(concat_lstm_memory, 1)
    print("\n\n\nconcat_lstm_memory size: {} \n\n\n".format(concat_lstm_memory.get_shape()))

    vlad_dim = concat_lstm_memory.get_shape().as_list()[1]

    concat_lstm_memory_weights = tf.get_variable("concat_lstm_memory_weights",
      [vlad_dim, activation_proj_dim],
      initializer=tf.glorot_uniform_initializer())

    activation = tf.matmul(concat_lstm_memory, concat_lstm_memory_weights) # None x lstm_size

    concat_lstm_memory_biases = tf.get_variable("concat_lstm_memory_biases",
      [activation_proj_dim],
      initializer = tf.random_normal_initializer(stddev=0.01))
    activation += concat_lstm_memory_biases

    ## gating 
    gating_weights = tf.get_variable("gating_weights_2",
      [activation_proj_dim, activation_proj_dim],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(activation_proj_dim)))
    
    gates = tf.matmul(activation, gating_weights)

    gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=self.is_training,
        scope="activation_gating_bn")

    gates = tf.sigmoid(gates)

    activation = tf.multiply(activation,gates)
    tf.summary.histogram("activation_before_video_model", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=self.is_training,
        **unused_params)    
















