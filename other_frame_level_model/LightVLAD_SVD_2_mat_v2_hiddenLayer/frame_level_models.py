# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math
import numpy as np 

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")

flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")

flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                    "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')

flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                   "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                   "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                   "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")


flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("SVD_dim", 256, 
                     "Dimension of SVD vector")

flags.DEFINE_bool("use_SVD", False, 
                  "Whether to use SVD for dim reduction. ")
flags.DEFINE_integer("CNN_filter_factor", 2, 
                     "Num of conv filters = cluster_size / CNN_filter_factor. ")
flags.DEFINE_integer("CNN_conv_strides", 1, 
                     "Num of strides for conv layer. ")

flags.DEFINE_integer("reduced_reshape_input_feature_size", 256, 
                     "Feature size after reduction")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")


flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                     "ReLU after the NetFV hidden layer.")


flags.DEFINE_bool("fv_couple_weights", True,
                     "Coupling cluster weights or not")
 
flags.DEFINE_float("fv_coupling_factor", 0.01,
                     "Coupling factor")


flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel_CG",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")



flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")
flags.DEFINE_integer("gru_cells_audio", 128, "Number of GRU cells (audio).")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                     "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                     "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")

flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")

flags.DEFINE_bool("attention_relu", True, "Attention relu")

flags.DEFINE_bool("use_attention_for_VLAD", False, "Attention for VLAD")

#python train.py --train_data_pattern=${HOME}/yt8m/v2/video/train*.tfrecord --model=NetVLADModelLF 
#--train_dir=~/yt8m/v2/models/frame/sample_model --frame_features=True 
#--feature_names="rgb,audio" --feature_sizes="1024,128" --batch_size=80 --base_learning_rate=0.0002 
#--netvlad_cluster_size=256 --netvlad_hidden_size=1024 --moe_l2=1e-6 --iterations=300 --learning_rate_decay=0.8 
#--netvlad_relu=False --gating=True --moe_prob_gating=True --max_step=700000

class NetVLADModelLF(models.BaseModel):
  """Creates a NetVLAD based model.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    relu = FLAGS.netvlad_relu
    dimred = FLAGS.netvlad_dimred
    gating = FLAGS.gating
    remove_diag = FLAGS.gating_remove_diag
    lightvlad = FLAGS.lightvlad
    vlagd = FLAGS.vlagd
    SVD_dim = FLAGS.SVD_dim

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)

    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    if lightvlad:
      video_NetVLAD = LightVLAD(1024,max_frames,int(cluster_size), add_batch_norm, is_training)
      audio_NetVLAD = LightVLAD(128,max_frames,int(cluster_size/2), add_batch_norm, is_training)
    elif vlagd:
      video_NetVLAD = NetVLAGD(1024,max_frames,int(cluster_size), add_batch_norm, is_training)
      audio_NetVLAD = NetVLAGD(128,max_frames,int(cluster_size/2), add_batch_norm, is_training)
    else:
      video_NetVLAD = NetVLAD(1024,max_frames,int(cluster_size), add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD(128,max_frames,int(cluster_size/2), add_batch_norm, is_training)

    if add_batch_norm:# and not lightvlad:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

    vlad = tf.concat([vlad_video, vlad_audio],1) # None x vlad_dim

    vlad_dim = vlad.get_shape().as_list()[1]

    ##### simplier SVD #####
    SVD_mat1 = tf.get_variable("hidden1_weights",
      [vlad_dim, SVD_dim],
      initializer=tf.glorot_uniform_initializer())

    SVD_mat2 = tf.get_variable("hidden2_weights",
      [SVD_dim, int(hidden1_size*2)],
      initializer=tf.glorot_uniform_initializer())

    SVD_mat1_biases = tf.get_variable("SVD_mat1_biases",
      [SVD_dim],
      initializer = tf.random_normal_initializer(stddev=0.01))

    SVD_mat2_biases = tf.get_variable("SVD_mat2_biases",
      [int(hidden1_size*2)],
      initializer = tf.random_normal_initializer(stddev=0.01))
    ##### simplier SVD #####

    activation = tf.matmul(vlad, SVD_mat1) # None x 256
    activation += SVD_mat1_biases
    activation = tf.matmul(activation, SVD_mat2) # None x 2*hidden1_size
    activation += SVD_mat2_biases
    tf.summary.histogram("activation_before_bn", activation)

    activation = slim.batch_norm(
        activation,
        center=True,
        scale=True,
        is_training=is_training,
        scope="hidden1_bn")
    tf.summary.histogram("activation_after_bn", activation)

    ## gating part 
    gating_weights = tf.get_variable("gating_weights_2",
      [int(2*hidden1_size), hidden1_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size)))
    
    gates = tf.matmul(activation, gating_weights)

    gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=is_training,
        scope="gating_bn")

    gates = tf.sigmoid(gates)
    tf.summary.histogram("gates_layer", gates)
    ## gating part 

    ## hidden layer 
    activation = tf.nn.tanh(activation)
    tf.summary.histogram("activation_after_bn_after_1_tanh", activation)

    activation_hidden_weights = tf.get_variable("activation_hidden_weights",
                                               [int(hidden1_size*2), hidden1_size],
                                               initializer=tf.glorot_uniform_initializer())

    activation = tf.matmul(activation, activation_hidden_weights) 
    activation = slim.batch_norm(
        activation,
        center=True,
        scale=True,
        is_training=is_training,
        scope="hidden_layer_bn")
    tf.summary.histogram("activation_after_bn_after_1_tanh_after_bn", activation)

    activation = tf.nn.tanh(activation)
    tf.summary.histogram("activation_after_bn_after_1_tanh_after_bn_after_2_tanh", activation)

    activation = tf.multiply(activation, gates)
    tf.summary.histogram("activation_right_before_video", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        **unused_params)

class LightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights) # None (None * max_frames) x cluster_size 
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size]) # None x max_frames x cluster_size
       
        activation = tf.transpose(activation,perm=[0,2,1]) # None x cluster_size x max_frame
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size]) # None x max_frame x feature_size
        vlad = tf.matmul(activation,reshaped_input)        # None x cluster_size x feature_size
    
        vlad = tf.transpose(vlad,perm=[0,2,1]) # None x feature_size x cluster_size 
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size]) # None x (feature_size x cluster_size)
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class NetVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True) # None x 1 x cluster_size

        cluster_weights2 = tf.get_variable("cluster_weights2",
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size))) # 1 x feature_size x cluster_size 
        
        a = tf.multiply(a_sum,cluster_weights2) # None x feature_size x cluster_size
        
        activation = tf.transpose(activation,perm=[0,2,1]) # None x cluster_size x max_frame
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size]) # None x max_frame x feature_size
        vlad = tf.matmul(activation,reshaped_input) # None x cluster_size x feature_size
        vlad = tf.transpose(vlad,perm=[0,2,1]) # None x feature_size x cluster_size 
        vlad = tf.subtract(vlad,a)
        
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad

class NetVLAGD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
        activation = tf.matmul(reshaped_input, cluster_weights)
        
        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn")
        else:
          cluster_biases = tf.get_variable("cluster_biases",
            [cluster_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        gate_weights = tf.get_variable("gate_weights",
            [1, self.cluster_size,self.feature_size],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
        
        gate_weights = tf.sigmoid(gate_weights)

        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])

        vlagd = tf.matmul(activation,reshaped_input)
        vlagd = tf.multiply(vlagd,gate_weights)

        vlagd = tf.transpose(vlagd,perm=[0,2,1])
        
        vlagd = tf.nn.l2_normalize(vlagd,1)

        vlagd = tf.reshape(vlagd,[-1,self.cluster_size*self.feature_size])
        vlagd = tf.nn.l2_normalize(vlagd,1)

        return vlagd


