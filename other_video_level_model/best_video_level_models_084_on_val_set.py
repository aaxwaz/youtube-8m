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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_float(
    "moe_l2", 1e-6,
    "L2 penalty for MoeModel.")
flags.DEFINE_integer(
    "moe_low_rank_gating", -1,
    "Low rank gating for MoeModel.")
flags.DEFINE_bool(
    "moe_prob_gating", True,
    "Prob gating for MoeModel.")
flags.DEFINE_string(
    "moe_prob_gating_input", "other",
    "input Prob gating for MoeModel.")
flags.DEFINE_bool(
    "vlad_gating", True,
    "Gating for vlad part. ")

#flags.DEFINE_bool("gating_remove_diag", False,
#                  "Remove diag for self gating")

class MOE_Label_Embedding(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   embeddings = None, 
                   is_training=True,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.
     It also includes the possibility of gating the probabilities
     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.
    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      is_training: Is this the training phase ?
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    low_rank_gating = FLAGS.moe_low_rank_gating
    l2_penalty = FLAGS.moe_l2;
    gating_probabilities = FLAGS.moe_prob_gating
    gating_input = FLAGS.moe_prob_gating_input

    input_size = model_input.get_shape().as_list()[1]
    remove_diag = FLAGS.gating_remove_diag

    expert_dim   = 1500
    expert_dim_1 = 1024
    expert_dim_2 = 512

    if is_training:
      label_embeddings = tf.get_variable("embeddings",
                                   [256, vocab_size],
                                   #initializer = tf.constant_initializer(embeddings))
                                   initializer = tf.glorot_uniform_initializer())
    else:
      label_embeddings = tf.get_variable("embeddings",
                                   [256, vocab_size],
                                   initializer = tf.glorot_uniform_initializer())
    
    label_embeddings = tf.nn.l2_normalize(label_embeddings,0)

    project_mat = tf.get_variable("project_mat",
                                 [expert_dim_2, 256], 
                                 initializer = tf.glorot_uniform_initializer())

    gate_activations = slim.fully_connected(
        model_input,
        expert_dim * num_mixtures,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    expert_activations = slim.fully_connected(
        model_input,
        expert_dim * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.tanh(tf.reshape(
        gate_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x (num_mixtures + 1)

    expert_distribution = tf.nn.tanh(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    probabilities_by_class_and_batch = tf.reduce_mean(
        gating_distribution * expert_distribution, 1)

    output = tf.reshape(probabilities_by_class_and_batch,
                                     [-1, expert_dim])
    output = slim.batch_norm(
        output,
        center=True,
        scale=True,
        is_training=is_training,
        scope="output_bn_1")

    output = tf.nn.tanh(output)

    ### Gating 1 
    gate_activations0 = slim.fully_connected(
        model_input,
        expert_dim,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gating_layer_0")
    
    gates0 = slim.batch_norm( # MUST have this bn!
          gate_activations0,
          center=True,
          scale=True,
          is_training=is_training,
          scope="gating_prob_bn_0")

    gates0 = tf.sigmoid(gates0)
    output = tf.multiply(output, gates0)
    ### Gating 1 

    output = slim.fully_connected(
        output,
        expert_dim_1,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="output_layer_2")

    output = slim.batch_norm(
        output,
        center=True,
        scale=True,
        is_training=is_training,
        scope="output_bn_2")

    output = tf.nn.tanh(output)
    ### Gating 2 
    gate_activations1 = slim.fully_connected(
        model_input,
        expert_dim_1,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gating_layer_1")
    
    gates1 = slim.batch_norm( # MUST have this bn!
          gate_activations1,
          center=True,
          scale=True,
          is_training=is_training,
          scope="gating_prob_bn_1")

    gates1 = tf.sigmoid(gates1)
    output = tf.multiply(output, gates1)
    ### Gating 2 

    output = slim.fully_connected(
        output,
        expert_dim_2,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="output_layer")

    output = slim.batch_norm(
        output,
        center=True,
        scale=True,
        is_training=is_training,
        scope="output_bn")

    output = tf.nn.tanh(output)

    ## project into embeddings space 
    output = tf.matmul(output, project_mat)
    output = tf.matmul(output, label_embeddings)
    probabilities = tf.nn.sigmoid(output)

    ### GATING A MUST 
    gate_activations = slim.fully_connected(
        model_input,
        vocab_size,
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gating_layer")
    
    gates = slim.batch_norm( # MUST have this bn!
          gate_activations,
          center=True,
          scale=True,
          is_training=is_training,
          scope="gating_prob_bn")

    gates = tf.sigmoid(gates)

    probabilities = tf.multiply(probabilities, gates)

    return {"predictions": probabilities}
