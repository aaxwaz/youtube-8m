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
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import glob
import json
import os
import time

import eval_util
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils
import numpy as np 
import pandas as pd 
from scipy.stats import hmean
from copy import copy


def evaluate(preds, labels, loss):
    evl_metrics = eval_util.EvaluationMetrics(3862, 20)

    evl_metrics.clear()
    iteration_info_dict = evl_metrics.accumulate(preds,
                                                 labels, loss)
    # calculate the metrics for the entire epoch
    epoch_info_dict = evl_metrics.get()
    return epoch_info_dict['gap']

dir = "/Users/weimin/Desktop/yt8m/youtube-8m/ensemble/final_1GB_ensemble/FINAL_6_EACH/"
labels = pd.read_csv('/Users/weimin/Desktop/yt8m/youtube-8m/ensemble/new_val_labels.csv')
models =  [#'Multiscale_CNN_LSTM_TanhCNN_166MB/eval_predictions.csv', \
           #'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_CNN_layers/eval_predictions.csv', \
           #'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_loops/eval_predictions.csv', \
           #'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_CNN_NEW1/eval_predictions.csv', \
           #'166_Two_world_one_layer/eval_predictions.csv', \
           #'166_LightVLAD_withVLADAddedOnLearnWeights/eval_predictions.csv', \
           #'256_LightVLAD_SVD_2_mat_v2_hiddenLayer/eval_predictions.csv', \
           #'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_loops/eval_predictions.csv', 
           #'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_CNN_NEW1/eval_predictions.csv',
           #'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_CNN_layers/eval_predictions.csv', 
           #'prev_LightVLAD_withVLADAddedOn/eval_predictions.csv',
           #'lightVLAD/eval_predictions.csv',
           #'prev_LightVLAD_SVD_2_mat_v2_hiddenLayer/eval_predictions.csv']
           #'FIVE_MODELS/Multiscale_CNN_LSTM_TanhCNN_166MB/eval_predictions.csv',
           ##'FIVE_MODELS/Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_loops/eval_predictions.csv',
           #'eval_predictions_full_nonRev.csv',
           #'eval_predictions_full_Rev.csv',
           #'eval_predictions_halfSeq_0_nonRev.csv', 
           #'eval_predictions_halfSeq_0_Rev.csv', 
           #'prev_models/LightVLAD_SVD_2_mat_v2_hiddenLayer/eval_predictions.csv', 
           #'prev_models/LightVLAD_withVLADAddedOn/eval_predictions.csv',
           ##'compare_ensemble_effects_for_5_instance/yt1_testing_rev_input_only/eval_predictions_full_rev.csv', 
           #'compare_ensemble_effects_for_5_instance/yt1_testing_rev_input_only/eval_predictions_full.csv', 

           #'compare_ensemble_effects_for_5_instance/tf_perm_4_val0863/eval_predictions_full_forward.csv', 
           #'compare_ensemble_effects_for_5_instance/tf_perm_4_val0863/eval_predictions_full_rev.csv', 
           #'compare_ensemble_effects_for_5_instance/tf_perm_4_val0863/eval_predictions_half_0_nonRev.csv', 
           #'compare_ensemble_effects_for_5_instance/tf_perm_4_val0863/eval_predictions_half_0_Rev.csv', 
           #'compare_ensemble_effects_for_5_instance/tf_perm_4_val0863/eval_predictions_half_1_nonRev.csv', 
           #'compare_ensemble_effects_for_5_instance/tf_perm_4_val0863/eval_predictions_half_1_rev.csv', 

           #'compare_ensemble_effects_for_5_instance/yt4_perm_4_val0862/eval_predictions_full_forward.csv', 
           #'compare_ensemble_effects_for_5_instance/yt4_perm_4_val0862/eval_predictions_full_rev.csv', 
           #'compare_ensemble_effects_for_5_instance/yt4_perm_4_val0862/eval_predictions_half_0_nonRev.csv', 
           #'compare_ensemble_effects_for_5_instance/yt4_perm_4_val0862/eval_predictions_half_0_Rev.csv', 
           #'compare_ensemble_effects_for_5_instance/yt4_perm_4_val0862/eval_predictions_half_1_nonRev.csv', 
           #'compare_ensemble_effects_for_5_instance/yt4_perm_4_val0862/eval_predictions_half_1_Rev.csv', 
           'LightVLAD_SVD_2_mat_v2_hiddenLayer/eval_predictions.csv',
           'FV_fv1Only_SVDMidTanh_hiddenLayer/eval_predictions.csv',
           'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_loops/eval_predictions.csv', 
           'prev_FIVE_Multiscale_CNN_LSTM_TanhCNN_166MB/eval_predictions.csv', 
           'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_CNN_NEW1/eval_predictions.csv']

          #'LightVLAD_SVD_2_mat_v2_hiddenLayer/eval_predictions.csv', 'multiscale_CNN_LSTM_concatLstmMemoriesOnly/eval_predictions.csv', \
          #'VLAD_SVD_2_mat_hiddenLayer/eval_predictions.csv']

dfs = []

for model in models:
  temp_model = os.path.join(dir, model)
  df = pd.read_csv(temp_model)
  cols = df.columns.tolist()
  cols[0] = 'id'
  df.columns = cols 
  df.set_index('id', inplace = True)

  df = df.ix[labels['id']]

  new_cols = [col for col in df.columns if 'labels' not in col]

  df = df[new_cols]

  dfs.append(df)

fake_loss = np.zeros((dfs[0].shape[0], 1))

print("Individual GAPs: ")
for df, md in zip(dfs, models):
  print(md.split('//')[0], evaluate(df.values, labels.iloc[:, 1:].values, fake_loss))

weights = [1.3,1.3,1.1,0.8,1.1]
df_final = sum([df.values*wt for df, wt in zip(dfs, weights)])
print("Ensemble GAP: ", evaluate(df_final, labels.iloc[:, 1:].values, fake_loss))




## choose 3 
temp_weights = [0] * len(weights)
for i1 in range(len(temp_weights)):
  for i2 in range(i1+1, len(temp_weights)):
    for i3 in range(i2+1, len(temp_weights)):
      temp_temp_weights = copy(temp_weights)
      temp_temp_weights[i1] = 1
      temp_temp_weights[i2] = 1
      temp_temp_weights[i3] = 1
      df_final = sum([df.iloc[:, 3862:].values*wt for df, wt in zip(dfs, temp_temp_weights)])
      print("{}: Ensemble GAP: ".format(temp_temp_weights), evaluate(df_final, labels, fake_loss))


## choose 4 
temp_weights = [0] * len(weights)
for i1 in range(len(temp_weights)):
  for i2 in range(i1+1, len(temp_weights)):
    for i3 in range(i2+1, len(temp_weights)):
      for i4 in range(i3+1, len(temp_weights)):
        temp_temp_weights = copy(temp_weights)
        temp_temp_weights[i1] = 1
        temp_temp_weights[i2] = 1
        temp_temp_weights[i3] = 1
        temp_temp_weights[i4] = 1
        df_final = sum([df.iloc[:, 3862:].values*wt for df, wt in zip(dfs, temp_temp_weights)])
        print("\n{}: Ensemble GAP: ".format(temp_temp_weights), evaluate(df_final, labels, fake_loss))


"""
[1, 1, 1, 1, 0]: Ensemble GAP:  0.8762875581164667
[1, 1, 1, 0, 1]: Ensemble GAP:  0.8754149386122887
[1, 1, 0, 1, 1]: Ensemble GAP:  0.8753543360934458
[1, 0, 1, 1, 1]: Ensemble GAP:  0.8752529965377961
[0, 1, 1, 1, 1]: Ensemble GAP:  0.8755567465113727
"""

"""
[1, 1, 1, 0, 0]: Ensemble GAP:  0.8737933950400538
[1, 1, 0, 1, 0]: Ensemble GAP:  0.8734770109522525
[1, 1, 0, 0, 1]: Ensemble GAP:  0.8732100971437586
[1, 0, 1, 1, 0]: Ensemble GAP:  0.8738317575936558
[1, 0, 1, 0, 1]: Ensemble GAP:  0.8733685391952352
[1, 0, 0, 1, 1]: Ensemble GAP:  0.8721525237285774
[0, 1, 1, 1, 0]: Ensemble GAP:  0.8741717994670876 Best
[0, 1, 1, 0, 1]: Ensemble GAP:  0.8731096364955323
[0, 1, 0, 1, 1]: Ensemble GAP:  0.8729611690918776
[0, 0, 1, 1, 1]: Ensemble GAP:  0.8727249234636709
"""
























