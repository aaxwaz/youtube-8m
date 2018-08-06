"""This script will load and merge all models inside current directory into one graph. 
The dir structure should be: 
./model1/...
./model2/...
./model3/...
...

The files inside each folder should include the three below: 
`inference_model.meta`
`inference_model.data-00000-of-00001`
`inference_model.index`

The params of `model_dir`, `models`, `model_weightings` as well as `input_node_one` and `input_node_two` must be declared before script is run
`model_dir` - dir to the folders of models 
`models` - list of model folder names 
`model_weightings` - list of model weightings for each model 

`input_node_one` - first input node 
`input_node_two` - second input node 

output: 
`merged_model` that contains the final merged model graph
"""

import tensorflow as tf 
import numpy as np 
import os 

model_dir = '/Users/weimin/Desktop/yt8m/youtube-8m/ensemble/final_1GB_ensemble/FINAL_6_EACH'
models = ['LightVLAD_SVD_2_mat_v2_hiddenLayer', 'FV_fv1Only_SVDMidTanh_hiddenLayer', 'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_CNN_NEW1', \
'Multiscale_CNN_LSTM_TanhCNN_halfFrame_2_loops', 'prev_FIVE_Multiscale_CNN_LSTM_TanhCNN_166MB']

model_weightings = [1.3, 1.3, 1.1, 1.1, 0.8]

input_node_one = 'eval_input/batch_join:1'
input_node_two = 'eval_input/batch_join:3'

################## tensor names for each models! ####################
# tf.get_collection('predictions')     -> 'tower/Mul_2:0'
# tf.get_collection('input_batch_raw') -> 'eval_input/batch_join:1'
# tf.get_collection('num_frames')      -> 'eval_input/batch_join:3'
#####################################################################

tf.reset_default_graph()

###### common input and num_frames ######
common_input = tf.placeholder(name = 'final_model/eval_input/input', shape=(None, 300, 1024 + 128), dtype = tf.float32)
common_num_frames = tf.placeholder(name = 'final_model/eval_input/frames', shape = [None], dtype = tf.int32)
input_map = {input_node_one:common_input, input_node_two:common_num_frames}
###### common input and num_frames ######

## load model graphs 
savers = []
for model in models:
	model_meta = os.path.join(model_dir, model, 'inference_model.meta')
	savers.append( tf.train.import_meta_graph(model_meta, import_scope=model, input_map = input_map) )

## get hold of prediction nodes
predictions = []
for pred in tf.get_collection('predictions'):
	predictions.append( tf.get_default_graph().get_tensor_by_name(pred.name) )

## remove collections 
tf.get_collection_ref('input_batch_raw').clear()
tf.get_collection_ref('num_frames').clear()
tf.get_collection_ref('predictions').clear()

with tf.Session() as sess:

	for saver, model in zip(savers, models):
		this_model_dir = os.path.join(model_dir, model, 'inference_model')
		saver.restore(sess, this_model_dir)

	prediction_final = sum([wt * pred for wt, pred in zip(model_weightings, predictions)])

	## add in final collections
	tf.add_to_collection('predictions', prediction_final)
	tf.add_to_collection('input_batch_raw', common_input)
	tf.add_to_collection('num_frames', common_num_frames)

	saver = tf.train.Saver()
	merged_model_dir = os.path.join(model_dir, 'merged_model', 'inference_model')
	saver.save(sess, merged_model_dir)

	print("done. ")










