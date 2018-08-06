import tensorflow as tf 
import numpy as np 
import os 


tf.reset_default_graph()

input_node_one = 'eval_input/batch_join:1'
input_node_two = 'eval_input/batch_join:3'

###### common input and num_frames ######
common_input = tf.placeholder(name = 'final_model/eval_input/input', shape=(None, 300, 1024 + 128), dtype = tf.float32)
common_num_frames = tf.placeholder(name = 'final_model/eval_input/frames', shape = [None], dtype = tf.int32)
input_map = {input_node_one:common_input, input_node_two:common_num_frames}
###### common input and num_frames ######

common_input_forward_concat = tf.concat([common_input, common_input], 1)
common_input_reversed = tf.reverse(common_input, [1])

## full reverse
common_input_full_rev = tf.map_fn(lambda x: x[0][x[1]:x[1]+300], (common_input_forward_concat, common_num_frames), tf.float32)
common_input_full_rev = tf.reverse(common_input_full_rev, [1])

### half seq below 
common_num_frames_float = tf.cast(common_num_frames, tf.float32)
half_n_frames =  tf.cast(tf.maximum(tf.round(common_num_frames_float/tf.constant(2.0)+0.000001), 1), tf.int32)
half_n_frames_start_index_1 = tf.cast(common_num_frames_float/tf.constant(2.0), tf.int32) # starting pos index for half seq 1 cases

## half reverse, at pos 0 
common_input_half_rev_0 = tf.map_fn(lambda x: tf.concat( [ tf.zeros([tf.constant(150)-x[1], 1152], dtype = tf.float32), x[0][:x[1]] ], 0), \
											  (common_input, half_n_frames), \
											  tf.float32)
common_input_half_rev_0 = tf.reverse(common_input_half_rev_0, [1])

## half non rev, at pos 0 
common_input_half_nonRev_0 = tf.map_fn(lambda x: tf.concat( [ x[0][:x[1]],tf.zeros([tf.constant(150)-x[1],1152],dtype=tf.float32) ], 0), \
												  (common_input, half_n_frames), \
												  tf.float32)
## half non rev, at pos 1 
common_input_half_nonRev_1 = tf.map_fn(lambda x: x[0][x[1]:x[1]+tf.constant(150)], \
											    (common_input, half_n_frames_start_index_1), \
											    tf.float32)
## half rev, at pos 1  
common_input_half_rev_1 = tf.map_fn(lambda x: tf.concat( [ tf.zeros([tf.constant(150)-x[2], 1152], dtype = tf.float32), x[0][x[1]:x[1]+x[2]] ], 0), \
											  (common_input, half_n_frames_start_index_1, half_n_frames), \
											  tf.float32)

common_input_half_rev_1 = tf.reverse(common_input_half_rev_1, [1])








###### testing cases 

## 1) speed 
random_input = np.random.normal(0, 1, (256, 300, 1024+128))
n_frames = np.random.choice(range(1, 301), [256])

sess = tf.InteractiveSession()
timeit sess.run([common_input_full_rev, common_input_half_rev_0, common_input_half_nonRev_0, common_input_half_nonRev_1, common_input_half_rev_1], \
	feed_dict = {common_input:random_input, common_num_frames:n_frames})

timeit sess.run([common_input_full_rev], feed_dict = {common_input:random_input, common_num_frames:n_frames})      # 256 cases: 1.1 s

timeit sess.run([common_input_half_rev_0], feed_dict = {common_input:random_input, common_num_frames:n_frames})    # 256 cases: 651 ms

timeit sess.run([common_input_half_nonRev_0], feed_dict = {common_input:random_input, common_num_frames:n_frames}) # 256 cases: 602 ms

timeit sess.run([common_input_half_nonRev_1], feed_dict = {common_input:random_input, common_num_frames:n_frames}) # 256 cases: 550 ms

timeit sess.run([common_input_half_rev_1], feed_dict = {common_input:random_input, common_num_frames:n_frames})    # 256 cases: 654 ms


## 2) cases 
random_input = np.random.normal(0, 1, (1, 2, 1152)).astype(np.float32)
random_input = np.concatenate([random_input, np.zeros([1, 298, 1152], dtype = np.float32)], 1)
n_frames = np.array([2])
a, b, c, d, e = sess.run([common_input_full_rev, common_input_half_rev_0, common_input_half_nonRev_0, common_input_half_nonRev_1, common_input_half_rev_1], \
	feed_dict = {common_input:random_input, common_num_frames:n_frames})


random_input = np.random.normal(0, 1, (1, 5, 1152)).astype(np.float32)
random_input = np.concatenate([random_input, np.zeros([1, 295, 1152], dtype = np.float32)], 1)
n_frames = np.array([5])
a, b, c, d, e = sess.run([common_input_full_rev, common_input_half_rev_0, common_input_half_nonRev_0, common_input_half_nonRev_1, common_input_half_rev_1], \
	feed_dict = {common_input:random_input, common_num_frames:n_frames})

random_input = np.random.normal(0, 1, (1, 300, 1152)).astype(np.float32)
#random_input = np.concatenate([random_input, np.zeros([1, 295, 1152], dtype = np.float32)], 1)
n_frames = np.array([300])
a, b, c, d, e = sess.run([common_input_full_rev, common_input_half_rev_0, common_input_half_nonRev_0, common_input_half_nonRev_1, common_input_half_rev_1], \
	feed_dict = {common_input:random_input, common_num_frames:n_frames})
















