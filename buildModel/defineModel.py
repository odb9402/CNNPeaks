import tensorflow as tf
from buildModel.hyperparameters import *
"""
All tensor objects will be defined as global variables.
Any source can access to these values for restoring or
saving or training tensors.
"""


############# Loss & Dropout probability  ##############################
is_test = tf.placeholder(tf.bool)
p_dropout = tf.placeholder(tf.float32)
loss_weight = tf.placeholder(tf.float32)



############################### INPUT ##################################
input_data_train = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="trainingData")
input_data_eval = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="testData")

label_data_train = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size))
label_data_eval = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size))

input_ref_data_train = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="TrainRefData")
input_ref_data_eval = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="TestRefData")


###################### STEM FOR REFGENEDEPTH ###########################
conv1_ref_weight = tf.get_variable("Conv_REF_1", shape=[4, 1, conv1_ref_features], initializer=tf.contrib.layers.xavier_initializer())
conv1_ref_bias = tf.Variable(tf.zeros([conv1_ref_features], dtype=tf.float32))

############################ STEM Layer  ###############################
conv1_weight = tf.get_variable("Conv_STEM1", shape=[4, 1, conv1_features], initializer=tf.contrib.layers.xavier_initializer())
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

conv2_weight = tf.get_variable("Conv_STEM2", shape=[3, conv1_features, conv2_features], initializer=tf.contrib.layers.xavier_initializer())
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

############################ Inception 1 ###############################
layer1_width = conv2_features + conv1_ref_features
conv1a_weight = tf.get_variable("Conv_1A", shape=[5, layer1_width, conv1a_features], initializer=tf.contrib.layers.xavier_initializer())
conv1a_bias = tf.Variable(tf.zeros([conv1a_features], dtype=tf.float32))
conv1b_weight = tf.get_variable("Conv_1B", shape=[3, layer1_width, conv1b_features], initializer=tf.contrib.layers.xavier_initializer())
conv1b_bias = tf.Variable(tf.zeros([conv1b_features], dtype=tf.float32))
conv1c_weight = tf.get_variable("Conv_1C", shape=[2, layer1_width, conv1c_features], initializer=tf.contrib.layers.xavier_initializer())
conv1c_bias = tf.Variable(tf.zeros([conv1c_features], dtype=tf.float32))
convMax1_weight = tf.get_variable("Conv_max_W1", shape=[1, layer1_width, convMax1_features], initializer=tf.contrib.layers.xavier_initializer())
convMax1_bias = tf.Variable(tf.zeros([convMax1_features], dtype=tf.float32))
convAvg1_weight = tf.get_variable("Conv_avg_W1", shape=[1, layer1_width, convAvg1_features], initializer=tf.contrib.layers.xavier_initializer())
convAvg1_bias = tf.Variable(tf.zeros([convAvg1_features], dtype=tf.float32))



############################ Inception 2 ###############################
#layer2_width = conv1a_features + conv1b_features + conv1c_features + convMax1_features + convAvg1_features# (layer1_width*2)
layer2_width = conv1a_features + conv1b_features + conv1c_features + (layer1_width*2)
conv2a_weight = tf.get_variable("Conv_2A", shape=[5, layer2_width, conv2a_features], initializer=tf.contrib.layers.xavier_initializer())
conv2a_bias = tf.Variable(tf.zeros([conv2a_features], dtype=tf.float32))
conv2b_weight = tf.get_variable("Conv_2B", shape=[3, layer2_width, conv2b_features], initializer=tf.contrib.layers.xavier_initializer())
conv2b_bias = tf.Variable(tf.zeros([conv2b_features], dtype=tf.float32))
conv2c_weight = tf.get_variable("Conv_2C", shape=[2, layer2_width, conv2c_features], initializer=tf.contrib.layers.xavier_initializer())
conv2c_bias = tf.Variable(tf.zeros([conv2c_features], dtype=tf.float32))
convMax2_weight = tf.get_variable("Conv_max_W2", shape=[1, layer2_width, convMax2_features], initializer=tf.contrib.layers.xavier_initializer())
convMax2_bias = tf.Variable(tf.zeros([convMax2_features], dtype=tf.float32))
convAvg2_weight = tf.get_variable("Conv_avg_W2", shape=[1, layer2_width, convAvg2_features], initializer=tf.contrib.layers.xavier_initializer())
convAvg2_bias = tf.Variable(tf.zeros([convAvg2_features], dtype=tf.float32))



############################ Inception 3 ###############################
layer3_width = conv2a_features + conv2b_features + (layer2_width*2) + conv2c_features
#layer3_width = conv2a_features + conv2b_features + convMax2_features + convAvg2_features
conv3a_weight = tf.get_variable("Conv_3A", shape=[4, layer3_width, conv3a_features],initializer=tf.contrib.layers.xavier_initializer())
conv3a_bias = tf.Variable(tf.zeros([conv3a_features], dtype=tf.float32))
conv3b_weight = tf.get_variable("Conv_3B", shape=[2, layer3_width, conv3b_features],initializer=tf.contrib.layers.xavier_initializer())
conv3b_bias = tf.Variable(tf.zeros([conv3b_features], dtype=tf.float32))
#convMax3_weight = tf.get_variable("Conv_max_W3", shape=[1, layer3_width, convMax3_features], initializer=tf.contrib.layers.xavier_initializer())
#convMax3_bias = tf.Variable(tf.zeros([convMax3_features], dtype=tf.float32))
#convAvg3_weight = tf.get_variable("Conv_avg_W3", shape=[1, layer3_width, convAvg3_features], initializer=tf.contrib.layers.xavier_initializer())
#convAvg3_bias = tf.Variable(tf.zeros([convAvg3_features], dtype=tf.float32))



############################ Inception 4 ###############################
layer4_width = conv3a_features + conv3b_features + (layer3_width*2)
#layer4_width = conv3a_features + conv3b_features + convMax3_features + convAvg3_features
conv4a_weight = tf.get_variable("Conv_4A", shape=[4, layer4_width, conv4a_features], initializer=tf.contrib.layers.xavier_initializer())
conv4a_bias = tf.Variable(tf.zeros([conv4a_features], dtype=tf.float32))
conv4b_weight = tf.get_variable("Conv_4B", shape=[2, layer4_width, conv4b_features], initializer=tf.contrib.layers.xavier_initializer())
conv4b_bias = tf.Variable(tf.zeros([conv4b_features], dtype=tf.float32))
#convMax4_weight = tf.get_variable("Conv_max_W4", shape=[1, layer4_width, convMax4_features], initializer=tf.contrib.layers.xavier_initializer())
#convMax4_bias = tf.Variable(tf.zeros([convMax4_features], dtype=tf.float32))
#convAvg4_weight = tf.get_variable("Conv_avg_W4", shape=[1, layer4_width, convAvg4_features], initializer=tf.contrib.layers.xavier_initializer())
#convAvg4_bias = tf.Variable(tf.zeros([convAvg4_features], dtype=tf.float32))


############################ Inception 5 ###############################
layer5_width = conv4a_features + conv4b_features + (layer4_width*2)
#layer5_width = conv4a_features + conv4b_features + convMax4_features + convAvg4_features
conv5a_weight = tf.get_variable("Conv_5A", shape=[6, layer5_width, conv5a_features],initializer=tf.contrib.layers.xavier_initializer())
conv5a_bias = tf.Variable(tf.zeros([conv5a_features], dtype=tf.float32))
conv5b_weight = tf.get_variable("Conv_5B", shape=[3, layer5_width, conv5b_features],initializer=tf.contrib.layers.xavier_initializer())
conv5b_bias = tf.Variable(tf.zeros([conv5b_features], dtype=tf.float32))
#convMax5_weight = tf.get_variable("Conv_max_W5", shape=[1, layer5_width, convMax5_features],initializer=tf.contrib.layers.xavier_initializer())
#convMax5_bias = tf.Variable(tf.zeros([convMax5_features], dtype=tf.float32))
#convAvg5_weight = tf.get_variable("Conv_avg_W5", shape=[1, layer5_width, convAvg5_features],initializer=tf.contrib.layers.xavier_initializer())
#convAvg5_bias = tf.Variable(tf.zeros([convAvg5_features], dtype=tf.float32))


############################ Inception 6  ###############################
layer6_width = conv5a_features + conv5b_features + (layer5_width*2)
#layer6_width = conv5a_features + conv5b_features + convMax5_features + convAvg5_features
conv6a_weight = tf.get_variable("Conv_6A", shape=[4, layer6_width, conv6a_features],initializer=tf.contrib.layers.xavier_initializer())
conv6a_bias = tf.Variable(tf.zeros([conv6a_features], dtype=tf.float32))
conv6b_weight = tf.get_variable("Conv_6B", shape=[2, layer6_width, conv6b_features],initializer=tf.contrib.layers.xavier_initializer())
conv6b_bias = tf.Variable(tf.zeros([conv6b_features], dtype=tf.float32))
#convMax6_weight = tf.get_variable("Conv_max_W6", shape=[1, layer6_width, convMax6_features],initializer=tf.contrib.layers.xavier_initializer())
#convMax6_bias = tf.Variable(tf.zeros([convMax6_features], dtype=tf.float32))
#convAvg6_weight = tf.get_variable("Conv_avg_W6", shape=[1, layer6_width, convAvg6_features],initializer=tf.contrib.layers.xavier_initializer())
#convAvg6_bias = tf.Variable(tf.zeros([convAvg6_features], dtype=tf.float32))


############################ Inception 7 ###############################
layer7_width = conv6a_features + conv6b_features + (layer6_width*2)
#layer7_width = conv6a_features + conv6b_features + convMax6_features + convAvg6_features
conv7a_weight = tf.get_variable("Conv_7A", shape=[8, layer7_width, conv7a_features],initializer=tf.contrib.layers.xavier_initializer())
conv7a_bias = tf.Variable(tf.zeros([conv7a_features], dtype=tf.float32))
conv7b_weight = tf.get_variable("Conv_7B", shape=[5, layer7_width, conv7b_features],initializer=tf.contrib.layers.xavier_initializer())
conv7b_bias = tf.Variable(tf.zeros([conv7b_features], dtype=tf.float32))
#convMax7_weight = tf.get_variable("Conv_max_W7", shape=[1, layer7_width, convMax7_features],initializer=tf.contrib.layers.xavier_initializer())
#convMax7_bias = tf.Variable(tf.zeros([convMax7_features], dtype=tf.float32))
#convAvg7_weight = tf.get_variable("Conv_avg_W7", shape=[1, layer7_width, convAvg7_features],initializer=tf.contrib.layers.xavier_initializer())
#convAvg7_bias = tf.Variable(tf.zeros([convAvg7_features], dtype=tf.float32))


############################ Inception 8 ###############################
layer8_width = conv7a_features + conv7b_features + (layer7_width*2)
#layer8_width = conv7a_features + conv7b_features + convMax7_features + convAvg7_features
conv8a_weight = tf.get_variable("Conv_8A", shape=[4, layer8_width, conv8a_features],initializer=tf.contrib.layers.xavier_initializer())
conv8a_bias = tf.Variable(tf.zeros([conv8a_features], dtype=tf.float32))
conv8b_weight = tf.get_variable("Conv_8B", shape=[2, layer8_width, conv8b_features],initializer=tf.contrib.layers.xavier_initializer())
conv8b_bias = tf.Variable(tf.zeros([conv8b_features], dtype=tf.float32))
#convMax8_weight = tf.get_variable("Conv_max_W8", shape=[1, layer8_width, convMax8_features],initializer=tf.contrib.layers.xavier_initializer())
#convMax8_bias = tf.Variable(tf.zeros([convMax8_features], dtype=tf.float32))
#convAvg8_weight = tf.get_variable("Conv_avg_W8", shape=[1, layer8_width, convAvg8_features],initializer=tf.contrib.layers.xavier_initializer())
#convAvg8_bias = tf.Variable(tf.zeros([convAvg8_features], dtype=tf.float32))


############################ Fully Connected ###############################
layer_full_width = conv6a_features + conv6b_features + (layer6_width * 2)
#layer_full_width = conv7a_features + conv7b_features + convMax7_features + convAvg7_features
resulting_width = 50#target_size // (max_pool_size_stem * max_pool_size1 * max_pool_size2 * max_pool_size3* max_pool_size4
                                 # * max_pool_size5  )#* max_pool_size6 )

full1_input_size = resulting_width * ( layer_full_width )
full1_weight = tf.get_variable("Full_W1", shape=[full1_input_size, fully_connected_size1],initializer=tf.contrib.layers.xavier_initializer())
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

full2_weight = tf.get_variable("Full_W2", shape=[fully_connected_size1, fully_connected_size2], initializer=tf.contrib.layers.xavier_initializer())
full2_bias = tf.Variable(tf.truncated_normal([fully_connected_size2], stddev=0.1, dtype=tf.float32))

############################ Output ###############################
output_weight = tf.get_variable("Full_Output", shape=[fully_connected_size2, threshold_division], initializer=tf.contrib.layers.xavier_initializer())
output_bias = tf.Variable(tf.truncated_normal([threshold_division], stddev=0.1, dtype=tf.float32))
