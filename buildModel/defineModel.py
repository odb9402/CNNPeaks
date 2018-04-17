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

label_data_train = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size // 5))
label_data_eval = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size // 5))



############################ STEM Layer  ###############################
conv1_weight = tf.get_variable("Conv_STEM", shape=[4, 1, conv1_features], initializer=tf.contrib.layers.xavier_initializer())
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))



############################ Inception 1 ###############################
conv1a_weight = tf.get_variable("Conv_1A", shape=[4, conv1_features, conv1a_features], initializer=tf.contrib.layers.xavier_initializer())
conv1a_bias = tf.Variable(tf.zeros([conv1a_features], dtype=tf.float32))
conv1b_weight = tf.get_variable("Conv_1B", shape=[2, conv1_features, conv1b_features], initializer=tf.contrib.layers.xavier_initializer())
conv1b_bias = tf.Variable(tf.zeros([conv1b_features], dtype=tf.float32))
convMax1_weight = tf.get_variable("Conv_max_W1", shape=[1, conv1_features, convMax1_features], initializer=tf.contrib.layers.xavier_initializer())
convMax1_bias = tf.Variable(tf.zeros([convMax1_features], dtype=tf.float32))
convAvg1_weight = tf.get_variable("Conv_avg_W1", shape=[1, conv1_features, convAvg1_features], initializer=tf.contrib.layers.xavier_initializer())
convAvg1_bias = tf.Variable(tf.zeros([convAvg1_features], dtype=tf.float32))



############################ Inception 2 ###############################
layer2_width = 32
conv2a_weight = tf.get_variable("Conv_2A", shape=[4, 32, conv2a_features], initializer=tf.contrib.layers.xavier_initializer())
conv2a_bias = tf.Variable(tf.zeros([conv2a_features], dtype=tf.float32))
conv2b_weight = tf.get_variable("Conv_2B", shape=[2, 32, conv2b_features], initializer=tf.contrib.layers.xavier_initializer())
conv2b_bias = tf.Variable(tf.zeros([conv2b_features], dtype=tf.float32))
#convMax2_weight = tf.get_variable("Conv_max_W2", shape=[1, 32, convMax2_features], initializer=tf.contrib.layers.xavier_initializer())
#convMax2_bias = tf.Variable(tf.zeros([convMax2_features], dtype=tf.float32))
#convAvg2_weight = tf.get_variable("Conv_avg_W2", shape=[1, 32, convAvg2_features], initializer=tf.contrib.layers.xavier_initializer())
#convAvg2_bias = tf.Variable(tf.zeros([convAvg2_features], dtype=tf.float32))



############################ Inception 3 ###############################
layer3_width = conv2a_features + conv2b_features + (layer2_width*2)
conv3a_weight = tf.get_variable("Conv_3A", shape=[4, layer3_width, conv3a_features],initializer=tf.contrib.layers.xavier_initializer())
conv3a_bias = tf.Variable(tf.zeros([conv3a_features], dtype=tf.float32))
conv3b_weight = tf.get_variable("Conv_3B", shape=[2, layer3_width, conv3b_features],initializer=tf.contrib.layers.xavier_initializer())
conv3b_bias = tf.Variable(tf.zeros([conv3b_features], dtype=tf.float32))
#convMax3_weight = tf.get_variable("Conv_max_W3", shape=[1, 96, convMax3_features], initializer=tf.contrib.layers.xavier_initializer())
#convMax3_bias = tf.Variable(tf.zeros([convMax3_features], dtype=tf.float32))
#convAvg3_weight = tf.get_variable("Conv_avg_W3", shape=[1, 96, convAvg3_features], initializer=tf.contrib.layers.xavier_initializer())
#convAvg3_bias = tf.Variable(tf.zeros([convAvg3_features], dtype=tf.float32))



############################ Inception 4 ###############################
layer4_width = conv3a_features + conv3b_features + (layer3_width*2)
conv4a_weight = tf.get_variable("Conv_4A", shape=[4, layer4_width, conv4a_features], initializer=tf.contrib.layers.xavier_initializer())
conv4a_bias = tf.Variable(tf.zeros([conv4a_features], dtype=tf.float32))
conv4b_weight = tf.get_variable("Conv_4B", shape=[2, layer4_width, conv4b_features], initializer=tf.contrib.layers.xavier_initializer())
conv4b_bias = tf.Variable(tf.zeros([conv4b_features], dtype=tf.float32))
#convMax4_weight = tf.get_variable("Conv_max_W4", shape=[1, 256, convMax4_features], initializer=tf.contrib.layers.xavier_initializer())
#convMax4_bias = tf.Variable(tf.zeros([convMax4_features], dtype=tf.float32))
#convAvg4_weight = tf.get_variable("Conv_avg_W4", shape=[1, 256, convAvg4_features], initializer=tf.contrib.layers.xavier_initializer())
#convAvg4_bias = tf.Variable(tf.zeros([convAvg4_features], dtype=tf.float32))



############################ Inception 5 ###############################
layer5_width = conv4a_features + conv4b_features + (layer4_width*2)
conv5a_weight = tf.get_variable("Conv_5A", shape=[2, layer5_width, conv5a_features],initializer=tf.contrib.layers.xavier_initializer())
conv5a_bias = tf.Variable(tf.zeros([conv5a_features], dtype=tf.float32))
conv5b_weight = tf.get_variable("Conv_5B", shape=[2, layer5_width, conv5b_features],initializer=tf.contrib.layers.xavier_initializer())
conv5b_bias = tf.Variable(tf.zeros([conv5b_features], dtype=tf.float32))
#convMax5_weight = tf.get_variable("Conv_max_W5", shape=[1, 640, convMax5_features],initializer=tf.contrib.layers.xavier_initializer())
#convMax5_bias = tf.Variable(tf.zeros([convMax5_features], dtype=tf.float32))
#convAvg5_weight = tf.get_variable("Conv_avg_W5", shape=[1, 640, convAvg5_features],initializer=tf.contrib.layers.xavier_initializer())
#convAvg5_bias = tf.Variable(tf.zeros([convAvg5_features], dtype=tf.float32))



############################ Inception 6 ###############################
#conv6a_weight = tf.get_variable("Conv_6A", shape=[2, 1536, conv6a_features], initializer=tf.contrib.layers.xavier_initializer())
#conv6a_bias = tf.Variable(tf.zeros([conv6a_features], dtype=tf.float32))
#conv6b_weight = tf.get_variable("Conv_6B", shape=[2, 1536, conv6b_features], initializer=tf.contrib.layers.xavier_initializer())
#conv6b_bias = tf.Variable(tf.zeros([conv6b_features], dtype=tf.float32))



############################ Fully Connected ###############################
resulting_width = target_size // (max_pool_size_stem * max_pool_size1 * max_pool_size2 * max_pool_size3* max_pool_size4 * max_pool_size5 )
full1_input_size = resulting_width * ( conv5a_features + conv5b_features + layer5_width*2)
full1_weight = tf.get_variable("Full_W1", shape=[full1_input_size, fully_connected_size1],initializer=tf.contrib.layers.xavier_initializer())
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

#full_hidden_weight = tf.get_variable("Full_Hidden", shape=[fully_connected_size1, fully_connected_size2],
#                                     initializer=tf.contrib.layers.xavier_initializer())
#full_hidden_bias = tf.Variable(tf.truncated_normal([fully_connected_size2], stddev=0.1, dtype=tf.float32))


############################ Output ###############################
full2_weight = tf.get_variable("Full_W2", shape=[fully_connected_size1, target_size // 5], initializer=tf.contrib.layers.xavier_initializer())
full2_bias = tf.Variable(tf.truncated_normal([target_size // 5], stddev=0.1, dtype=tf.float32))
