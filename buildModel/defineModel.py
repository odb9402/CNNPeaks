import tensorflow as tf
from buildModel.hyperparameters import *
"""
All tensor objects will be defined as global variables.
Any source can access to these values for restoring or
saving or training tensors.
"""

def tensorSummaries(tensor):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(tensor)
        gf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(tensor))
        tf.summary.scalar('min', tf.reduce_min(tensor))
        tf.summary.histogram('histogram', tensor)


def wrapTensorToTensorboard():
    """
    tensorSummaries(conv1_weight)
    tensorSummaries(conv2_weight)
    tensorSummaries(conv1_ref_weight)
    tensorSummaries(conv1a_weight)
    tensorSummaries(conv1b_weight)
    tensorSummaries(conv1c_weight)
    tensorSummaries(conv2a_weight)
    tensorSummaries(conv2b_weight)
    tensorSummaries(conv2c_weight)
    tensorSummaries(conv3a_weight)
    tensorSummaries(conv3b_weight)
    tensorSummaries(conv3c_weight)
    tensorSummaries(conv4a_weight)
    tensorSummaries(conv4b_weight)
    tensorSummaries(conv4c_weight)
    tensorSummaries(conv5a_weight)
    tensorSummaries(conv5b_weight)
    tensorSummaries(conv5c_weight)
    tensorSummaries(conv6a_weight)
    tensorSummaries(conv6b_weight)
    tensorSummaries(conv6c_weight)
    tensorSummaries(full1_weight)
    tensorSummaries(full2_weight)
    tensorSummaries(output_weight)
    tensorSummaries(output_bias)
    """
    tensorSummaries(loss_weight)
    tensorSummaries(sensitivity_train)
    tensorSummaries(sensitivity_test)
    tensorSummaries(specificity_train)
    tensorSummaries(specificity_test)


############# Loss & Dropout probability  ##############################
is_train_step = tf.placeholder(tf.bool)
loss_weight = tf.placeholder(tf.float32)
class_threshold = tf.placeholder(tf.float32)

sensitivity_train = tf.placeholder(tf.float32)
specificity_train = tf.placeholder(tf.float32)

sensitivity_test = tf.Variable(0)
specificity_test = tf.Variable(0)
############################### INPUT ##################################
input_data_train = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="trainingData")
input_data_eval = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="testData")

label_data_train = tf.placeholder(tf.float32, shape=(batch_size, 1, target_size))
label_data_eval = tf.placeholder(tf.float32, shape=(batch_size, 1, target_size))

input_ref_data_train = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="TrainRefData")
input_ref_data_eval = tf.placeholder(tf.float32, shape=(batch_size, target_size, 1), name="TestRefData")


###################### STEM FOR REFGENEDEPTH ###########################
conv1_ref_weight = tf.get_variable("Conv_REF_1", shape=[4, 1, conv1_ref_features], initializer=tf.contrib.layers.xavier_initializer())

############################ STEM Layer  ###############################
conv1_weight = tf.get_variable("Conv_STEM1", shape=[4, 1, conv1_features], initializer=tf.contrib.layers.xavier_initializer())
conv2_weight = tf.get_variable("Conv_STEM2", shape=[2, conv1_features, conv2_features], initializer=tf.contrib.layers.xavier_initializer())

############################ Inception 1 ###############################
layer1_width = conv2_features + conv1_ref_features
conv1a_weight = tf.get_variable("Conv_1A", shape=[5, layer1_width, conv1a_features], initializer=tf.contrib.layers.xavier_initializer())
conv1b_weight = tf.get_variable("Conv_1B", shape=[3, layer1_width, conv1b_features], initializer=tf.contrib.layers.xavier_initializer())
conv1c_weight = tf.get_variable("Conv_1C", shape=[2, layer1_width, conv1c_features], initializer=tf.contrib.layers.xavier_initializer())
convMax1_weight = tf.get_variable("Conv_max_W1", shape=[2, layer1_width, convMax1_features], initializer=tf.contrib.layers.xavier_initializer())
convAvg1_weight = tf.get_variable("Conv_avg_W1", shape=[2, layer1_width, convAvg1_features], initializer=tf.contrib.layers.xavier_initializer())

############################ Inception 2 ###############################
layer2_width = conv1a_features + conv1b_features + conv1c_features + (layer1_width*2)
conv2a_weight = tf.get_variable("Conv_2A", shape=[5, layer2_width, conv2a_features], initializer=tf.contrib.layers.xavier_initializer())
conv2b_weight = tf.get_variable("Conv_2B", shape=[3, layer2_width, conv2b_features], initializer=tf.contrib.layers.xavier_initializer())
conv2c_weight = tf.get_variable("Conv_2C", shape=[2, layer2_width, conv2c_features], initializer=tf.contrib.layers.xavier_initializer())
convMax2_weight = tf.get_variable("Conv_max_W2", shape=[2, layer2_width, convMax2_features], initializer=tf.contrib.layers.xavier_initializer())
convAvg2_weight = tf.get_variable("Conv_avg_W2", shape=[2, layer2_width, convAvg2_features], initializer=tf.contrib.layers.xavier_initializer())

############################ Inception 3 ###############################
layer3_width = conv2a_features + conv2b_features + (layer2_width *2 ) + conv2c_features
conv3a_weight = tf.get_variable("Conv_3A", shape=[4, layer3_width, conv3a_features],initializer=tf.contrib.layers.xavier_initializer())
conv3b_weight = tf.get_variable("Conv_3B", shape=[2, layer3_width, conv3b_features],initializer=tf.contrib.layers.xavier_initializer())
conv3c_weight = tf.get_variable("Conv_3C", shape=[3, layer3_width, conv3c_features],initializer=tf.contrib.layers.xavier_initializer())

############################ Inception 4 ###############################
layer4_width = conv3a_features + conv3b_features + conv3c_features + layer3_width
conv4a_weight = tf.get_variable("Conv_4A", shape=[4, layer4_width, conv4a_features], initializer=tf.contrib.layers.xavier_initializer())
conv4b_weight = tf.get_variable("Conv_4B", shape=[2, layer4_width, conv4b_features], initializer=tf.contrib.layers.xavier_initializer())
conv4c_weight = tf.get_variable("Conv_4C", shape=[3, layer4_width, conv4c_features],initializer=tf.contrib.layers.xavier_initializer())

############################ Inception 5 ###############################
layer5_width = conv4a_features + conv4b_features + conv4c_features +layer4_width
conv5a_weight = tf.get_variable("Conv_5A", shape=[5, layer5_width, conv5a_features],initializer=tf.contrib.layers.xavier_initializer())
conv5b_weight = tf.get_variable("Conv_5B", shape=[5, layer5_width, conv5b_features],initializer=tf.contrib.layers.xavier_initializer())
conv5c_weight = tf.get_variable("Conv_5C", shape=[5, layer5_width, conv5c_features],initializer=tf.contrib.layers.xavier_initializer())

############################ Inception 6  ###############################
layer6_width = conv5a_features + conv5b_features + conv5c_features +layer5_width
conv6a_weight = tf.get_variable("Conv_6A", shape=[6, layer6_width, conv6a_features],initializer=tf.contrib.layers.xavier_initializer())
conv6b_weight = tf.get_variable("Conv_6B", shape=[5, layer6_width, conv6b_features],initializer=tf.contrib.layers.xavier_initializer())
conv6c_weight = tf.get_variable("Conv_6C", shape=[4, layer6_width, conv6c_features],initializer=tf.contrib.layers.xavier_initializer())
conv6d_weight = tf.get_variable("Conv_6D", shape=[3, layer6_width, conv6d_features],initializer=tf.contrib.layers.xavier_initializer())

############################ Fully Connected ###############################
layer_full_width = conv6a_features + conv6b_features + conv6c_features + conv6d_features
resulting_width = 25

full1_input_size = resulting_width * ( layer_full_width )
full1_weight = tf.get_variable("Full_W1", shape=[full1_input_size, fully_connected_size1],initializer=tf.contrib.layers.xavier_initializer())
full2_weight = tf.get_variable("Full_W2", shape=[fully_connected_size1, fully_connected_size2], initializer=tf.contrib.layers.xavier_initializer())

############################ sub_optimizer #############################
"""
It cannot be tried because of lack of GPU memory

full_sub_input_size = layer1_width * 6000
full_sub_weight = tf.get_variable("Full_W_sub", shape=[full_sub_input_size, fully_connected_size_sub],initializer=tf.contrib.layers.xavier_initializer())
output_sub_weight = tf.get_variable("Full_Output_sub", shape=[fully_connected_size_sub, threshold_division], initializer=tf.contrib.layers.xavier_initializer())
output_sub_bias = tf.Variable(tf.truncated_normal([threshold_division], stddev=0.1, dtype=tf.float32))
"""

############################ Output ###############################
output_weight = tf.get_variable("Full_Output", shape=[fully_connected_size2, threshold_division], initializer=tf.contrib.layers.xavier_initializer())
output_bias = tf.Variable(tf.truncated_normal([threshold_division], stddev=0.1, dtype=tf.float32))
