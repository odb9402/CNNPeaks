import glob
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def run(dir_name, logger, grid=2000):
    """


    :param dir_name:
    :param logger:
    :param grid:
    :return:
    """
    PATH = os.path.abspath(dir_name)
    bam_files = glob.glob(PATH + '/*.bam')
    label_files = glob.glob(PATH + '/*.txt')

    dir_list = []
    for bam_file in bam_files:
        dir_list.append(bam_file[:-4])

    for dir in dir_list:
        logger.info("DIRECTORY (TARGET) : <" + dir +">")

    input_list = {}
    for dir in dir_list:
        input_list[dir] = extractChrClass(dir)

    ##################### Hyperparameters #####################
    batch_size = 1
    evaluation_size = 1
    generations = 500
    eval_every = 5
    learning_rate = 0.005
    target_size = 20

    conv1_features = 25
    conv2_features = 50
    max_pool_size1 = 2
    max_pool_size2 = 2
    fully_connected_size1 = 30
    ###########################################################

    input_data_train = tf.placeholder(tf.float32, shape=(batch_size, grid, 1), name="readCount")
    input_data_eval = tf.placeholder(tf.float32, shape=(batch_size, grid, 1), name="readCount")

    output_data = tf.placeholder(tf.float32, shape=(batch_size, grid, 1))

    ## For convolution layers
    conv1_weight = tf.Variable(tf.truncated_normal([4,1,conv1_features],stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

    conv2_weight = tf.Variable(tf.truncated_normal([4,conv1_features,conv2_features], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

    ## For fully connected layers
    resulting_width = grid // (max_pool_size2)

    full1_input_size = resulting_width * conv2_features
    full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
    full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

    full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
    full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

    sess = tf.Session()

    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = (dir + "/" + chr + "_" + cls + "_grid" + str(grid) + ".ct")
                label_file_name = (dir + "/label_" + chr + "_" + cls + "_grid" + str(grid) + ".ct")



def peakPredictConvModel(input_data, conv_w1, conv_b1, conv_w2, conv_b2, full_w1, full_b1, full_w2, full_b2):

    ##################### Hyperparameters #####################
    batch_size = 1
    evaluation_size = 1
    generations = 500
    eval_every = 5
    learning_rate = 0.005
    target_size = 20

    conv1_features = 25
    conv2_features = 50
    max_pool_size1 = 2
    max_pool_size2 = 2
    fully_connected_size1 = 30
    ###########################################################

    conv1 = tf.nn.conv1d(input_data, conv_w1, stride=[1,1,1], padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv_b1))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1,max_pool_size1,1], strides=[1,max_pool_size1,1], padding='SAME')

    conv2 = tf.nn.conv1d(max_pool1, conv_w2, stride=[1,1,1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv_b2))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1,max_pool_size2,1], strides=[1,max_pool_size2,1], padding='SAME')

    final_conv_shape = 0
    final_shape = 0
    flat_output = 0

    fully_connected1 = tf.nn.relu()
    final_model_output = tf.add()

    return (final_model_output)


def extractChrClass(dir):
    """

    :param dir:
    :return:
    """

    chr_list = set()
    for ct_file in glob.glob(dir + "/*.ct"):
        chr_list.add(ct_file.rsplit('/', 1)[1].split('_')[0])

    data_direction = {}
    for chr in chr_list:
        cls_list = []
        for ct_file in glob.glob(dir + "/" + chr + "_*.ct"):
            cls_list.append(ct_file.rsplit('/', 1)[1].split('_')[1])  # Class numbers are string.
        data_direction[chr] = cls_list

    return data_direction