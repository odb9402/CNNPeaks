import glob
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def run(dir_name, logger, grid=2000):
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
    logger.info(input_list)


    ##################### Hyperparameters #####################
    batch_size = None
    evaluation_size = None
    generations = 500
    eval_every = 5
    learning_rate = 0.005

    conv1_features = 25
    conv2_features = 50
    max_pool_size1 = 2
    max_pool_size2 = 2
    fully_connected_size1 = 30
    ###########################################################

    sess = tf.Session()

    input_data_train = tf.placeholder(tf.int32, shape=(None,grid,1), name="readCount")
    input_data_eval = tf.placeholder(tf.int32, shape=(None,grid,1), name="readCount")

    output_data = tf.placeholder(tf.float64, shape=(None))

    conv1_weight = tf.Variable(tf.truncated_normal([4,4,1,conv1_features],stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

    conv2_weight = tf.Variable(tf.truncated_normal([4,4,conv1_features,conv2_features], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))



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