import glob
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold


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
    global batch_size
    global evaluation_size
    global generations
    global eval_every
    global learning_rate
    global target_size
    global conv1_features
    global conv2_features
    global max_pool_size1
    global max_pool_size2
    global fully_connected_size1

    batch_size = 1
    evaluation_size = 1
    generations = 300
    eval_every = 1
    learning_rate = 0.01
    target_size = grid

    conv1_features = 25
    conv2_features = 50
    max_pool_size1 = 5
    max_pool_size2 = 5
    fully_connected_size1 = 10
    ###########################################################

    global conv1_weight
    global conv1_bias
    global conv2_weight
    global conv2_bias
    global full1_weight
    global full1_bias
    global full2_weight
    global full2_bias

    train_data_list = []
    train_label_list = []
    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = (dir + "/" + chr + "_" + cls + "_grid" + str(grid) + ".ct")
                label_file_name = (dir + "/label_" + chr + "_" + cls + "_grid" + str(grid) + ".lb")
                train_data_list.append(pd.read_csv(input_file_name))
                train_label_list.append(pd.read_csv(label_file_name))

    test_data_list, test_label_list = splitTrainingData(train_data_list, train_label_list)

    input_data_train = tf.placeholder(tf.float32, shape=(batch_size, grid, 1), name="readCount")
    input_data_eval = tf.placeholder(tf.float32, shape=(batch_size, grid, 1), name="readCount")

    output_data_train = tf.placeholder(tf.float32, shape=(evaluation_size, grid))
    output_data_eval = tf.placeholder(tf.float32, shape=(evaluation_size, grid))

    ## For convolution layers
    conv1_weight = tf.Variable(tf.truncated_normal([4,1,conv1_features],stddev=0.1, dtype=tf.float32))
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

    conv2_weight = tf.Variable(tf.truncated_normal([4,conv1_features,conv2_features], stddev=0.1, dtype=tf.float32))
    conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

    ## For fully connected layers
    resulting_width = grid // ( max_pool_size1 * max_pool_size2)

    full1_input_size = resulting_width * conv2_features
    full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
    full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

    full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
    full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

    model_output = peakPredictConvModel(input_data_train, logger)
    test_model_output = peakPredictConvModel(input_data_eval, logger)

    loss = tf.reduce_mean(((model_output - (tf.reshape(output_data_train, model_output.shape))))**2)

    prediction = model_output
    test_prediction = test_model_output

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train_loss = []
    train_acc = []
    test_acc = []
    for i in range(generations):
        rand_index = np.random.choice(len(train_data_list), size=batch_size)
        rand_x = train_data_list[rand_index[0]]['readCount'].as_matrix()
        rand_x = rand_x.reshape(input_data_train.shape)
        rand_y = train_label_list[rand_index[0]]['peak'].as_matrix()
        rand_y = rand_y.reshape(output_data_train.shape)
        train_dict = {input_data_train: rand_x, output_data_train: rand_y}

        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        #logger.info(temp_train_preds)
        #logger.info(rand_y)
        temp_train_acc = getAccuracy(temp_train_preds, rand_y)

        if (i+1) % eval_every == 0:
            eval_index = np.random.choice(len(test_data_list), size=batch_size)
            eval_x = test_data_list[eval_index[0]]['readCount'].as_matrix()
            eval_x = eval_x.reshape(input_data_eval.shape)
            eval_y = test_label_list[eval_index[0]]['peak'].as_matrix()
            eval_y = eval_y.reshape(output_data_eval.shape)
            test_dict = {input_data_eval: eval_x, output_data_eval: eval_y}

            test_preds = sess.run(test_prediction, feed_dict=test_dict)
            temp_test_acc = getAccuracy(test_preds, eval_y)

            train_loss.append(temp_train_loss)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)

            acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
            acc_and_loss = [np.round(x,2) for x in acc_and_loss]
            print('Generation # {}. TrainLoss: {:.2f}. TrainACC (TestACC): {:.2f}. ({:.2f}.)'.format(*acc_and_loss))


    visualizeTrainingProcess(eval_every, generations, test_acc, train_acc, train_loss)


def visualizeTrainingProcess(eval_every, generations, test_acc, train_acc, train_loss):
    """

    :param eval_every:
    :param generations:
    :param test_acc:
    :param train_acc:
    :param train_loss:
    :return:
    """
    eval_indices = range(0, generations, eval_every)

    plt.plot(eval_indices, train_loss, 'k-')
    plt.title('L1 Loss per generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def peakPredictConvModel(input_data, logger):
    """

    :param logger:
    :param input_data:
    :return:
    """

    logger.debug(input_data)
    conv1 = tf.nn.conv1d(input_data, conv1_weight, stride=1, padding='SAME')
    logger.debug(conv1)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    logger.debug(relu1)
    max_pool1 = tf.nn.pool(relu1, [max_pool_size1], strides=[max_pool_size1], padding='SAME', pooling_type='MAX')
    logger.debug(max_pool1)

    conv2 = tf.nn.conv1d(max_pool1, conv2_weight, stride=1, padding='SAME')
    logger.debug(conv2)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))
    logger.debug(relu2)
    max_pool2 = tf.nn.pool(relu2, [max_pool_size2], strides=[max_pool_size2], padding='SAME', pooling_type='MAX')
    logger.debug(max_pool2)

    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0] , final_shape])

    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
    logger.debug(fully_connected1)
    final_model_output = tf.add(tf.matmul(fully_connected1,full2_weight), full2_bias)
    final_model_output = tf.reshape(final_model_output,[1,final_model_output.shape[1],1])
    logger.debug(final_model_output)
    logger.debug("///////////////////////////////////")

    return (final_model_output)


def getAccuracy(logits, targets):
    """

    :param logits:
    :param targets:
    :return:
    """

    logits = logits.reshape(2000)
    targets = targets.reshape(2000)

    correct_num = 0.

    for index in range(len(logits)):
        if logits[index] > 0 and targets[index] > 0:
            correct_num += 1
        elif logits[index] < 0 and targets[index] < 0:
            correct_num += 1
        else:
            pass

    return correct_num / len(logits)


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


def splitTrainingData(data_list, label_list, Kfold=4):
    """

    :param list_data:
    :param Kfold:
    :return:
    """

    counter = len(data_list) / Kfold

    test_data = []
    test_label = []
    while True:
        if counter <= 0:
            break
        pop_index = random.randint(0,len(data_list) - 1)
        test_data.append(data_list.pop(pop_index))
        test_label.append(label_list.pop(pop_index))
        counter -= 1

    return test_data, test_label