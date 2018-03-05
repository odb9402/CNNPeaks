import glob
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


def run(dir_name, logger, num_grid=10000):
    """
    This is the main function to build convolution neural network model
    for peak prediction.

    :param dir_name:
    :param logger:
    :param num_grid:
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
    global conv1a_features
    global conv1b_features
    global convMax1_features
    global convAvg1_features

    global conv2a_features
    global conv2b_features
    global convMax2_features
    global convAvg2_features

    global conv3a_features
    global conv3b_features
    global convMax3_features
    global convAvg3_features

    global conv4a_features
    global conv4b_features
    global convMax4_features
    global convAvg4_features

    global conv5a_features
    global conv5b_features
    global convMax5_features
    global convAvg5_features

    global max_pool_size1
    global max_pool_size2
    global max_pool_size3
    global max_pool_size4
    global max_pool_size5

    global fully_connected_size1

    batch_size = 1
    evaluation_size = 1
    generations = 5000
    eval_every = 10
    learning_rate = 0.005
    target_size = num_grid

    conv1_features = 16

    conv1a_features = 8
    conv1b_features = 8
    convMax1_features = 8
    convAvg1_features = 8

    conv2a_features = 32
    conv2b_features = 32
    convMax2_features = 32
    convAvg2_features = 32

    conv3a_features = 64
    conv3b_features = 64
    convMax3_features = 64
    convAvg3_features = 64

    conv4a_features = 128
    conv4b_features = 128
    convMax4_features = 128
    convAvg4_features = 128

    conv5a_features = 256
    conv5b_features = 256
    convMax5_features = 256
    convAvg5_features = 256

    max_pool_size_stem = 2
    max_pool_size1 = 2
    max_pool_size2 = 2
    max_pool_size3 = 2
    max_pool_size4 = 2
    max_pool_size5 = 5
    fully_connected_size1 = 1000
    ###########################################################

    global conv1_weight
    global conv1_bias

    global conv1a_weight
    global conv1a_bias
    global conv1b_weight
    global conv1b_bias
    global convMax1_weight
    global convMax1_bias
    global convAvg1_weight
    global convAvg1_bias

    global conv2a_weight
    global conv2a_bias
    global conv2b_weight
    global conv2b_bias
    global convMax2_weight
    global convMax2_bias
    global convAvg2_weight
    global convAvg2_bias

    global conv3a_weight
    global conv3a_bias
    global conv3b_weight
    global conv3b_bias
    global convMax3_weight
    global convMax3_bias
    global convAvg3_weight
    global convAvg3_bias

    global conv4a_weight
    global conv4a_bias
    global conv4b_weight
    global conv4b_bias
    global convMax4_weight
    global convMax4_bias
    global convAvg4_weight
    global convAvg4_bias

    global conv5a_weight
    global conv5a_bias
    global conv5b_weight
    global conv5b_bias
    global convMax5_weight
    global convMax5_bias
    global convAvg5_weight
    global convAvg5_bias

    global convMax1_weight
    global convMax1_bias
    global convAvg1_weight
    global convAvg1_bias
    global convMax1_weight
    global convMax1_bias
    global convAvg1_weight
    global convAvg1_bias

    global full1_weight
    global full1_bias
    global full2_weight
    global full2_bias
    global model_output
    global test_model_output
    global input_data_train
    global input_data_eval
    global label_data_train
    global label_data_eval
    global p_dropout
    global loss_weight
    global is_test
    global iteration

    train_data_list = []
    train_label_list = []
    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = (dir + "/" + chr + "_" + cls + "_grid" + str(num_grid) + ".ct")
                label_file_name = (dir + "/label_" + chr + "_" + cls + "_grid" + str(num_grid) + ".lb")
                train_data_list.append(pd.read_csv(input_file_name))
                train_label_list.append(pd.read_csv(label_file_name))

    test_data_list, test_label_list = splitTrainingData(train_data_list, train_label_list)

    input_data_train = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="trainingData")
    input_data_eval = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="testData")

    label_data_train = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size))
    label_data_eval = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size))

    p_dropout = tf.placeholder(tf.float32)
    loss_weight = tf.placeholder(tf.float32)
    is_test = tf.placeholder(tf.bool)
    iteration = tf.placeholder(tf.int32)

    ## For convolution layers
    conv1_weight = tf.get_variable("Conv_W1",shape=[4, 1, conv1_features], initializer=tf.contrib.layers.xavier_initializer())
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

    conv1a_weight = tf.get_variable("Conv_W6", shape=[4, conv1_features, conv1a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv1a_bias = tf.Variable(tf.zeros([conv1a_features], dtype=tf.float32))
    conv1b_weight = tf.get_variable("Conv_W7", shape=[2, conv1_features, conv1b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv1b_bias = tf.Variable(tf.zeros([conv1b_features], dtype=tf.float32))
    convMax1_weight = tf.get_variable("Conv_max_W1", shape=[1, conv1_features, convMax1_features], initializer=tf.contrib.layers.xavier_initializer())
    convMax1_bias = tf.Variable(tf.zeros([convMax1_features],dtype=tf.float32))
    convAvg1_weight = tf.get_variable("Conv_avg_W1", shape=[1, conv1_features, convAvg1_features], initializer=tf.contrib.layers.xavier_initializer())
    convAvg1_bias = tf.Variable(tf.zeros([convAvg1_features],dtype=tf.float32))

    conv2a_weight = tf.get_variable("Conv_W8", shape=[4, 32, conv2a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv2a_bias = tf.Variable(tf.zeros([conv2a_features], dtype=tf.float32))
    conv2b_weight = tf.get_variable("Conv_W9", shape=[2, 32, conv2b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv2b_bias = tf.Variable(tf.zeros([conv2b_features], dtype=tf.float32))

    conv3a_weight = tf.get_variable("Conv_W10", shape=[4, 128, conv3a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv3a_bias = tf.Variable(tf.zeros([conv3a_features], dtype=tf.float32))
    conv3b_weight = tf.get_variable("Conv_W11", shape=[2, 128, conv3b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv3b_bias = tf.Variable(tf.zeros([conv3b_features], dtype=tf.float32))
    convMax3_weight = tf.get_variable("Conv_max_W3", shape=[1, 128, convMax3_features],initializer=tf.contrib.layers.xavier_initializer())
    convMax3_bias = tf.Variable(tf.zeros([convMax3_features], dtype=tf.float32))
    convAvg3_weight = tf.get_variable("Conv_avg_W3", shape=[1, 128, convAvg3_features],initializer=tf.contrib.layers.xavier_initializer())
    convAvg3_bias = tf.Variable(tf.zeros([convAvg3_features], dtype=tf.float32))

    conv4a_weight = tf.get_variable("Conv_W12", shape=[4, 384, conv4a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv4a_bias = tf.Variable(tf.zeros([conv4a_features], dtype=tf.float32))
    conv4b_weight = tf.get_variable("Conv_W13", shape=[2, 384, conv4b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv4b_bias = tf.Variable(tf.zeros([conv4b_features], dtype=tf.float32))

    conv5a_weight = tf.get_variable("Conv_W14", shape=[4, 1024, conv5a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv5a_bias = tf.Variable(tf.zeros([conv5a_features], dtype=tf.float32))
    conv5b_weight = tf.get_variable("Conv_W15", shape=[2, 1024, conv5b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv5b_bias = tf.Variable(tf.zeros([conv5b_features], dtype=tf.float32))
    convMax5_weight = tf.get_variable("Conv_max_W5", shape=[1, 1024, convMax5_features], initializer=tf.contrib.layers.xavier_initializer())
    convMax5_bias = tf.Variable(tf.zeros([convMax5_features],dtype=tf.float32))
    convAvg5_weight = tf.get_variable("Conv_avg_W5", shape=[1, 1024, convAvg5_features], initializer=tf.contrib.layers.xavier_initializer())
    convAvg5_bias = tf.Variable(tf.zeros([convAvg5_features],dtype=tf.float32))


    ## For fully connected layers
    resulting_width = num_grid // (max_pool_size_stem * max_pool_size1 * max_pool_size2 * max_pool_size3 * max_pool_size4 * max_pool_size5)
    full1_input_size = resulting_width * (2560)

    full1_weight = tf.get_variable("Full_W1", shape=[full1_input_size, fully_connected_size1], initializer=tf.contrib.layers.xavier_initializer())
    full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

    full2_weight = tf.get_variable("Full_W2", shape=[fully_connected_size1, target_size] , initializer=tf.contrib.layers.xavier_initializer())
    full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

    model_output = peakPredictConvModel(input_data_train, logger)
    test_model_output = peakPredictConvModel(input_data_eval, logger)

    prediction = tf.nn.sigmoid(model_output)
    test_prediction = tf.nn.sigmoid(test_model_output)#

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train\
            ,logits=model_output, pos_weight=loss_weight))*100

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

        rand_y = train_label_list[rand_index[0]][['peak']].as_matrix().transpose()
        rand_y = rand_y.reshape(label_data_train.shape)

        p_n_rate = (pnRate(rand_y))

        train_dict = {input_data_train: rand_x, label_data_train: rand_y,\
                      p_dropout:0.5, loss_weight:p_n_rate, is_test:True, iteration:i}

        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        temp_train_acc = getAccuracy(temp_train_preds, rand_y, num_grid=num_grid)

        if (i+1) % eval_every == 0:
            eval_index = np.random.choice(len(test_data_list), size=batch_size)

            eval_x = test_data_list[eval_index[0]]['readCount'].as_matrix()
            eval_x = eval_x.reshape(input_data_eval.shape)

            eval_y = test_label_list[eval_index[0]][['peak']].as_matrix().transpose()
            eval_y = (eval_y.reshape(label_data_eval.shape))

            test_dict = {input_data_eval: eval_x, label_data_eval: eval_y,\
                         p_dropout:1, loss_weight:p_n_rate, is_test:False, iteration:i}

            test_preds = sess.run(test_prediction, feed_dict=test_dict)
            temp_test_acc = getAccuracy(test_preds, eval_y, num_grid=num_grid)
            TP_rate, TN_rate = tpTnRate(test_preds, eval_y, num_grid=num_grid)

            train_loss.append(temp_train_loss)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)

            acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc, TP_rate, TN_rate]
            logger.info('Generation # {}. TrainLoss: {:.2f}. TrainACC (TestACC): {:.2f}. ({:.2f}.) TPR:{:.2f} TNR:{:.2f}'\
                        .format(*acc_and_loss))

    visualizeTrainingProcess(eval_every, generations, test_acc, train_acc, train_loss)


    visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list, test_label_list,
                        test_prediction, k=-1)


    #ECCB
def peakPredictConvModel(input_data, logger):
    """

    :param logger:
    :param input_data:
    :return:
    """

    #input_data = tf.nn.batch_normalization(input_data,0,1.,0,1,0.00001)
    conv1 = tf.nn.conv1d(input_data, conv1_weight, stride=1, padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.pool(relu1, [max_pool_size1], strides=[max_pool_size1], padding='SAME', pooling_type='MAX')

    concat1 = concatLayer_B(max_pool1, conv1a_weight, convMax1_weight, conv1b_weight, convAvg1_weight,\
                            conv1a_bias, convMax1_bias, conv1b_bias, convAvg1_bias, max_pool_size1)

    concat2 = concatLayer_A(concat1, conv2a_weight, conv2b_weight, conv2a_bias, conv2b_bias, max_pool_size2)

    concat3 = concatLayer_A(concat2, conv3a_weight, conv3b_weight, conv3a_bias, conv3b_bias, max_pool_size3)

    concat4 = concatLayer_A(concat3, conv4a_weight, conv4b_weight, conv4a_bias, conv4b_bias, max_pool_size4)

    concat5 = concatLayer_A(concat4, conv5a_weight, conv5b_weight, conv5a_bias, conv5b_bias, max_pool_size5)

    final_conv_shape = concat5.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2]
    flat_output = tf.reshape(concat5, [final_conv_shape[0] , final_shape])

    fully_connected1 = tf.nn.leaky_relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias)\
                                        ,alpha=0.01, name="FullyConnected1")
    fully_connected1 = tf.nn.dropout(fully_connected1, keep_prob=p_dropout)

    final_model_output = tf.add(tf.matmul(fully_connected1,full2_weight), full2_bias)
    final_model_output = tf.reshape(final_model_output,[batch_size, 1, target_size], name="FullyConnected2")

    return (final_model_output)


def concatLayer_A(source_layer, conv1_w, conv2_w, conv1_b, conv2_b, pooling_size):
    """

    :param source_layer:
    :param conv1_w:
    :param conv2_w:
    :param conv1_b:
    :param conv2_b:
    :param pooling_size:
    :return:
    """
    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=pooling_size, padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    print(relu1.shape)
    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    print(relu2.shape)
    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size], padding='SAME', pooling_type='MAX')
    print(max_pool.shape)
    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size], padding='SAME', pooling_type='AVG')
    print(avg_pool.shape)
    concat = tf.concat([relu1, relu2, max_pool, avg_pool], axis=2)
    print(concat.shape)
    return concat


def concatLayer_B(source_layer, conv1_w, conv_max_w, conv2_w, conv_avg_w,\
                  conv1_b, conv_max_b, conv2_b, conv_avg_b, pooling_size):
    """

    :param source_layer:
    :param conv1_w:
    :param conv_max_w:
    :param conv2_w:
    :param conv_avg_w:
    :param conv1_b:
    :param conv_max_b:
    :param conv2_b:
    :param conv_avg_b:
    :param pooling_size:
    :return:
    """
    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=pooling_size, padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    print(relu1.shape)
    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    print(relu2.shape)
    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size], padding='SAME', pooling_type='MAX')
    conv_max = tf.nn.conv1d(max_pool, conv_max_w, stride=1, padding='SAME')
    relu_max = tf.nn.relu(tf.nn.bias_add(conv_max, conv_max_b))
    print(relu_max.shape)
    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size], padding='SAME', pooling_type='AVG')
    conv_avg = tf.nn.conv1d(avg_pool, conv_avg_w, stride=1, padding='SAME')
    relu_avg = tf.nn.relu(tf.nn.bias_add(conv_avg, conv_avg_b))
    print(relu_avg.shape)
    concat = tf.concat([relu1, relu2, relu_max, relu_avg], axis=2)
    print(concat.shape)
    return concat


def getAccuracy(logits, targets, num_grid=2000):
    """
    Return accuracy of the result.
    Acc = ( TP + TF ) / ( TP + TF + FN + FF )
    ( TP + TF + FN + FF ) = num_grid

    :param logits:
    :param targets:
    :return:
    """
    logits = logits.reshape(1,num_grid)
    targets = targets.reshape(1,num_grid)

    correct_num = 0.

    for index in range(len(logits[0])):
        if (logits[0][index]) >= 0.5 and targets[0][index] >= 0.5:
            correct_num += 1
        elif (logits[0][index]) < 0.5 and targets[0][index] < 0.5:
            correct_num += 1
        else:
            pass

    return correct_num / len(logits[0])


def tpTnRate(logits, targets, num_grid=2000):
    """
    Return true positive rate and true negative rate.
    By adjusting value from tpTnRate function, the loss function can get
    equilibrium between sensitivity and specificity derived from
    unbalanced ratio from regions which are peak and not peak.

    :param logits:
    :param targets:
    :param num_grid:
    :return:
    """
    logits = logits.reshape(1,num_grid)
    targets = targets.reshape(1,num_grid)

    P_num = 0.
    TP_num = 0.

    N_num = 0.
    TN_num = 0.

    for index in range(len(logits[0])):
        if targets[0][index] > 0:
            P_num += 1
            if logits[0][index] >= 0.5:
                TP_num += 1
        else:
            N_num += 1
            if logits[0][index] < 0.5:
                TN_num += 1

    if P_num == 0.:
        return (-1., TN_num/N_num)
    else:
        return (TP_num/ P_num , TN_num/ N_num)


def pnRate(targets, num_grid=2000):
    """
    Return the The ratio of Negative#/ Positive#.
    It will be used for weights of loss function to adjust
    between sensitivity and specificity.

    :param targets:
    :param num_grid:
    :return:
    """
    count = 0.
    for index in range(len(targets[0][0])):
        if targets[0][0][index] > 0:
            count += 1

    if count == 0.:
        count = 1

    return len(targets[0][0]) / count


def classValueFilter(output_value, num_grid=2000):
    """
    For output of final softmax layer,

    :param output_value:
    :param num_grid:
    :return:
    """

    class_value_list = []

    for index in range(output_value.shape[2]):
        if output_value[0][0][index] >= 0.5:
            class_value_list.append(1)
        elif output_value[0][0][index] < 0.5:
            class_value_list.append(0)

    return class_value_list


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


def visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list, test_label_list,
                        test_prediction, k = 1):
    """

    :param batch_size:
    :param input_data_eval:
    :param num_grid:
    :param label_data_eval:
    :param sess:
    :param test_data_list:
    :param test_label_list:
    :param test_prediction:
    :param k:
    :return:
    """

    if k > 0:
        for i in range(k):
            show_index = np.random.choice(len(test_data_list), size=batch_size)
            show_x = test_data_list[show_index[0]]['readCount'].as_matrix()
            show_x = show_x.reshape(input_data_eval.shape)
            show_y = test_label_list[show_index[0]][['peak']].as_matrix().transpose()
            show_y = show_y.reshape(label_data_eval.shape)
            show_dict = {input_data_eval: show_x, label_data_eval: show_y, \
                         p_dropout: 0.5, is_test: False, iteration:i}
            show_preds = sess.run(test_prediction, feed_dict=show_dict)
            show_preds = classValueFilter(show_preds, num_grid)
            show_y = classValueFilter(show_y, num_grid)
            for index in range(len(show_preds)):
                show_preds[index] += 3
            plt.plot(show_x.reshape(num_grid).tolist())
            plt.plot(show_y, 'k.', label='Real prediction')
            plt.plot(show_preds, 'r.', label='Model prediction')
            plt.title('Peak prediction result by regions')
            plt.xlabel('Regions')
            plt.ylabel('Peak')
            plt.legend(loc='lower right')
            plt.show()

    else:
        for i in range(len(test_data_list)):
            show_x = test_data_list[i]['readCount'].as_matrix()
            show_x = show_x.reshape(input_data_eval.shape)
            show_y = test_label_list[i][['peak']].as_matrix().transpose()
            show_y = show_y.reshape(label_data_eval.shape)
            show_dict = {input_data_eval: show_x, label_data_eval: show_y, \
                         p_dropout: 0.5, is_test: False, iteration:i}
            show_preds = sess.run(test_prediction, feed_dict=show_dict)
            show_preds = classValueFilter(show_preds, num_grid)
            show_y = classValueFilter(show_y, num_grid)
            for index in range(len(show_preds)):
                show_preds[index] += 3
            plt.plot(show_x.reshape(num_grid).tolist())
            plt.plot(show_y, 'k.', label='Real prediction')
            plt.plot(show_preds, 'r.', label='Model prediction')
            plt.title('Peak prediction result by regions')
            plt.xlabel('Regions')
            plt.ylabel('Peak')
            plt.legend(loc='lower right')
            plt.show()
