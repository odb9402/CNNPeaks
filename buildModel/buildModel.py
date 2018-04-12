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
    for label_file in label_files:
        dir_list.append(label_file[:-4])

    for dir in dir_list:
        logger.info("DIRECTORY (TARGET) : <" + dir +">")

    input_list = {}
    for dir in dir_list:
        input_list[dir] = extractChrClass(dir)

    ##################### Hyperparameters #####################
    global batch_size, evaluation_size, generations, eval_every, learning_rate, target_size,\
        conv1_features, conv1a_features, conv1b_features, convMax1_features, convAvg1_features,\
        conv2a_features, conv2b_features, convMax2_features, convAvg2_features,\
        conv3a_features, conv3b_features, convMax3_features, convAvg3_features,\
        conv4a_features, conv4b_features, convMax4_features, convAvg4_features,\
        conv5a_features, conv5b_features, convMax5_features, convAvg5_features,\
        max_pool_size_stem, max_pool_size1, max_pool_size2, max_pool_size3,\
        max_pool_size4, max_pool_size5, max_pool_size6,\
        fully_connected_size1, fully_connected_size2

    batch_size = 1
    evaluation_size = 1
    generations = 20000
    eval_every = 20
    learning_rate = 0.005
    target_size = num_grid

    conv1_features = 8

    conv1a_features = 8
    conv1b_features = 8
    convMax1_features = 8
    convAvg1_features = 8

    conv2a_features = 16
    conv2b_features = 16
    convMax2_features = 32
    convAvg2_features = 32

    conv3a_features = 32
    conv3b_features = 32
    convMax3_features = 64
    convAvg3_features = 64

    conv4a_features = 64
    conv4b_features = 64
    convMax4_features = 128
    convAvg4_features = 128

    conv5a_features = 128
    conv5b_features = 128
    convMax5_features = 256
    convAvg5_features = 256

    conv6a_features = 128
    conv6b_features = 128

    max_pool_size_stem = 2
    max_pool_size1 = 2
    max_pool_size2 = 2
    max_pool_size3 = 2
    max_pool_size4 = 2
    max_pool_size5 = 2
    max_pool_size6 = 5

    fully_connected_size1 = 800
    fully_connected_size2 = 300

    ####################Defining tensor objects############################

    global conv1_weight, conv1_bias, conv1a_weight, conv1a_bias, conv1b_weight, conv1b_bias,\
        convMax1_weight, convMax1_bias, convAvg1_weight, convAvg1_bias

    global conv2a_weight, conv2a_bias, conv2b_weight, conv2b_bias,\
        convMax2_weight, convMax2_bias, convAvg2_weight, convAvg2_bias

    global conv3a_weight, conv3a_bias, conv3b_weight, conv3b_bias,\
        convMax3_weight, convMax3_bias, convAvg3_weight, convAvg3_bias

    global conv4a_weight, conv4a_bias, conv4b_weight, conv4b_bias,\
        convMax4_weight, convMax4_bias, convAvg4_weight, convAvg4_bias

    global conv5a_weight, conv5a_bias, conv5b_weight, conv5b_bias,\
        convMax5_weight, convMax5_bias, convAvg5_weight, convAvg5_bias

    global conv6a_weight, conv6a_bias, conv6b_weight, conv6b_bias

    global full1_weight, full1_bias, full2_weight, full2_bias, full_hidden_weight, full_hidden_bias

    global model_output, test_model_output, input_data_train, input_data_eval,\
        label_data_train, label_data_eval, p_dropout, loss_weight, is_test


    input_data_train = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="trainingData")
    input_data_eval = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="testData")

    label_data_train = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size//5))
    label_data_eval = tf.placeholder(tf.float32, shape=(evaluation_size, 1, target_size//5))

    p_dropout = tf.placeholder(tf.float32)
    loss_weight = tf.placeholder(tf.float32)
    is_test = tf.placeholder(tf.bool)

    ## For convolution layers
    conv1_weight = tf.get_variable("Conv_STEM",shape=[4, 1, conv1_features], initializer=tf.contrib.layers.xavier_initializer())
    conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

    conv1a_weight = tf.get_variable("Conv_1A", shape=[4, conv1_features, conv1a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv1a_bias = tf.Variable(tf.zeros([conv1a_features], dtype=tf.float32))
    conv1b_weight = tf.get_variable("Conv_1B", shape=[2, conv1_features, conv1b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv1b_bias = tf.Variable(tf.zeros([conv1b_features], dtype=tf.float32))
    convMax1_weight = tf.get_variable("Conv_max_W1", shape=[1, conv1_features, convMax1_features], initializer=tf.contrib.layers.xavier_initializer())
    convMax1_bias = tf.Variable(tf.zeros([convMax1_features],dtype=tf.float32))
    convAvg1_weight = tf.get_variable("Conv_avg_W1", shape=[1, conv1_features, convAvg1_features], initializer=tf.contrib.layers.xavier_initializer())
    convAvg1_bias = tf.Variable(tf.zeros([convAvg1_features],dtype=tf.float32))

    conv2a_weight = tf.get_variable("Conv_2A", shape=[4, 32, conv2a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv2a_bias = tf.Variable(tf.zeros([conv2a_features], dtype=tf.float32))
    conv2b_weight = tf.get_variable("Conv_2B", shape=[2, 32, conv2b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv2b_bias = tf.Variable(tf.zeros([conv2b_features], dtype=tf.float32))
    convMax2_weight = tf.get_variable("Conv_max_W2", shape=[1, 32, convMax2_features], initializer=tf.contrib.layers.xavier_initializer())
    convMax2_bias = tf.Variable(tf.zeros([convMax2_features],dtype=tf.float32))
    convAvg2_weight = tf.get_variable("Conv_avg_W2", shape=[1, 32, convAvg2_features], initializer=tf.contrib.layers.xavier_initializer())
    convAvg2_bias = tf.Variable(tf.zeros([convAvg2_features],dtype=tf.float32))


    conv3a_weight = tf.get_variable("Conv_3A", shape=[4, 96, conv3a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv3a_bias = tf.Variable(tf.zeros([conv3a_features], dtype=tf.float32))
    conv3b_weight = tf.get_variable("Conv_3B", shape=[2, 96, conv3b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv3b_bias = tf.Variable(tf.zeros([conv3b_features], dtype=tf.float32))
    convMax3_weight = tf.get_variable("Conv_max_W3", shape=[1, 96, convMax3_features],initializer=tf.contrib.layers.xavier_initializer())
    convMax3_bias = tf.Variable(tf.zeros([convMax3_features], dtype=tf.float32))
    convAvg3_weight = tf.get_variable("Conv_avg_W3", shape=[1, 96, convAvg3_features],initializer=tf.contrib.layers.xavier_initializer())
    convAvg3_bias = tf.Variable(tf.zeros([convAvg3_features], dtype=tf.float32))

    conv4a_weight = tf.get_variable("Conv_4A", shape=[4, 256, conv4a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv4a_bias = tf.Variable(tf.zeros([conv4a_features], dtype=tf.float32))
    conv4b_weight = tf.get_variable("Conv_4B", shape=[2, 256, conv4b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv4b_bias = tf.Variable(tf.zeros([conv4b_features], dtype=tf.float32))
    convMax4_weight = tf.get_variable("Conv_max_W4", shape=[1, 256, convMax4_features], initializer=tf.contrib.layers.xavier_initializer())
    convMax4_bias = tf.Variable(tf.zeros([convMax4_features],dtype=tf.float32))
    convAvg4_weight = tf.get_variable("Conv_avg_W4", shape=[1, 256, convAvg4_features], initializer=tf.contrib.layers.xavier_initializer())
    convAvg4_bias = tf.Variable(tf.zeros([convAvg4_features],dtype=tf.float32))

    conv5a_weight = tf.get_variable("Conv_5A", shape=[2, 640, conv5a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv5a_bias = tf.Variable(tf.zeros([conv5a_features], dtype=tf.float32))
    conv5b_weight = tf.get_variable("Conv_5B", shape=[2, 640, conv5b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv5b_bias = tf.Variable(tf.zeros([conv5b_features], dtype=tf.float32))
    convMax5_weight = tf.get_variable("Conv_max_W5", shape=[1, 640, convMax5_features], initializer=tf.contrib.layers.xavier_initializer())
    convMax5_bias = tf.Variable(tf.zeros([convMax5_features],dtype=tf.float32))
    convAvg5_weight = tf.get_variable("Conv_avg_W5", shape=[1, 640, convAvg5_features], initializer=tf.contrib.layers.xavier_initializer())
    convAvg5_bias = tf.Variable(tf.zeros([convAvg5_features],dtype=tf.float32))


    conv6a_weight = tf.get_variable("Conv_6A", shape=[2, 1536, conv6a_features], initializer=tf.contrib.layers.xavier_initializer())
    conv6a_bias = tf.Variable(tf.zeros([conv6a_features], dtype=tf.float32))
    conv6b_weight = tf.get_variable("Conv_6B", shape=[2, 1536, conv6b_features], initializer=tf.contrib.layers.xavier_initializer())
    conv6b_bias = tf.Variable(tf.zeros([conv6b_features], dtype=tf.float32))


    ## For fully connected layers
    resulting_width = num_grid // (max_pool_size_stem * max_pool_size1 * max_pool_size2 * max_pool_size3 * max_pool_size4 * max_pool_size5 * max_pool_size6)
    full1_input_size = resulting_width * (3328)

    full1_weight = tf.get_variable("Full_W1", shape=[full1_input_size, fully_connected_size1], initializer=tf.contrib.layers.xavier_initializer())
    full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))

    full_hidden_weight = tf.get_variable("Full_Hidden", shape=[fully_connected_size1, fully_connected_size2], initializer=tf.contrib.layers.xavier_initializer())
    full_hidden_bias = tf.Variable(tf.truncated_normal([fully_connected_size2], stddev=0.1, dtype=tf.float32))

    full2_weight = tf.get_variable("Full_W2", shape=[fully_connected_size1, target_size//5] , initializer=tf.contrib.layers.xavier_initializer())
    full2_bias = tf.Variable(tf.truncated_normal([target_size//5], stddev=0.1, dtype=tf.float32))

    model_output = peakPredictConvModel(input_data_train, logger)
    test_model_output = peakPredictConvModel(input_data_eval, logger)

    prediction = tf.nn.sigmoid(model_output)
    test_prediction = tf.nn.sigmoid(test_model_output)

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train\
            ,logits=model_output, pos_weight=loss_weight))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)



    ###################### Training start with cross validation ##########################################

    train_data_list = []
    train_label_list = []
    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = (dir + "/" + chr + "_" + cls + "_grid" + str(num_grid) + ".ct")
                label_file_name = (dir + "/label_" + chr + "_" + cls + "_grid" + str(num_grid) + ".lb")
                train_data_list.append(pd.read_csv(input_file_name))
                train_label_list.append(pd.read_csv(label_file_name))

    K_fold = 4
    test_data_list, test_label_list = splitTrainingData(train_data_list, train_label_list, Kfold=K_fold)

    #K_fold Cross Validation
    for i in range(K_fold):
        training_data = []
        training_label = []
        test_data = []
        test_label = []
        for j in range(K_fold):
            if i == j:
                test_label += test_label_list[j]
                test_data += test_data_list[j]
            else:
                training_data += test_data_list[j]
                training_label += test_label_list[j]

        if not os.path.isdir("model_{}".format(i)):
            os.mkdir("model_{}".format(i))

        training(training_data, training_label , test_data, test_label, \
                 train_step, loss, prediction, test_prediction, logger, num_grid, i)


def training(train_data_list, train_label_list, test_data_list, test_label_list, \
             train_step, loss, prediction, test_prediction, logger, num_grid, step_num):
    """

    :param train_data_list:
    :param train_label_list:
    :param test_data_list:
    :param test_label_list:
    :param train_step:
    :param loss:
    :param prediction:
    :param test_prediction:
    :param logger:
    :param num_grid:
    :param step_num:
    :return:
    """

    init = tf.global_variables_initializer()
    sess = tf.Session()#config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)
    print("{} {}".format(len(train_data_list),len(test_data_list)))
    train_loss = []
    train_acc = []
    test_acc = []

    loss_containor_for_mean = []
    acc_containor_for_mean = []
    # Start of the training process
    for i in range(generations):
        rand_index = np.random.choice(len(train_data_list), size=batch_size)

        rand_x = train_data_list[rand_index[0]]['readCount'].as_matrix()
        rand_x = rand_x.reshape(input_data_train.shape)
        mean_x = np.mean(rand_x)
        rand_x = rand_x - mean_x

        rand_y = train_label_list[rand_index[0]][['peak']].as_matrix().transpose()
        rand_y = rand_y.reshape(label_data_train.shape)

        p_n_rate = pnRate(rand_y)

        train_dict = {input_data_train: rand_x, label_data_train: rand_y, \
                      p_dropout: 0.7, loss_weight: p_n_rate, is_test: True}

        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_preds = sess.run([loss, prediction],
                feed_dict=train_dict)
        temp_train_acc = getAccuracy(temp_train_preds, rand_y, num_grid=num_grid//5)

        loss_containor_for_mean.append(temp_train_loss)
        acc_containor_for_mean.append(temp_train_acc)

        # Recording results of test data
        if (i + 1) % eval_every == 0:
            eval_index = np.random.choice(len(test_data_list), size=batch_size)

            eval_x = test_data_list[eval_index[0]]['readCount'].as_matrix()
            eval_x = eval_x.reshape(input_data_eval.shape)
            mean_eval_x = np.mean(eval_x)
            eval_x = eval_x - mean_eval_x

            eval_y = test_label_list[eval_index[0]][['peak']].as_matrix().transpose()
            eval_y = (eval_y.reshape(label_data_eval.shape))

            p_n_rate_eval = pnRate(rand_y)

            test_dict = {input_data_eval: eval_x, label_data_eval: eval_y, \
                         p_dropout: 1, loss_weight: p_n_rate_eval, is_test: False}

            test_preds = sess.run(test_prediction, feed_dict=test_dict)
            temp_test_acc = getAccuracy(test_preds, eval_y, num_grid=num_grid//5)
            TP_rate, TN_rate = tpTnRate(test_preds, eval_y, num_grid=num_grid//5)

            loss_mean = sum(loss_containor_for_mean)/float(len(loss_containor_for_mean))
            acc_mean = sum(acc_containor_for_mean)/float(len(acc_containor_for_mean))
            train_loss.append(loss_mean)
            train_acc.append(acc_mean)
            test_acc.append(temp_test_acc)
            loss_containor_for_mean.clear()
            acc_containor_for_mean.clear()

            acc_and_loss = [(i + 1), loss_mean, acc_mean, temp_test_acc, TP_rate, TN_rate, p_n_rate_eval]
            if TP_rate == -1.0:
                logger.info('Generation # {}. TrainLoss: {:.2f}. TrainACC (TestACC): {:.2f}. ({:.2f}.) TPR:-.-- TNR:{:.2f} PN_rate:-.--' .format(i+1, loss_mean, acc_mean, temp_train_acc,  TN_rate))
            else:
                logger.info('Generation # {}. TrainLoss: {:.2f}. TrainACC (TestACC): {:.2f}. ({:.2f}.) TPR:{:.2f} TNR:{:.2f} PN_rate:{:.2f}'.format(*acc_and_loss))

    visualizeTrainingProcess(eval_every, generations, test_acc, train_acc, train_loss,
            K_fold=str(step_num))
    visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess,
            test_data_list, test_label_list,test_prediction, k=10, K_fold=str(step_num))
    saver = tf.train.Saver()
    save_path = saver.save(sess, os.getcwd() + "/model_" + str(step_num) +".ckpt")
    logger.info("Model saved in path : %s" % save_path)


def peakPredictConvModel(input_data, logger):
    """
    Define structure of convolution model.

    :param logger:
    :param input_data:
    :return: Tensor of the output layer
    """

    #Stem of model
    conv1 = tf.nn.conv1d(input_data, conv1_weight, stride=1, padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.pool(relu1, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')

    # Inception modules 1 to 6
    concat1 = concatLayer_B(max_pool1, conv1a_weight, convMax1_weight, conv1b_weight,
            convAvg1_weight,conv1a_bias, convMax1_bias, conv1b_bias, convAvg1_bias, max_pool_size1)

    concat2 = concatLayer_A(concat1, conv2a_weight, conv2b_weight, conv2a_bias, conv2b_bias, max_pool_size2)

    concat3 = concatLayer_A(concat2, conv3a_weight, conv3b_weight, conv3a_bias, conv3b_bias, max_pool_size3)

    concat4 = concatLayer_A(concat3, conv4a_weight, conv4b_weight, conv4a_bias, conv4b_bias, max_pool_size4)

    concat5 = concatLayer_A(concat4, conv5a_weight, conv5b_weight, conv5a_bias, conv5b_bias, max_pool_size5)

    concat6 = concatLayer_A(concat5, conv6a_weight, conv6b_weight, conv6a_bias, conv6b_bias,max_pool_size6)

    final_conv_shape = concat6.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2]
    flat_output = tf.reshape(concat6, [final_conv_shape[0] , final_shape])

    fully_connected1 = tf.nn.leaky_relu(tf.add(tf.matmul(flat_output, full1_weight),
        full1_bias),alpha=0.005 ,name="FullyConnected1")
    fully_connected1 = tf.nn.dropout(fully_connected1, keep_prob=p_dropout)

    fully_connected2 = tf.nn.selu(tf.add(tf.matmul(fully_connected1
        ,full_hidden_weight), full_hidden_bias),name = "FullyConnectedHidden")
    fully_connected2 = tf.nn.dropout(fully_connected2, keep_prob=p_dropout)

    final_model_output = (tf.add(tf.matmul(fully_connected1,full2_weight), full2_bias))
    final_model_output = tf.reshape(final_model_output,[batch_size, 1, target_size//5], name="FullyConnected2")

    return (final_model_output)


def concatLayer_A(source_layer, conv1_w, conv2_w, conv1_b, conv2_b, pooling_size):
    """
    Define concat layer which like Inception module.

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

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='MAX')

    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='AVG')

    concat = tf.concat([relu1, avg_pool, relu2, max_pool], axis=2)
    print(concat.shape)
    return concat


def concatLayer_B(source_layer, conv1_w, conv_max_w, conv2_w, conv_avg_w,\
                  conv1_b, conv_max_b, conv2_b, conv_avg_b, pooling_size):
    """
    Define concat layer which like Inception module.

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

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='MAX')
    conv_max = tf.nn.conv1d(max_pool, conv_max_w, stride=1, padding='SAME')
    relu_max = tf.nn.relu(tf.nn.bias_add(conv_max, conv_max_b))

    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='AVG')
    conv_avg = tf.nn.conv1d(avg_pool, conv_avg_w, stride=1, padding='SAME')
    relu_avg = tf.nn.relu(tf.nn.bias_add(conv_avg, conv_avg_b))

    concat = tf.concat([relu1, relu_max, relu2, relu_avg], axis=2)
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

    # For the label only has negative samples.
    if count == 0.:
        return 1

    return (len(targets[0][0]) - count) / count


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
    Extract a chromosome number and a class number from label file names.

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
            cls_list.append(ct_file.rsplit('/', 1)[1].split('_')[1])
        data_direction[chr] = cls_list

    return data_direction


def splitTrainingData(data_list, label_list, Kfold=4):
    """

    :param list_data:
    :param Kfold:
    :return:
    """
    print("##################NUMBER OF LABEL DATA : {}".format(len(data_list)))

    size = len(data_list)
    counter = size / Kfold

    test_data = []
    test_label = []
    for i in range(Kfold - 1):
        test_data_temp = []
        test_label_temp = []
        while True:
            if counter <= 0:
                test_data.append(test_data_temp)
                test_label.append(test_label_temp)
                counter = size / Kfold
                break

            pop_index = random.randint(0,len(test_data))
            test_data_temp.append(data_list.pop(pop_index))
            test_label_temp.append(label_list.pop(pop_index))
            counter -= 1

    test_data.append(data_list)
    test_label.append(label_list)

    return test_data, test_label


def visualizeTrainingProcess(eval_every, generations, test_acc, train_acc, train_loss, K_fold =""):
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
    plt.axis([0,generations,0,3])
    plt.title('L1 Loss per generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('model_{}/LossPerGen.png'.format(K_fold))
    plt.clf()

    plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(eval_indices, test_acc, 'r-', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('model_{}/AccPerGen.png'.format(K_fold))
    plt.clf()


def expandingPrediction(input_list, multiple=5):
    """

    :param input_list:
    :param multiple:
    :return:
    """
    expanded_list = []
    for prediction in input_list:
        for i in range(multiple):
            expanded_list.append(prediction)

    return expanded_list


def visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess,
        test_data_list, test_label_list, test_prediction, k = 1, K_fold=""):
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
                         p_dropout: 0.5, is_test: False}
            show_preds = sess.run(test_prediction, feed_dict=show_dict)

            show_preds = expandingPrediction(classValueFilter(show_preds, num_grid))
            show_y = expandingPrediction(classValueFilter(show_y, num_grid))

            for index in range(len(show_preds)):
                if show_preds[index] > 0:
                    show_preds[index] += 1

            y_index = []
            y = []
            pred_index = []
            pred = []

            for index in range(len(show_preds)):
                if show_y[index] > 0:
                    y_index.append(index)
                    y.append(show_y[index])
                if show_preds[index] > 0:
                    pred_index.append(index)
                    pred.append(show_preds[index])

            plt.plot(show_x.reshape(num_grid).tolist(),'k')
            plt.plot(y_index,y, 'b.', label='Real prediction')
            plt.plot(pred_index,pred, 'r.', label='Model prediction')
            plt.title('Peak prediction result by regions')
            plt.xlabel('Regions')
            plt.ylabel('Read Count')
            plt.legend(loc='lower right')
            plt.show()
            plt.savefig('model_{}/peak{}.png'.format(K_fold,i))
            plt.clf()
