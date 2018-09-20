import glob
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from .defineModel import *
from .hyperparameters import *

def run(dir_name, logger, num_grid=0, K_fold_in=10):
    """
    This is a main function to build convolution neural network model
    for peak prediction.

    :param dir_name:
    :param logger:
    :param num_grid:
    :param K_fold_in:
    :return:
    """
    PATH = os.path.abspath(dir_name)

    dir_list = os.listdir(PATH)

    for dir in dir_list:
        dir = PATH + '/' + dir
        logger.info("DIRECTORY (TARGET) : <" + dir +">")

    input_list = {}
    for dir in dir_list:
        dir = PATH + '/' + dir
        input_list[dir] = extractChrClass(dir)

    model_output = peakPredictConvModel(input_data_train, input_ref_data_train, logger)
    test_model_output = peakPredictConvModel(input_data_eval, input_ref_data_eval, logger)

    prediction = (generateOutput(model_output, input_data_train, div=threshold_division))
    test_prediction = (generateOutput(test_model_output, input_data_eval, div=threshold_division))

    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train, logits=prediction,
                                                                   pos_weight=loss_weight))

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    #################### Training start with cross validation ##################################
    input_data_names = []
    ref_data_names = []
    label_data_names = []

    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = "{}/{}_{}_grid{}.ct".format(dir, chr, cls, num_grid)
                ref_file_name = "{}/ref_{}_{}_grid{}.ref".format(dir, chr, cls, num_grid)
                label_file_name = "{}/label_{}_{}_grid{}.lb".format(dir, chr, cls, num_grid)
                input_data_names.append(pd.read_csv(input_file_name))
                ref_data_names.append(pd.read_csv(ref_file_name))
                label_data_names.append(pd.read_csv(label_file_name))

    K_fold = K_fold_in
    input_data_list, label_data_list, ref_data_list = splitTrainingData(input_data_names, label_data_names, ref_data_names
                                                                       , Kfold=K_fold)

    if not os.path.isdir(os.getcwd() + "/models"):
        os.mkdir(os.getcwd() + "/models")


    #K_fold Cross Validation
    for i in range(K_fold):
        training_data = []
        training_ref = []
        training_label = []
        test_data = []
        test_ref = []
        test_label = []
        for j in range(K_fold):
            if i == j:
                test_data += input_data_list[j]
                test_ref += ref_data_list[j]
                test_label += label_data_list[j]
            else:
                training_data += input_data_list[j]
                training_ref += ref_data_list[j]
                training_label += label_data_list[j]

        if not os.path.isdir(os.getcwd() + "/models/model_{}".format(i)):
            os.mkdir(os.getcwd() + "/models/model_{}".format(i))

        training(training_data, training_label , training_ref, test_data, test_label, test_ref, train_step, loss, prediction,
                test_prediction, logger, num_grid, i)


def training(train_data_list, train_label_list, train_ref_list, test_data_list, test_label_list, test_ref_list, train_step,
             loss, prediction, test_prediction, logger, num_grid, step_num):
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
    sess = tf.Session()
    sess.run(init)

    train_loss = []
    train_spec = []
    train_sens = []
    test_spec = []
    test_sens = []

    loss_containor_for_mean = []
    spec_containor_for_mean = []
    sens_containor_for_mean = []

    # Start of the training process
    for i in range(generations):
        rand_index = np.random.choice(len(train_data_list), size=batch_size)

        rand_x = []
        rand_ref = []
        rand_y = []
        for j in range(batch_size):
            rand_x.append(train_data_list[rand_index[j]]['readCount'].values)
            rand_ref.append(train_ref_list[rand_index[j]]['refGeneCount'].values)
            rand_y.append(np.repeat(train_label_list[rand_index[j]][['peak']].values.transpose(),5))

        #rand_x = train_data_list[rand_index[0]]['readCount'].values
        rand_x = np.array(rand_x).reshape(input_data_train.shape)
        #rand_ref = train_ref_list[rand_index[0]]['refGeneCount'].values
        rand_ref = np.array(rand_ref).reshape(input_ref_data_train.shape)

        #rand_y = train_label_list[rand_index[0]][['peak']].values.transpose()
        #rand_y = np.repeat(rand_y, 5)
        rand_y = np.array(rand_y).reshape(label_data_train.shape)

        p_n_rate = max(100,pnRate(rand_y))

        train_dict = {input_data_train: rand_x, label_data_train: rand_y, input_ref_data_train: rand_ref, p_dropout: 0.5,
                      loss_weight: p_n_rate, is_train_step:True}

        sess.run(train_step, feed_dict=train_dict)
        temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        temp_train_stat = getStat(temp_train_preds, rand_y, num_grid=num_grid)

        loss_containor_for_mean.append(temp_train_loss)
        spec_containor_for_mean.append(temp_train_stat['spec'])
        if temp_train_stat['sens'] != -1:
            sens_containor_for_mean.append(temp_train_stat['sens'])

        # Recording results of test data
        if (i + 1) % eval_every == 0:
            eval_index = np.random.choice(len(test_data_list), size=batch_size)

            eval_x = test_data_list[eval_index[0]]['readCount'].values
            eval_x = eval_x.reshape(input_data_eval.shape)
            eval_ref = test_ref_list[eval_index[0]]['refGeneCount'].values
            eval_ref = eval_ref.reshape(input_ref_data_eval.shape)

            eval_y = test_label_list[eval_index[0]][['peak']].values.transpose()
            eval_y = np.repeat(eval_y, 5)
            eval_y = eval_y.reshape(label_data_eval.shape)

            pnRate_eval = max(100,pnRate(eval_y))

            test_dict = {input_data_eval: eval_x, label_data_eval: eval_y, input_ref_data_eval: eval_ref, p_dropout: 1,
                         loss_weight: pnRate_eval, is_train_step:False}

            test_preds = sess.run(test_prediction, feed_dict=test_dict)
            test_stat = getStat(test_preds, eval_y, num_grid=num_grid)

            loss_mean = sum(loss_containor_for_mean)/float(len(loss_containor_for_mean))
            if len(sens_containor_for_mean) == 0:
                sens_mean = -1.
            else:
                sens_mean = sum(sens_containor_for_mean)/float(len(sens_containor_for_mean))
            spec_mean = sum(spec_containor_for_mean)/float(len(spec_containor_for_mean))

            train_loss.append(loss_mean)
            train_spec.append(spec_mean)
            train_sens.append(sens_mean)
            test_spec.append(test_stat['spec'])
            test_sens.append(test_stat['sens'])

            loss_containor_for_mean.clear()
            spec_containor_for_mean.clear()
            sens_containor_for_mean.clear()


            if test_stat['sens'] == -1.0:
                logger.info('Generation # {}. TrainLoss: {:.2f}.  PNRate:--.--| Test: SENS:-.-- SPEC:{:.2f}| Train: SENS:{:.2f}, SPEC:{:.2f}\n'.
                        format(i+1, loss_mean, test_stat['spec'], sens_mean, spec_mean))
            else:
                logger.info('Generation # {}. TrainLoss: {:.2f}.  PNRate:{:.2f}| Test: SENS:{:.2f} SPEC:{:.2f}| Train: SENS:{:.2f}, SPEC:{:.2f}\n'.
                        format(i+1, loss_mean, pnRate_eval, test_stat['sens'], test_stat['spec'], sens_mean, spec_mean))

    visualizeTrainingProcess(eval_every, generations, test_sens, test_spec, train_sens, train_spec, train_loss, K_fold=str(step_num))
    visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list, test_label_list,
                        test_ref_list, test_prediction, k=len(test_data_list), K_fold=str(step_num))

    saver = tf.train.Saver()
    save_path = saver.save(sess, os.getcwd() + "/models/model{}.ckpt".format(step_num,step_num))
    logger.info("Model saved in path : %s" % save_path)


def peakPredictConvModel(input_data_depth, input_data_ref, logger=None):
    """
    Define structure of convolution model.

    :param logger:
    :param input_data:
    :return: Tensor of the output layer
    """
    input_data_depth = tf.nn.batch_normalization(input_data_depth, 0, 1, 0, 1, 0.9)

    #Stem of read depth data
    conv1 = tf.nn.conv1d(input_data_depth, conv1_weight, stride=1, padding='SAME')
    conv1_bn = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC')
    relu1 = tf.nn.relu(conv1_bn)#tf.nn.bias_add(conv1, conv1_bias))

    conv2 = tf.nn.conv1d(relu1, conv2_weight, stride=1, padding='SAME')
    conv2_bn = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC')
    relu2 = tf.nn.relu(conv2_bn)#tf.nn.bias_add(conv2, conv2_bias))
    max_pool1 = tf.nn.pool(relu2, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')

    #Stem of ref gene data
    conv1_ref = tf.nn.conv1d(input_data_ref, conv1_ref_weight, stride=1, padding='SAME')
    relu1_ref = tf.nn.relu(tf.nn.bias_add(conv1_ref, conv1_ref_bias))
    max_pool1_ref = tf.nn.pool(relu1_ref, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')

    input_concat = tf.concat([max_pool1, max_pool1_ref],axis = 2)

    # Inception modules 1 to 6
    concat1 = concatLayer_C(input_concat, conv1a_weight, convMax1_weight, conv1b_weight, convAvg1_weight, conv1c_weight, 3)

    concat2 = concatLayer_C(concat1, conv2a_weight, convMax2_weight, conv2b_weight, convAvg2_weight, conv2c_weight, max_pool_size2)
    concat2 = tf.nn.pool(concat2, [3], strides=[3], padding='SAME', pooling_type='AVG')

    concat3 = concatLayer_A(concat2, conv3a_weight, conv3b_weight, 2)

    concat4 = concatLayer_A(concat3, conv4a_weight, conv4b_weight, 2)

    concat5 = concatLayer_A(concat4, conv5a_weight, conv5b_weight, 2)

    concat6 = concatLayer_A(concat5, conv6a_weight, conv6b_weight, 5)

    final_conv_shape = concat6.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2]
    flat_output = tf.reshape(concat6, [final_conv_shape[0] , final_shape])

    fully_connected1 = tf.matmul(flat_output, full1_weight)
    fully_connected1 = tf.contrib.layers.batch_norm(fully_connected1, is_training=is_train_step)
    fully_connected1 = tf.nn.leaky_relu(fully_connected1, alpha=0.0005, name="FullyConnected1")
    #fully_connected1 = tf.nn.dropout(fully_connected1, keep_prob=p_dropout)
    print("Fully connected A :{}".format(fully_connected1.shape))

    fully_connected2 = tf.matmul(fully_connected1, full2_weight)
    fully_connected2 = tf.contrib.layers.batch_norm(fully_connected2, is_training=is_train_step)
    fully_connected2 = tf.nn.leaky_relu(fully_connected2, alpha=0.0005, name="FullyConnected2")
    #fully_connected2 = tf.nn.dropout(fully_connected2, keep_prob=p_dropout)
    print("Fully connected B :{}".format(fully_connected2.shape))

    final_threshold_output = (tf.add(tf.matmul(fully_connected2, output_weight), output_bias))
    print("Output :{}".format(final_threshold_output.shape))

    return final_threshold_output


def concatLayer_A(source_layer, conv1_w, conv2_w, pooling_size):
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
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC')
    relu1 = tf.nn.relu(conv1)#tf.nn.bias_add(conv1, conv1_b))

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC')
    relu2 = tf.nn.relu(conv2)#tf.nn.bias_add(conv2, conv2_b))

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='MAX')

    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='AVG')

    concat = tf.concat([relu1, avg_pool, relu2, max_pool], axis=2)
    print("Concat Type A :{}".format(concat.shape))
    return concat


def concatLayer_B(source_layer, conv1_w, conv_max_w, conv2_w, conv_avg_w, conv1_b, conv_max_b, conv2_b, conv_avg_b,
                  pooling_size):
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
    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=1, padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=1, padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[1],
            padding='SAME', pooling_type='MAX')
    conv_max = tf.nn.conv1d(max_pool, conv_max_w, stride=1, padding='SAME')
    relu_max = tf.nn.leaky_relu(tf.nn.bias_add(conv_max, conv_max_b))

    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[1],
            padding='SAME', pooling_type='AVG')
    conv_avg = tf.nn.conv1d(avg_pool, conv_avg_w, stride=1, padding='SAME')
    relu_avg = tf.nn.leaky_relu(tf.nn.bias_add(conv_avg, conv_avg_b))

    concat = tf.concat([relu1, relu_max, relu2, relu_avg], axis=2)
    print("Concat Type B :{}".format(concat.shape))
    return concat


def concatLayer_C(source_layer, conv1_w, conv_max_w, conv2_w, conv_avg_w, conv3_w, pooling_size):
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
    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=1, padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC')
    relu1 = tf.nn.relu(conv1)#tf.nn.bias_add(conv1, conv1_b))

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=1, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC')
    relu2 = tf.nn.relu(conv2)#tf.nn.bias_add(conv2, conv2_b))

    conv3 = tf.nn.conv1d(source_layer, conv3_w, stride=1, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train_step, data_format='NHWC')
    relu3 = tf.nn.relu(conv3)#tf.nn.bias_add(conv3, conv3_b))

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[1],
            padding='SAME', pooling_type='MAX')
    conv_max = tf.nn.conv1d(max_pool, conv_max_w, stride=1, padding='SAME')
    conv_max = tf.contrib.layers.batch_norm(conv_max, is_training=is_train_step, data_format='NHWC')
    relu_max = tf.nn.relu(conv_max)#tf.nn.bias_add(conv_max, conv_max_b))

    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[1],
            padding='SAME', pooling_type='AVG')
    conv_avg = tf.nn.conv1d(avg_pool, conv_avg_w, stride=1, padding='SAME')
    conv_avg = tf.contrib.layers.batch_norm(conv_avg, is_training=is_train_step, data_format='NHWC')
    relu_avg = tf.nn.relu(conv_avg)#tf.nn.bias_add(conv_avg, conv_avg_b))

    concat = tf.concat([relu1, avg_pool, relu2, max_pool, relu3], axis=2)
    print("Concat Type C :{}".format(concat.shape))
    return concat


def getStat(logits, targets, batch_size_in=batch_size, num_grid=0):
    """
    Return accuracy of the result.
    Acc = ( TP + TN ) / ( TP + TN + FN + FP )
    ( TP + TN + FN + FP ) = num_grid

    :param logits:
    :param targets:
    :return:
    """
    logits = logits.reshape(batch_size_in, num_grid)
    targets = targets.reshape(batch_size_in, num_grid)

    TP = 0.
    TN = 0.
    FN = 0.
    FP = 0.

    for i in range(batch_size_in):
        for index in range(len(logits[0])):
            if (logits[i][index]) >= class_threshold and targets[i][index] >= class_threshold:
                TP += 1
            elif (logits[i][index]) >= class_threshold and targets[i][index] < class_threshold:
                FP += 1
            elif (logits[i][index]) < class_threshold and targets[i][index] >= class_threshold:
                FN += 1
            elif (logits[i][index]) < class_threshold and targets[i][index] < class_threshold:
                TN += 1
            else:
                pass

    if TP+FN == 0:
        return {'sens': -1, 'spec': TN/(TN+FP), 'acc':(TP+TN)/(TP+TN+FN+FP)}
    else:
        return {'sens': TP/(TP+FN), 'spec': TN/(TN+FP), 'acc':(TP+TN)/(TP+TN+FN+FP)}


def pnRate(targets, batch_size_in=batch_size):
    """
    Return the The ratio of Negative#/ Positive#.
    It will be used for weights of loss function to adjust
    between sensitivity and specificity.

    :param targets:
    :param num_grid:
    :return:
    """
    positive_count = 0.

    for i in range(batch_size_in):
        for index in range(len(targets[i][0])):
            if targets[i][0][index] > 0:
                positive_count += 1

    # For the label only has negative samples.
    if positive_count == 0.:
        return 1

    negative_count = len(targets[0][0])*batch_size_in - positive_count

    return negative_count  / positive_count


def classValueFilter(output_value):
    """
    For output of model, probabilities of a final vector will be changed
    as binary values by checking whether elements of vector are higher or lower than
    class_threshold that defined in hyperparameters.py.

    :param output_value:
    :return: a binary vector that indicates having peak or not.
    """

    class_value_list = []

    for index in range(output_value.shape[2]):
        if output_value[0][0][index] >= class_threshold:
            class_value_list.append(1)
        elif output_value[0][0][index] < class_threshold:
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


def splitTrainingData(data_list, label_list, ref_list, Kfold=10):
    """

    If Kfold is zero, it just split two parts

    :param list_data:
    :param Kfold:
    :return:
    """
    print("################## THE NUMBER OF LABEL DATA : {}".format(len(data_list)))

    size = len(data_list)
    counter = size / Kfold

    test_data = []
    test_label = []
    test_ref = []

    for i in range(Kfold - 1):
        test_data_temp = []
        test_ref_temp = []
        test_label_temp = []
        while True:
            if counter <= 0:
                test_ref.append(test_ref_temp)
                test_data.append(test_data_temp)
                test_label.append(test_label_temp)
                counter = size // Kfold
                break

            pop_index = random.randint(0, len(data_list)-1)
            test_ref_temp.append(ref_list.pop(pop_index))
            test_data_temp.append(data_list.pop(pop_index))
            test_label_temp.append(label_list.pop(pop_index))
            counter -= 1

    test_data.append(data_list)
    test_ref.append(ref_list)
    test_label.append(label_list)

    return test_data, test_label, test_ref


def visualizeTrainingProcess(eval_every, generations, test_sens, test_spec, train_sens, train_spec, train_loss, K_fold =""):
    """
    Create matplotlib figures about a plot of loss function values and accuracy values.

    :param eval_every:
    :param generations:
    :param test_acc:
    :param train_acc:
    :param train_loss:
    :return:
    """
    eval_indices = range(0, generations, eval_every)

    for i in range(len(train_sens) - 1):
        if train_sens[i+1] == -1:
            train_sens[i+1] = train_sens[i]
        if test_sens[i+1] == -1:
            test_sens[i+1] = test_sens[i]

    plt.plot(eval_indices, train_loss, 'k-')
    plt.title('Cross entropy Loss per generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('models/model_{}/LossPerGen.png'.format(K_fold))
    plt.clf()

    plt.plot(eval_indices, test_sens, label='Test Set sensitivity')
    plt.plot(eval_indices, train_sens, label='Train Set sensitivity')
    plt.title('Train and Test Sensitivity')
    plt.xlabel('Generation')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('models/model_{}/SensPerGen.png'.format(K_fold))
    plt.clf()

    plt.plot(eval_indices, test_spec, label='Test Set specificity')
    plt.plot(eval_indices, train_spec, label='Train Set specificity')
    plt.title('Train and Test specificity')
    plt.xlabel('Generation')
    plt.ylabel('Specificity')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('models/model_{}/SpecPerGen.png'.format(K_fold))
    plt.clf()

    plt.plot(eval_indices, [test_sens[i] + test_spec[i] - 1 for i in range(len(test_sens))], label='Test Set Yosen`s Index')
    plt.plot(eval_indices, [train_sens[i] + train_spec[i] - 1 for i in range(len(train_sens))], label='Train Set Yosen`s Index')
    plt.title('Yosen`s Index')
    plt.xlabel('Generation')
    plt.ylabel('Yosen`s Index`')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('models/model_{}/YosensPerGen.png'.format(K_fold))
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


def generateOutput(threshold_tensor, depth_tensor, div=10, input_size=12000, batch_size_in=batch_size):
    """
    It generate

    :param threshold_tensor: This tensor represents read-depth thresholds which have size of 'div'
    :param depth_tensor: This tensor represents
    :param div:
    :return:
    """

    depth_tensor = tf.reshape(depth_tensor,[batch_size_in ,div, input_size//div])
    threshold_tensor = tf.reshape(threshold_tensor,[batch_size_in,div,1])

    result_tensor = tf.subtract(depth_tensor, threshold_tensor, name="results")
    result_tensor = tf.reshape(result_tensor,[batch_size_in, 1, input_size])

    return result_tensor


def visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list, test_label_list,
                        test_ref_list, test_prediction, k = 1, K_fold=""):
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
            show_x = test_data_list[i]['readCount'].values
            show_x = show_x.reshape(input_data_eval.shape)
            show_ref = test_ref_list[i]['refGeneCount'].values
            show_ref = show_ref.reshape(input_data_eval.shape)

            show_y = test_label_list[i][['peak']].values.transpose()
            show_y = np.repeat(show_y, 5)
            show_y = show_y.reshape(label_data_train.shape)
            show_dict = {input_data_eval: show_x, input_ref_data_eval: show_ref, label_data_eval: show_y, p_dropout: 1,
                         is_train_step: False}
            show_preds = sess.run(test_prediction, feed_dict=show_dict)

            show_preds = classValueFilter(show_preds)
            show_y = classValueFilter(show_y)

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
            plt.legend(loc='upper right')
            plt.show()
            plt.savefig('models/model_{}/peak{}.png'.format(K_fold,i))
            plt.clf()
