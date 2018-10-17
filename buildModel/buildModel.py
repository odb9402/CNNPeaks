import glob
import os
import shutil

from scipy import stats
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from .defineModel import *
from .hyperparameters import *

def run(dir_name, logger, num_grid=0, K_fold_in=10, cross_valid = True):
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

    input_list = {}
    for dir in dir_list:
        dir = PATH + '/' + dir
        input_list[dir] = extractChrClass(dir)


    model_output  = peakPredictConvModel(input_data_train, input_ref_data_train, logger)[0]
    test_model_output = peakPredictConvModel(input_data_eval, input_ref_data_eval, logger)[0]

    prediction_before_sigmoid = (generateOutput(model_output, input_data_train, div=threshold_division))
    prediction = tf.nn.sigmoid(prediction_before_sigmoid)

    test_prediction = tf.nn.sigmoid((generateOutput(test_model_output, input_data_eval, div=threshold_division)))

    loss = aggregatedLoss(label_data_train, prediction_before_sigmoid)
    tf.summary.scalar("Loss",loss)

    sens, spec = getTensorStat(label_data_train, prediction)
    tf.summary.scalar("Train sens", sens)
    tf.summary.scalar("Train spec", spec)

    #test_sens, test_spec = getTensorStat(label_data_eval, test_prediction)
    #tf.summary.scalar("Test sens", test_sens)
    #tf.summary.scalar("Test spec", test_spec)


    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #################### Training start with cross validation ##################################
    input_data_names = []
    ref_data_names = []
    label_data_names = []

    for dir in input_list:
        label_num = 0
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                label_num += 1
                input_file_name = "{}/{}_{}_grid{}.ct".format(dir, chr, cls, num_grid)
                ref_file_name = "{}/ref_{}_{}_grid{}.ref".format(dir, chr, cls, num_grid)
                label_file_name = "{}/label_{}_{}_grid{}.lb".format(dir, chr, cls, num_grid)
                input_data_names.append(pd.read_csv(input_file_name))
                ref_data_names.append(pd.read_csv(ref_file_name))
                label_data_names.append(pd.read_csv(label_file_name))
        logger.info("DIRECTORY (TARGET) [{}]# of labels : <{}>".format(label_num,dir))

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

        training(sess, loss, prediction, test_prediction, train_step, training_data, training_label , training_ref, test_data, test_label, test_ref, logger, num_grid, i)

        if cross_valid == False:
            break

def training(sess, loss, prediction, test_prediction, train_step,
        train_data_list, train_label_list, train_ref_list, test_data_list, test_label_list, test_ref_list,
            logger, num_grid, step_num):
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

    #wrapTensorToTensorboard()

    LOG_DIR = os.path.join(os.path.dirname(__file__),'tensorLog')

    if os.path.exists(LOG_DIR) is False:
        os.mkdir(LOG_DIR)
    else:
        shutil.rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, graph=sess.graph)

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

        rand_x = np.array(rand_x).reshape(input_data_train.shape)
        rand_x = (rand_x - rand_x.mean())/rand_x.std()
        rand_x[np.isnan(rand_x)] = 0

        rand_ref = np.array(rand_ref).reshape(input_ref_data_train.shape)

        rand_y = np.array(rand_y).reshape(label_data_train.shape)

        train_dict = {input_data_train: rand_x, label_data_train: rand_y, input_ref_data_train: rand_ref,
                loss_weight:pnRate(rand_y), is_train_step:True}

        summary, _ =  sess.run([merged, train_step], feed_dict=train_dict)

        temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
        temp_train_stat = getStat(temp_train_preds, rand_y, num_grid=num_grid)

        loss_containor_for_mean.append(temp_train_loss)
        spec_containor_for_mean.append(temp_train_stat['spec'])
        if temp_train_stat['sens'] != -1:
            sens_containor_for_mean.append(temp_train_stat['sens'])

        writer.add_summary(summary, i)


        # Recording results of test data
        if (i + 1) % eval_every == 0:
            eval_index = np.random.choice(len(test_data_list), size=batch_size)

            eval_x = test_data_list[eval_index[0]]['readCount'].values
            eval_x = eval_x.reshape(input_data_eval.shape)
            eval_x = (eval_x - eval_x.mean())/eval_x.std()
            eval_x[np.isnan(eval_x)] = 0
            eval_ref = test_ref_list[eval_index[0]]['refGeneCount'].values
            eval_ref = eval_ref.reshape(input_ref_data_eval.shape)

            eval_y = test_label_list[eval_index[0]][['peak']].values.transpose()
            eval_y = np.repeat(eval_y, 5)
            eval_y = eval_y.reshape(label_data_eval.shape)

            pnRate_eval = pnRate(eval_y)

            test_dict = {input_data_eval: eval_x, label_data_eval: eval_y, input_ref_data_eval: eval_ref, is_train_step:False}

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

            #summary = sess.run
            #writer.add_summary(summary, i)

    visualizeTrainingProcess(eval_every, generations, test_sens, test_spec, train_sens, train_spec, train_loss, K_fold=str(step_num))
    visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list, test_label_list,
                        test_ref_list, test_prediction, k=len(test_data_list), K_fold=str(step_num))

    saver = tf.train.Saver()
    save_path = saver.save(sess, os.getcwd() + "/models/model{}.ckpt".format(step_num,step_num))
    logger.info("Model saved in path : %s" % save_path)


def peakPredictConvModel(input_data_depth, input_data_ref, logger=None):
    """
    Define structure of convolution model.

    It will return two final output tensor.
     1. Sub output tensor for training steps
     2. Final output tensor for training and test steps

    :param logger:
    :param input_data:
    :return: Tensor of the output layer
    """
    #input_data_depth = tf.nn.batch_normalization(input_data_depth, 0, 1, 0, 1, 0.9)

    #Stem of read depth data
    conv1 = tf.nn.conv1d(input_data_depth, conv1_weight, stride=1, padding='SAME')
    conv1_bn = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC')
    relu1 = tf.nn.relu(conv1_bn)#tf.nn.bias_add(conv1, conv1_bias))
    print("Stem 1 : {}".format(relu1.shape))

    conv2 = tf.nn.conv1d(relu1, conv2_weight, stride=1, padding='SAME')
    conv2_bn = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC')
    relu2 = tf.nn.relu(conv2_bn)#tf.nn.bias_add(conv2, conv2_bias))
    max_pool1 = tf.nn.pool(relu2, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')
    print("Stem 2 : {}".format(max_pool1.shape))

    #Stem of ref gene data
    conv1_ref = tf.nn.conv1d(input_data_ref, conv1_ref_weight, stride=1, padding='SAME')
    conv1_bn = tf.contrib.layers.batch_norm(conv1_ref, is_training=is_train_step, data_format="NHWC")
    relu1_ref = tf.nn.relu(conv1_ref)
    max_pool1_ref = tf.nn.pool(relu1_ref, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')
    print("Stem_Ref 1 : {}".format(max_pool1_ref.shape))

    input_concat = tf.concat([max_pool1, max_pool1_ref],axis = 2)

    """
    sub_final_conv_shape = input_concat.get_shape().as_list()
    sub_final_shape = sub_final_conv_shape[1]*sub_final_conv_shape[2]
    sub_flat_output = tf.reshape(input_concat, [sub_final_conv_shape[0], sub_final_shape])

    fully_connected_sub = tf.matmul(sub_flat_output, full_sub_weight)
    fully_connected_sub = tf.contrib.layers.batch_norm(fully_connected_sub, is_training=is_train_step)
    fully_connected_sub = tf.nn.leaky_relu(fully_connected_sub, alpha=0.000, name="FullyConnectedSub")
    print("Fully connected SUB :{}".format(fully_connected_sub.shape))

    sub_final_threshold_output = (tf.add(tf.matmul(fully_connected_sub, output_sub_weight), output_sub_bias))
    print("Sub Output :{}".format(sub_final_threshold_output.shape))
    """

    # Inception modules 1 to 6
    concat1 = concatLayer_C(input_concat, conv1a_weight, convMax1_weight, conv1b_weight, convAvg1_weight, conv1c_weight, 3)
    #concat1 = tf.nn.pool()

    concat2 = concatLayer_C(concat1, conv2a_weight, convMax2_weight, conv2b_weight, convAvg2_weight, conv2c_weight, 3)
    #concat2 = tf.nn.pool(concat2, [3], strides=[3], padding='SAME', pooling_type='MAX')

    concat3 = concatLayer_A(concat2, conv3a_weight, conv3b_weight, conv3c_weight, 2)

    concat4 = concatLayer_A(concat3, conv4a_weight, conv4b_weight, conv4c_weight,2)

    concat5 = concatLayer_A(concat4, conv5a_weight, conv5b_weight, conv5c_weight, 5)

    concat6 = concatLayer_B(concat5, conv6a_weight, conv6b_weight, conv6c_weight, conv6d_weight, 3)

    final_conv_shape = concat6.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2]
    flat_output = tf.reshape(concat6, [final_conv_shape[0] , final_shape])

    fully_connected1 = tf.matmul(flat_output, full1_weight)
    fully_connected1 = tf.contrib.layers.batch_norm(fully_connected1, is_training=is_train_step)
    fully_connected1 = tf.nn.leaky_relu(fully_connected1, alpha=0.000, name="FullyConnected1")
    print("Fully connected A :{}".format(fully_connected1.shape))

    fully_connected2 = tf.matmul(fully_connected1, full2_weight)
    fully_connected2 = tf.contrib.layers.batch_norm(fully_connected2, is_training=is_train_step)
    fully_connected2 = tf.nn.leaky_relu(fully_connected2, alpha=0.000, name="FullyConnected2")
    print("Fully connected B :{}".format(fully_connected2.shape))

    final_threshold_output = (tf.add(tf.matmul(fully_connected2, output_weight), output_bias))
    print("Output :{}".format(final_threshold_output.shape))

    return final_threshold_output #, sub_final_threshold_output


def concatLayer_A(source_layer, conv1_w, conv2_w, conv3_w, pooling_size):
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

    conv3 = tf.nn.conv1d(source_layer, conv3_w, stride=pooling_size, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train_step, data_format='NHWC')
    relu3 = tf.nn.relu(conv3)#tf.nn.bias_add(conv2, conv2_b))

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[pooling_size],
            padding='SAME', pooling_type='MAX')

    concat = tf.concat([relu1, relu2, relu3,max_pool], axis=2)
    print("Concat Type A :{}".format(concat.shape))
    return concat


def concatLayer_B(source_layer, conv1_w, conv2_w, conv3_w, conv4_w, pooling_size):
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
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC')
    relu1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=pooling_size, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC')
    relu2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv1d(source_layer, conv3_w, stride=pooling_size, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train_step, data_format='NHWC')
    relu3 = tf.nn.relu(conv3)

    conv4 = tf.nn.conv1d(source_layer, conv4_w, stride=pooling_size, padding='SAME')
    conv4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train_step, data_format='NHWC')
    relu4 = tf.nn.relu(conv4)

    concat = tf.concat([relu1, relu2, relu3, relu4], axis=2)
    print("Concat Type B :{}".format(concat.shape))
    return concat


def concatLayer_C(source_layer, conv1_w, conv_max_w, conv2_w, conv_avg_w, conv3_w, pooling_size):
    """
    Define concat layer which like Inception module.

    :param source_layer:
    :param pooling_size:
    :return:
    """
    conv1 = tf.nn.conv1d(source_layer, conv1_w, stride=2, padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train_step, data_format='NHWC')
    relu1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv1d(source_layer, conv2_w, stride=2, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train_step, data_format='NHWC')
    relu2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv1d(source_layer, conv3_w, stride=2, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train_step, data_format='NHWC')
    relu3 = tf.nn.relu(conv3)

    max_pool = tf.nn.pool(source_layer, [pooling_size], strides=[2],
            padding='SAME', pooling_type='MAX')
    conv_max = tf.nn.conv1d(max_pool, conv_max_w, stride=1, padding='SAME')
    conv_max = tf.contrib.layers.batch_norm(conv_max, is_training=is_train_step, data_format='NHWC')
    relu_max = tf.nn.relu(conv_max)

    avg_pool = tf.nn.pool(source_layer, [pooling_size], strides=[2],
            padding='SAME', pooling_type='AVG')
    conv_avg = tf.nn.conv1d(avg_pool, conv_avg_w, stride=1, padding='SAME')
    conv_avg = tf.contrib.layers.batch_norm(conv_avg, is_training=is_train_step, data_format='NHWC')
    relu_avg = tf.nn.relu(conv_avg)

    concat = tf.concat([relu1, avg_pool, relu2, max_pool, relu3], axis=2)
    #concat = tf.concat([relu1, relu_avg, relu2, relu_max, relu3], axis=2)
    print("Concat Type C :{}".format(concat.shape))
    return concat


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


def aggregatedLoss(label_data_train, prediction_before_sigmoid):
    """

    """
    loss_a = tf.reduce_mean(tf.nn.top_k(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train, logits=prediction_before_sigmoid,
        pos_weight=loss_weight/4), k = topK_set_a).values)
    tf.summary.scalar("Top {} Loss".format(topK_set_a),loss_a)

    loss_b = tf.reduce_mean(tf.nn.top_k(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train, logits=prediction_before_sigmoid,
        pos_weight=loss_weight/4), k = topK_set_b).values)
    tf.summary.scalar("Top {} Loss".format(topK_set_b),loss_b)

    loss_c = tf.reduce_mean(tf.nn.top_k(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train, logits=prediction_before_sigmoid,
        pos_weight=loss_weight/4), k = topK_set_c).values)
    tf.summary.scalar("Top {} Loss".format(topK_set_c),loss_c)

    loss_d = tf.reduce_mean(tf.nn.top_k(tf.nn.weighted_cross_entropy_with_logits(targets=label_data_train, logits=prediction_before_sigmoid,
        pos_weight=loss_weight/4), k = topK_set_d).values)
    tf.summary.scalar("Top {} Loss".format(topK_set_d),loss_d)

    return loss_a + loss_b + loss_c + loss_d


def getTensorStat(logits, targets, batch_size_in=batch_size, num_grid=0):
    """
    Return accuracy of the result.
    Acc = ( TP + TN ) / ( TP + TN + FN + FP )
    ( TP + TN + FN + FP ) = num_grid

    :param logits:
    :param targets:
    :return:
    """

    print(targets[0][0].shape)

    threshold_tensor = tf.constant(class_threshold)

    sensDummy = tf.constant(-1.)

    for i in range(batch_size_in):
        TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.less_equal(threshold_tensor, logits[i][0]),
                tf.less_equal(threshold_tensor, targets[i][0])), dtype=tf.float32))

        TN = tf.reduce_sum(tf.cast(tf.logical_and(tf.less_equal(logits[i][0],threshold_tensor),
                tf.less_equal(targets[i][0], threshold_tensor)), dtype=tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(tf.less(logits[i][0], threshold_tensor),
                tf.less(threshold_tensor, targets[i][0])), dtype=tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.less(threshold_tensor, logits[i][0]),
                tf.less(targets[i][0], threshold_tensor)), dtype=tf.float32))

    sens = tf.cond( tf.equal(tf.add(TP,FN),0) , true_fn=lambda:sensDummy, false_fn=lambda:tf.divide(TP,tf.add(TP,FN)))
    spec = tf.divide(TN,tf.add(TN,FP))

    return sens, spec


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

    ### TODO :: adjust these SHITTY EQUATION.
    return negative_count / positive_count


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


def visualizePeakResult(batch_size, input_data_eval, num_grid, label_data_eval, sess, test_data_list, test_label_list,
                        test_ref_list, test_prediction, k = 1, K_fold="", min_peak_size=10, max_peak_num=50):
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
            show_x = (show_x-show_x.mean())/show_x.std()
            show_x[np.isnan(show_x)] = 0
            show_ref = test_ref_list[i]['refGeneCount'].values
            show_ref = show_ref.reshape(input_data_eval.shape)

            show_y = test_label_list[i][['peak']].values.transpose()
            show_y = np.repeat(show_y, 5)
            show_y = show_y.reshape(label_data_train.shape)
            show_dict = {input_data_eval: show_x, input_ref_data_eval: show_ref, label_data_eval: show_y,
                         is_train_step: False}
            show_preds = sess.run(test_prediction, feed_dict=show_dict)

            show_preds = classValueFilter(show_preds)
            show_y = classValueFilter(show_y)

            for index in range(len(show_preds)):
                if show_preds[index] > 0:
                    show_preds[index] += 1

            ############# Peak post processing ##########
            peak_num = 0

            if show_preds[0] > 0:
                peak_size = 1
                peak_num += 1
            else:
                peak_size = 0

            for pred_index in range(len(show_preds)-1):
                if show_preds[pred_index+1] > 0:
                    peak_size += 1
                else:
                    if peak_size < min_peak_size:
                        for j in range(peak_size):
                            show_preds[pred_index-j] = 0
                        peak_size=0
                    else:
                        peak_num += 1

            if peak_num < max_peak_num:
                for k in range(len(show_preds)):
                    show_preds[k] = 0
            #############################################

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
    plt.ylim([0,1])
    plt.show()
    plt.savefig('models/model_{}/SensPerGen.png'.format(K_fold))
    plt.clf()

    plt.plot(eval_indices, test_spec, label='Test Set specificity')
    plt.plot(eval_indices, train_spec, label='Train Set specificity')
    plt.title('Train and Test specificity')
    plt.xlabel('Generation')
    plt.ylabel('Specificity')
    plt.legend(loc='lower right')
    plt.ylim([0,1])
    plt.show()
    plt.savefig('models/model_{}/SpecPerGen.png'.format(K_fold))
    plt.clf()

    plt.plot(eval_indices, [test_sens[i] + test_spec[i] - 1 for i in range(len(test_sens))], label='Test Set Yosen`s Index')
    plt.plot(eval_indices, [train_sens[i] + train_spec[i] - 1 for i in range(len(train_sens))], label='Train Set Yosen`s Index')
    plt.title('Yosen`s Index')
    plt.xlabel('Generation')
    plt.ylabel('Yosen`s Index`')
    plt.legend(loc='lower right')
    plt.ylim([0,1])
    plt.show()
    plt.savefig('models/model_{}/YosensPerGen.png'.format(K_fold))
    plt.clf()
