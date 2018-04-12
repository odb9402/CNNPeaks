import preProcessing.preProcessing as preProcessing
import buildModel.buildModel as buildModel
import tensorflow as tf
import numpy as np
import pysam
import os
import random
import string
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Process, Manager
from buildModel.defineModel import *

def run(input_bam, logger, window_size, num_grid=4000):
    """

    :param dir_name:
    :param logger:
    :param input_bam:
    :param window_size:
    :param num_grid:
    :return:
    """
    global num_peaks
    num_peaks = 0

    tf.reset_default_graph()

    input_data = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="testData")

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.getcwd() + "/model.ckpt")

    model_output = peakPredictConvModel(input_data, logger)
    prediction = tf.nn.sigmoid(model_output)
    ###################################################################################

    if not os.path.isdir(input_bam[:-4]):
        os.makedirs(input_bam[:-4])

    if not os.path.isfile(input_bam + '.sort.bai'):
        preProcessing.createBamIndex(input_bam)
        logger.info("Creating index file of [" + input_bam + "]")
    else:
        logger.info("[" + input_bam + "] already has index file.")

    input_bam = os.path.abspath(input_bam)

    processes = []

    MAX_CORE = cpu_count()

    for chr_no in range(22):
        call_peak(chr_no, input_bam, input_data, logger, num_grid, prediction, sess, window_size)
        #process = Process(target=call_peak,\
        #                  args=(chr_no, input_bam, input_data, logger, num_grid, prediction, sess, window_size,))
        #preProcessing.parallel_execution(MAX_CORE-1, process, processes)

    #for proc in processes:
    #    proc.join()


def call_peak(chr_no, input_bam, input_data, logger, num_grid, prediction, sess, window_size):

    window_count = 1
    bam_alignment = pysam.AlignmentFile(input_bam + '.sort', 'rb', index_filename=input_bam + '.sort.bai')
    bam_length = bam_alignment.lengths
    stride = window_size / num_grid
    eval_counter = 0

    while True:
        read_count_by_grid = []
        if window_count + window_size > bam_length[chr_no]:
            break
        for step in range(num_grid):
            count = bam_alignment.count(region=preProcessing.createRegionStr("chr" + str(chr_no + 1), \
                                                                             int(window_count + stride * step)))
            read_count_by_grid.append(count)
        read_count_by_grid = np.array(read_count_by_grid, dtype=int)
        read_count_by_grid = read_count_by_grid.reshape(input_data.shape)
        result_dict = {input_data: read_count_by_grid, p_dropout: 1, is_test: True}
        preds = sess.run(prediction, feed_dict=result_dict)
        class_value_prediction = buildModel.expandingPrediction(buildModel.classValueFilter(preds))
        predictionToBedString(input_bam, class_value_prediction, "chr" + str(chr_no + 1), window_count, stride,
                              num_grid, logger, read_count_by_grid.reshape(num_grid).tolist())
        eval_counter += 1
        if eval_counter == 100:
            logger.info("Reading . . . :[chr" + str(chr_no + 1) + ":" \
                        + str(window_count - (window_size * 99)) + "-" + str(window_count + window_size) + "]")
            eval_counter = 0
        window_count += window_size


def predictionToBedString(input_bam ,prediction, chromosome, region_start, stride, num_grid,\
                          logger, reads, min_peak_size=10):
    """

    :param prediction:
    :param logger:
    :return:
    """
    global num_peaks
    peak_size = 0
    step = 0
    peak_switch = False
    peaks = []

    output_file_name = input_bam.rsplit('.')[0] + "_" +str(chromosome) + ".bed"

    while True:
        if step > num_grid - 1:
            break
        if prediction[step] is 1:
            peak_size += 1
        else:
            if peak_size is not 0:
                if peak_size > min_peak_size:
                    end_point = region_start + ( stride * step )
                    start_point = end_point - ( peak_size * stride )
                    peak_size = 0
                    peak_switch = True
                    num_peaks += 1

                    logger.info("{}:{:,}-{:,} , # peaks :{}".format(chromosome, int(start_point), int(end_point), num_peaks))
                    peaks.append("{}\t{}\t{}\n".format(chromosome, int(start_point), int(end_point)))
                    writeBed(output_file_name, peaks)
                    peaks = []
                else:
                    peak_size = 0
        step += 1

    #if peak_switch is True:
    #    plt.plot(prediction, 'r.')
    #    plt.plot(reads)
    #    plt.show()


def writeBed(output_file, peaks):
    if not os.path.isfile(output_file):
        bed_file = open(output_file, 'w')
    else:
        bed_file = open(output_file, 'a')
    for peak in peaks:
        bed_file.write(peak)


def removeCentromere():
    pass


def peakPredictConvModel(input_data, logger):
    """

    :param logger:
    :param input_data:
    :return:
    """

    #input_data = tf.nn.batch_normalization(input_data,0,1.,0,1,0.00001)
    conv1 = tf.nn.conv1d(input_data, conv1_weight, stride=1, padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.pool(relu1, [max_pool_size_stem], strides=[max_pool_size_stem], padding='SAME', pooling_type='MAX')

    concat1 = buildModel.concatLayer_B(max_pool1, conv1a_weight, convMax1_weight, conv1b_weight, convAvg1_weight,\
                            conv1a_bias, convMax1_bias, conv1b_bias, convAvg1_bias, max_pool_size1)

    concat2 = buildModel.concatLayer_A(concat1, conv2a_weight, conv2b_weight, conv2a_bias, conv2b_bias, max_pool_size2)

    concat3 = buildModel.concatLayer_A(concat2, conv3a_weight, conv3b_weight, conv3a_bias, conv3b_bias, max_pool_size3)

    concat4 = buildModel.concatLayer_A(concat3, conv4a_weight, conv4b_weight, conv4a_bias, conv4b_bias, max_pool_size4)

    concat5 = buildModel.concatLayer_A(concat4, conv5a_weight, conv5b_weight, conv5a_bias, conv5b_bias, max_pool_size5)

    final_conv_shape = concat5.get_shape().as_list()
    final_shape = final_conv_shape[1] * final_conv_shape[2]
    flat_output = tf.reshape(concat5, [final_conv_shape[0] , final_shape])

    fully_connected1 = tf.nn.leaky_relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias)\
                                        ,alpha=0.003, name="FullyConnected1")
    fully_connected1 = tf.nn.dropout(fully_connected1, keep_prob=p_dropout)

    final_model_output = tf.add(tf.matmul(fully_connected1,full2_weight), full2_bias)
    final_model_output = tf.reshape(final_model_output,[batch_size, 1, target_size], name="FullyConnected2")

    return (final_model_output)


def visualizeEachLayers():
    pass


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))