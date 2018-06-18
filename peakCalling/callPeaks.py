import preProcessing.preProcessing as preProcessing
import buildModel.buildModel as buildModel
import tensorflow as tf
import numpy as np
import pandas as pd
import pysam
import os
import random
import string
import matplotlib.pyplot as plt
import subprocess as sp
import progressbar as pgb

from multiprocessing import cpu_count, Process, Manager
from buildModel.hyperparameters import *
from buildModel.defineModel import *


def run(input_bam, logger, window_size=100000, num_grid=4000):
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

    #tf.reset_default_graph()

    input_data = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="testData")

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.getcwd() + "/models/model0.ckpt")

    model_output = buildModel.peakPredictConvModel(input_data, logger)
    prediction = tf.nn.sigmoid(model_output)
    ###################################################################################

    if not os.path.isdir(input_bam[:-4]):
        os.makedirs(input_bam[:-4])

    if not os.path.isfile(input_bam + '.bai'):
        logger.info("Creating index file of [{}]".format(input_bam))
        preProcessing.createBamIndex(input_bam)
        logger.info("[{} was created.]".format(input_bam+".bai"))
    else:
        logger.info("[" + input_bam + "] already has index file.")

    input_bam = os.path.abspath(input_bam)

    #processes = []

    MAX_CORE = cpu_count()

    bam_alignment = pysam.AlignmentFile(input_bam , 'rb', index_filename=input_bam + '.bai')

    for chr_no in range(22):
        logger.info("Peak calling in chromosome chr{}:".format(chr_no + 1))
        call_peak(chr_no, bam_alignment, input_bam, input_data, logger, num_grid, prediction, sess, window_size)
        #process = Process(target=call_peak,\
        #                  args=(chr_no, input_bam, input_data, logger, num_grid, prediction, sess, window_size,))
        #preProcessing.parallel_execution(MAX_CORE-1, process, processes)

    #for proc in processes:
    #    proc.join()


def call_peak(chr_no, bam_alignment, file_name, input_data, logger, num_grid, prediction, sess, window_size):
    """

    :param chr_no: Chromosome number of regions
    :param bam_alignment: pysam bam alignment class
    :param file_name: file name of bam file
    :param input_data:
    :param logger:
    :param num_grid:
    :param prediction:
    :param sess:
    :param window_size:
    :return:
    """
    window_count = 1
    bam_length = bam_alignment.lengths
    stride = window_size / num_grid
    eval_counter = 0

    output_file_name = "{}.bed".format(file_name.rsplit('.')[0])
    peaks = []

    while True:
        if (eval_counter % 100) == 0:
            bar = pgb.ProgressBar(max_value=100)
            logger.info("Reading . . . :[chr{}:{}-{}]".format(chr_no+1,window_count,window_count+window_size*100))

        if window_count + window_size > bam_length[chr_no]:
            logger.info("Reading . . . :[chr{}:{}-{}]".format(chr_no+1,window_count-(window_size*(eval_counter -1)), window_count + window_size))
            writeBed(output_file_name, peaks, logger, printout=True)
            break

        read_count_by_grid = generateReadcounts(input_data, window_count, window_count + window_size, chr_no, file_name, num_grid)

        result_dict = {input_data: read_count_by_grid, p_dropout: 1, is_test: True}
        preds = sess.run(prediction, feed_dict=result_dict)
        class_value_prediction = buildModel.expandingPrediction(buildModel.classValueFilter(preds))

        peaks += predictionToBedString(class_value_prediction, "chr" + str(chr_no + 1), window_count, stride,
                              num_grid, logger, read_count_by_grid.reshape(num_grid).tolist())

        #visualizeEachLayers(input_bam, read_count_by_grid, sess, logger)

        bar.update(eval_counter)
        eval_counter += 1

        if eval_counter == 100:
            writeBed(output_file_name, peaks, logger, printout=False)
            eval_counter = 0
            peaks =[]

        window_count += window_size


def generateReadcounts(input_data, region_start, region_end, chr_no, file_name, num_grid):
    read_count_by_grid = []
    stride = (region_end - (region_start + 1)) / num_grid

    samtools_call = ['samtools depth -aa -r {} {} > tmp_depth'.format(
        preProcessing.createRegionStr("chr{}".format(chr_no + 1), int(region_start),int(region_end - 1)), file_name)]
    sp.call(samtools_call, shell=True)

    depth_data = pd.read_table('tmp_depth', header=None, usecols=[2], names=['readCount'])

    for step in range(num_grid):
        read_count_by_grid.append(depth_data['readCount'][int(step * stride)])

    read_count_by_grid = np.array(read_count_by_grid, dtype=float)
    read_count_by_grid = read_count_by_grid.reshape(input_data.shape)
    #read_count_by_grid = np.maximum(read_count_by_grid - np.sqrt(np.mean(read_count_by_grid)), 0)

    return read_count_by_grid


def predictionToBedString(prediction, chromosome, region_start, stride,
        num_grid,logger, reads, min_peak_size=8, max_peak_num=15):
    """
    Python list "prediction" which has binary values will will be changed
    as bed-file string. There are two conditions to accept as peak for each
    prediction.
    
    1. Peak size must be higher than min_peak_size.
    2. The number of peak in the single window cannot be higher than max_peak_num.

    If "predicition" cannot satisfy these conditions, the function return empty list.

    :param prediction:
    :param logger:
    :return:
    """
    global num_peaks
    peak_size = 0
    peaks = []
    num_peaks_in_window = 0

    for step in range(num_grid):
        if prediction[step] is 1:
            peak_size += 1
        else:
            if peak_size is not 0:
                # Condition 2: peak size should be higher than min_peak_size
                if peak_size > min_peak_size:
                    end_point = region_start + ( stride * step )
                    start_point = end_point - ( peak_size * stride )

                    avg_depth = np.mean(reads[step - peak_size : step])

                    peaks.append("{}\t{}\t{}\t{}\t{}\t{}\n".format(chromosome, int(start_point), int(end_point),
                        "chr{}_{:5}".format(chromosome,random.randint(0,100000)), avg_depth, '.'))
                    num_peaks_in_window += 1

                peak_size = 0

    if len(peaks) > max_peak_num:
        return []
    else:
        num_peaks += num_peaks_in_window
        return peaks
    #if peak_switch is True:
    #    plt.plot(prediction, 'r.')
    #    plt.plot(reads)
    #    plt.show()


def writeBed(output_file, peaks, logger, printout=False):
    global num_peaks

    if not os.path.isfile(output_file):
        bed_file = open(output_file, 'w')
    else:
        bed_file = open(output_file, 'a')

    if printout == True:
        logger.info("# peaks:{}, #peaks in window: {}".format(num_peaks, len(peaks)))

    for peak in peaks:
        bed_file.write(peak)
        if printout == True:
            logger.info("{}".format(peak[:-1]))


def removeCentromere():
    pass


def visualizeEachLayers(input_bam, input_counts, sess, logger):
    dir_name = os.getcwd() + "/fig/Layers_"

    if not os.path.isdir(os.getcwd()+"/fig"):
        os.mkdir(os.getcwd()+"/fig")
        if not os.path.isdir(dir_name + "STEM"):
            os.mkdir(dir_name + "STEM")
        for i in range(6):
            if not os.path.isdir(dir_name + str(i+1)):
                os.mkdir(dir_name + str(i+1))


    plt.plot(input_counts.reshape(8000), label='Input read counts')
    plt.savefig('{}/{}.png'.format(dir_name + "STEM",'Inputs'))
    plt.clf()

    input_counts = input_counts - np.mean(input_counts)

    conv1 = tf.nn.conv1d(input_data_eval, conv1_weight, stride=1, padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.pool(relu1, [max_pool_size_stem], strides=[max_pool_size_stem],
            padding='SAME', pooling_type='MAX')

    result_dict = {input_data_eval: input_counts, p_dropout: 1}
    result = sess.run(max_pool1, feed_dict=result_dict)
    layer_0 = np.split(result, 8, axis=2)

    savePatternFig(dir_name+"STEM", layer_0, 8, 4000, "STEM")

    ### Concat 1 ###
    conv1a = tf.nn.conv1d(max_pool1, conv1a_weight, stride=max_pool_size1, padding='SAME')
    relu1a = tf.nn.relu(tf.nn.bias_add(conv1a, conv1a_bias))

    conv1b = tf.nn.conv1d(max_pool1, conv1b_weight, stride=max_pool_size1, padding='SAME')
    relu1b = tf.nn.relu(tf.nn.bias_add(conv1b, conv1b_bias))

    max_pool_1 = tf.nn.pool(max_pool1, [max_pool_size1], strides=[max_pool_size1],
            padding='SAME', pooling_type='MAX')

    avg_pool_1 = tf.nn.pool(max_pool1, [max_pool_size1], strides=[max_pool_size1],
            padding='SAME', pooling_type='AVG')

    concat_1 = tf.concat([relu1a,max_pool_1,relu1b,avg_pool_1],axis=2)

    layer_avg1 = np.split(sess.run(max_pool_1,feed_dict=result_dict), layer1_width, axis=2)
    layer_max1 = np.split(sess.run(avg_pool_1,feed_dict=result_dict), layer1_width, axis=2)
    layer_1b = np.split(sess.run(relu1a,feed_dict=result_dict), conv1b_features, axis=2)
    layer_1a = np.split(sess.run(relu1b,feed_dict=result_dict), conv1a_features, axis=2)

    savePatternFig(dir_name+"1", layer_1a, conv1b_features, 2000, "Conv1_a")
    savePatternFig(dir_name+"1", layer_1b, conv1a_features, 2000, "Conv1_b")
    savePatternFig(dir_name+"1", layer_avg1, layer1_width, 2000, "Conv1_max")
    savePatternFig(dir_name+"1", layer_max1, layer1_width, 2000, "Conv1_avg")

    conv2a = tf.nn.conv1d(concat_1, conv2a_weight, stride=max_pool_size2, padding='SAME')
    relu2a = tf.nn.relu(tf.nn.bias_add(conv2a, conv2a_bias))

    conv2b = tf.nn.conv1d(concat_1, conv2b_weight, stride=max_pool_size2, padding='SAME')
    relu2b = tf.nn.relu(tf.nn.bias_add(conv2b, conv2b_bias))

    max_pool_2 = tf.nn.pool(concat_1, [max_pool_size2], strides=[max_pool_size2],
                          padding='SAME', pooling_type='MAX')

    avg_pool_2 = tf.nn.pool(concat_1, [max_pool_size2], strides=[max_pool_size2],
                          padding='SAME', pooling_type='AVG')

    layer_avg2 = np.split(sess.run(avg_pool_2,feed_dict=result_dict), layer2_width, axis=2)
    layer_max2 = np.split(sess.run(max_pool_2,feed_dict=result_dict), layer2_width, axis=2)
    layer_2b = np.split(sess.run(relu2a,feed_dict=result_dict), conv2b_features, axis=2)
    layer_2a = np.split(sess.run(relu2b,feed_dict=result_dict), conv2a_features, axis=2)

    savePatternFig(dir_name+"2", layer_2a, conv2a_features, 1000, "Conv2_a")
    savePatternFig(dir_name+"2", layer_2b, conv2b_features, 1000, "Conv2_b")
    savePatternFig(dir_name+"2", layer_avg2, layer2_width, 1000, "avg2")
    savePatternFig(dir_name+"2", layer_max2, layer2_width, 1000, "max2")

    concat_2 = tf.concat([relu2a, max_pool_2, relu2b, avg_pool_2], axis=2)

    conv3a = tf.nn.conv1d(concat_2, conv3a_weight, stride=max_pool_size3, padding='SAME')
    relu3a = tf.nn.relu(tf.nn.bias_add(conv3a, conv3a_bias))

    conv3b = tf.nn.conv1d(concat_2, conv3b_weight, stride=max_pool_size3, padding='SAME')
    relu3b = tf.nn.relu(tf.nn.bias_add(conv3b, conv3b_bias))

    max_pool_3 = tf.nn.pool(concat_2, [max_pool_size3], strides=[max_pool_size3],
                            padding='SAME', pooling_type='MAX')

    avg_pool_3 = tf.nn.pool(concat_2, [max_pool_size3], strides=[max_pool_size3],
                            padding='SAME', pooling_type='AVG')

    layer_avg3 = np.split(sess.run(avg_pool_3, feed_dict=result_dict), layer3_width, axis=2)
    layer_max3 = np.split(sess.run(max_pool_3, feed_dict=result_dict), layer3_width, axis=2)
    layer_3a = np.split(sess.run(relu3a, feed_dict=result_dict), conv3a_features, axis=2)
    layer_3b = np.split(sess.run(relu3b, feed_dict=result_dict), conv3b_features, axis=2)

    savePatternFig(dir_name + "3", layer_3a, conv3a_features, 500, "Conv3_a")
    savePatternFig(dir_name + "3", layer_3b, conv3b_features, 500, "Conv3_b")
    savePatternFig(dir_name + "3", layer_avg3, layer3_width, 500, "avgx3")
    savePatternFig(dir_name + "3", layer_max3, layer3_width, 500, "max3")

    concat_3 = tf.concat([relu3a, max_pool_3, relu3b, avg_pool_3], axis=2)

    conv4a = tf.nn.conv1d(concat_3, conv4a_weight, stride=max_pool_size4, padding='SAME')
    relu4a = tf.nn.relu(tf.nn.bias_add(conv4a, conv4a_bias))

    conv4b = tf.nn.conv1d(concat_3, conv4b_weight, stride=max_pool_size4, padding='SAME')
    relu4b = tf.nn.relu(tf.nn.bias_add(conv4b, conv4b_bias))

    max_pool_4 = tf.nn.pool(concat_3, [max_pool_size4], strides=[max_pool_size4],
                          padding='SAME', pooling_type='MAX')

    avg_pool_4 = tf.nn.pool(concat_3, [max_pool_size4], strides=[max_pool_size4],
                          padding='SAME', pooling_type='AVG')

    layer_avg4 = np.split(sess.run(avg_pool_4,feed_dict=result_dict), layer4_width, axis=2)
    layer_max4 = np.split(sess.run(max_pool_4,feed_dict=result_dict), layer4_width, axis=2)
    layer_4b = np.split(sess.run(relu4a,feed_dict=result_dict), conv4b_features, axis=2)
    layer_4a = np.split(sess.run(relu4b,feed_dict=result_dict), conv4a_features, axis=2)

    savePatternFig(dir_name+"4", layer_4a, conv4a_features, 250, "Conv4_a")
    savePatternFig(dir_name+"4", layer_4b, conv4b_features, 250, "Conv4_b")
    savePatternFig(dir_name+"4", layer_avg4, layer4_width, 250, "max4")
    savePatternFig(dir_name+"4", layer_max4, layer4_width, 250, "avg4")

    concat_4 = tf.concat([relu4a, max_pool_4, relu4b, avg_pool_4], axis=2)

    conv5a = tf.nn.conv1d(concat_4, conv5a_weight, stride=max_pool_size5, padding='SAME')
    relu5a = tf.nn.relu(tf.nn.bias_add(conv5a, conv5a_bias))

    conv5b = tf.nn.conv1d(concat_4, conv5b_weight, stride=max_pool_size5, padding='SAME')
    relu5b = tf.nn.relu(tf.nn.bias_add(conv5b, conv5b_bias))

    max_pool_5 = tf.nn.pool(concat_4, [max_pool_size5], strides=[max_pool_size5],
                            padding='SAME', pooling_type='MAX')

    avg_pool_5 = tf.nn.pool(concat_4, [max_pool_size5], strides=[max_pool_size5],
                            padding='SAME', pooling_type='AVG')

    layer_avg5 = np.split(sess.run(avg_pool_5, feed_dict=result_dict), layer5_width, axis=2)
    layer_max5 = np.split(sess.run(max_pool_5, feed_dict=result_dict), layer5_width, axis=2)
    layer_5b = np.split(sess.run(relu5a, feed_dict=result_dict), conv5b_features, axis=2)
    layer_5a = np.split(sess.run(relu5b, feed_dict=result_dict), conv5a_features, axis=2)

    savePatternFig(dir_name + "5", layer_5a, conv5a_features, 125, "Conv5_a")
    savePatternFig(dir_name + "5", layer_5b, conv5b_features, 125, "Conv5_b")
    savePatternFig(dir_name + "5", layer_avg5, layer5_width, 125, "max5")
    savePatternFig(dir_name + "5", layer_max5, layer5_width, 125, "avg5")


def savePatternFig(dir_name, layer_0, feature, wide, name):
    for i in range(feature):
        plt.plot(layer_0[i].reshape(wide), label=name)
        plt.savefig('{}/{}_{}.png'.format(dir_name, name, i))
        plt.clf()


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
