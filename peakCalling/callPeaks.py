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


def run(input_bam, logger, window_size=100000, num_grid=0, model_num=1):
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
    input_data_ref = tf.placeholder(tf.float32, shape=(batch_size, num_grid, 1), name="TestRefData")

    sess = tf.Session()

    #model_output = peakPredictConvModel(input_data, input_data_ref, logger)[0]
    #prediction = tf.nn.sigmoid(generateOutput(model_output, input_data, div=threshold_division))

    model_output = test_model_output
    prediction = test_prediction

    saver = tf.train.Saver()
    saver.restore(sess, os.getcwd() + "/models/model{}.ckpt".format(model_num))
    logger.info("{}` th model will be used during peak calling. . . ".format(model_num))

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

    bam_alignment = pysam.AlignmentFile(input_bam , 'rb', index_filename=input_bam + '.bai')
    chr_lengths = bam_alignment.lengths

    for chr_no in range(22):
        #chr_no+=6
        ref_data_df = pd.read_table("geneRef/chr{}.bed".format(chr_no + 1), names=['chr','start','end'] , header=None, usecols=[0,1,2])
        logger.info("Peak calling in chromosome chr{}:".format(chr_no + 1))
        call_peak(chr_no, chr_lengths, input_bam, ref_data_df, input_data, input_data_ref,
                logger, num_grid, prediction, sess, window_size)


def call_peak(chr_no, chr_lengths, file_name, ref_data_df, input_data, input_data_ref, logger, num_grid, prediction, sess, window_size, pgb_on=False):
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
    stride = window_size / num_grid
    eval_counter = 0
    output_file_name = "{}.bed".format(file_name.rsplit('.')[0])
    peaks = []

    while True:
        if (eval_counter % 100) == 0:
            if pgb_on:
                bar = pgb.ProgressBar(max_value=100)
            logger.info("Reading . . . :[chr{}:{}-{}]".format(chr_no+1,window_count,window_count+window_size*100))

        if window_count + window_size > chr_lengths[chr_no]:
            logger.info("Reading . . . :[chr{}:{}-{}]".format(chr_no+1,window_count-(window_size*(eval_counter -1)), window_count + window_size))
            writeBed(output_file_name, peaks, logger, printout=True)
            break

        read_count_by_grid = generateReadcounts(input_data, window_count, window_count + window_size, chr_no, file_name, num_grid).reshape(input_data_eval.shape)
        ref_data_by_grid = generateRefcounts(input_data_ref, window_count, window_count + window_size, chr_no, ref_data_df, num_grid).reshape(input_ref_data_eval.shape)

        result_dict = {input_data_eval: read_count_by_grid, input_ref_data_eval: ref_data_by_grid, is_train_step: False}
        preds = sess.run(prediction, feed_dict=result_dict)
        class_value_prediction = buildModel.classValueFilter(preds)

        peaks += predictionToBedString(class_value_prediction, "chr" + str(chr_no + 1), window_count, stride,
                              num_grid, logger, read_count_by_grid.reshape(num_grid).tolist())
        if pgb_on:
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
    FNULL = open(os.devnull, 'w')
    sp.call(samtools_call, shell=True, stdout=FNULL, stderr=sp.STDOUT)

    depth_data = pd.read_table('tmp_depth', header=None, usecols=[2], names=['readCount'])

    for step in range(num_grid):
        read_count_by_grid.append(depth_data['readCount'][int(step * stride)])

    read_count_by_grid = np.array(read_count_by_grid, dtype=float)
    read_count_by_grid = read_count_by_grid.reshape(input_data.shape)

    return read_count_by_grid


def generateRefcounts(input_data_ref, region_start, region_end, chr_no, refGene_df, num_grid):
    stride = (region_end - (region_start + 1)) / num_grid

    sub_refGene = preProcessing.makeRefGeneTags(refGene_df[(refGene_df['start'] > region_start)&(refGene_df['end'] < region_end)],
            region_start, region_end, stride, num_grid)

    sub_refGene = np.array(sub_refGene, dtype=float)
    sub_refGene = sub_refGene.reshape(input_data_ref.shape)

    return sub_refGene


def predictionToBedString(prediction, chromosome, region_start, stride,
        num_grid,logger, reads, min_peak_size=10, max_peak_num=50):
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
                # Condition 1: peak size should be higher than min_peak_size.
                if peak_size > min_peak_size:
                    end_point = region_start + ( stride * step )
                    start_point = end_point - ( peak_size * stride )

                    min_depth = np.amin(reads[step - peak_size : step])
                    avg_depth = np.mean(reads[step - peak_size : step])
                    max_depth = np.amax(reads[step - peak_size : step])

                    peaks.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(chromosome, int(start_point), int(end_point), "{}_{:5}".format(chromosome,random.randint(0,100000)), min_depth, avg_depth, max_depth))
                    num_peaks_in_window += 1

                peak_size = 0

    # Condition 2 : The number of peaks must be lower than max_peak_num.
    if len(peaks) > max_peak_num:
        return []
    else:
        num_peaks += num_peaks_in_window
        return peaks


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


def savePatternFig(dir_name, layer_0, feature, wide, name):
    for i in range(feature):
        plt.plot(layer_0[i].reshape(wide), label=name)
        plt.savefig('{}/{}_{}.png'.format(dir_name, name, i))
        plt.clf()


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
