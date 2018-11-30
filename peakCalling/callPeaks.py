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
import time
from readCounts import generateReadcounts, generateRefcounts

from multiprocessing import cpu_count, Process, Manager
from buildModel.hyperparameters import *
from buildModel.defineModel import *


def run(input_bam, logger, window_size=100000, num_grid=0, model_num=0, regions=None, genome=None):
    """

    :param dir_name:
    :param logger:
    :param input_bat:
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

    if genome=='hg38':
        chr_table = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11'
            ,'chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21'
           ,'chr22','chrX','chrY']

    elif genome=='hg19':
        chr_table = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chrX','chr8','chr9','chr10',
                'chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr20','chrY',
                'chr19','chr22','chr21']

    elif genome=='hg18':
        #chr_table = ['chr1','chr1_random','chr10','chr10_random','chr11','chr11_random','chr12','chr13','chr13_random','chr14','chr15','chr15_random','chr16','chr16_random','chr17','chr17_random','chr18','chr18_random','chr19','chr19_random','chr20','chr21','chr21_random','chr22','chr22_random','chr22_h2_hap1','chr_random','chr','chr_random','chr','chr_random','chr','chr_random','chr','chr_random','chr','chr_random','chr','chr_random','chr','chr_random','chr','chr_random','chr','chr_random']
        logger.info("hg18 reference genome is not valid yet.")

    else:
        chr_table = []
        logger.info("Reference genome must be selected among {hg19, hg18, hg38}")

    logger.info("{} reference genome was selected.".format(genome))
    #logger.info("{}`s table :".format(chr_table))
    #logger.info("{}`s lengths :".format(chr_lengths))

#    if not (regions == None):
#        chr_index = []

    if regions is not None:
        logger.info("Specific calling regions was defined :: {}".format(regions))
        regions = regions.split(':')
        chromosome = regions[0]
        regions = regions[1].split('-')
        call_start = regions[0]
        call_end = regions[1]

        for i in range(len(chr_table)):
            if chr_table[i] == chromosome:
                chr_no = i
                break

        if call_start == 's':
            call_start = 1
        else:
            call_start = int(call_start)

        if call_end == 'e':
            call_end = chr_lengths[chr_no]
        else:
            call_end = int(call_end)

        logger.info("Chromosome<{}> , <{}> to <{}>".format(chromosome, call_start, call_end))

        ref_data_df = pd.read_table("geneRef/{}.bed".format(chromosome), names=['chr','start','end'] , header=None, usecols=[0,1,2])
        logger.info("Peak calling in chromosome {}:".format(chromosome))
        call_peak(chr_no, chr_table, chr_lengths, input_bam, ref_data_df, input_data, input_data_ref,
                logger, num_grid, prediction, sess, window_size, pgb_on=False, window_start=call_start, window_end=call_end)
    else:
        for chr_no in range(len(chr_table)):
            ref_data_df = pd.read_table("geneRef/{}.bed".format(chr_table[chr_no]), names=['start','end'] , header=None, usecols=[1,2])
            logger.info("Peak calling in chromosome {}:".format(chr_table[chr_no]))
            call_peak(chr_no, chr_table, chr_lengths, input_bam, ref_data_df, input_data, input_data_ref,
                    logger, num_grid, prediction, sess, window_size, pgb_on=True)




def call_peak(chr_no, chr_table, chr_lengths, file_name, ref_data_df, input_data, input_data_ref,
        logger, num_grid, prediction, sess, window_size, pgb_on=False, window_start=1, window_end=None, window_chunk=500):
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
    window_count = window_start
    if window_end == None:
        window_end = chr_lengths[chr_no]

    stride = window_size / num_grid
    eval_counter = 0
    output_file_name = "{}.bed".format(file_name.rsplit('.')[0])
    peaks = []

    logger.info("Length of [{}] is : {}".format(chr_table[chr_no], window_end))

    ref_data = ref_data_df.values

    while True:
        if (eval_counter % window_chunk) == 0:
            if pgb_on:
                bar = pgb.ProgressBar(max_value=window_chunk)
            logger.info("Generate Read Counts . . .")
            if window_count + window_size*window_chunk < window_end:
                read_count_chunk = generateReadcounts(window_count,window_count+window_size*window_chunk, chr_table[chr_no], file_name, num_grid, window_chunk)
                end = window_count+window_size*window_chunk
            else:
                window_n = int((window_end - window_count)/ window_size)
                read_count_chunk = generateReadcounts(window_count,
                        window_count+window_size*window_n, chr_table[chr_no], file_name, num_grid, window_n)
                end = window_end
            logger.info("Calling . . . :[{}:{}-{}]".
                    format(chr_table[chr_no],window_count,end))

        if window_count + window_size > window_end:
            logger.info("Peak calling for [{}] is done.".format(chr_table[chr_no]))
            writeBed(output_file_name, peaks, logger, printout=False)
            break

        read_count_by_grid = read_count_chunk[eval_counter*num_grid:(eval_counter+1)*num_grid].reshape(input_data_eval.shape)
        ref_data_by_grid = generateRefcounts(window_count, window_count+window_size,
                ref_data, num_grid).reshape(input_ref_data_eval.shape)

        result_dict = {input_data_eval: read_count_by_grid, input_ref_data_eval: ref_data_by_grid, is_train_step: False}
        preds = sess.run(prediction, feed_dict=result_dict)
        class_value_prediction = buildModel.classValueFilter(preds)

        peaks += predictionToBedString(class_value_prediction, chr_table[chr_no], window_count, stride,
                num_grid, logger, read_count_by_grid.reshape(num_grid).tolist())
        if pgb_on:
            bar.update(eval_counter)

        eval_counter += 1

        if eval_counter == window_chunk:
            writeBed(output_file_name, peaks, logger, printout=False)
            eval_counter = 0
            peaks =[]

        window_count += window_size


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
