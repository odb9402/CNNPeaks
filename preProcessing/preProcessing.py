import pandas as pd
import pysam
import glob
import os
import re
import subprocess
import random
import time

from multiprocessing import cpu_count, Process, Manager

from sklearn.cluster import DBSCAN

def run(dir_name, logger, bp_eps=30000, searching_dist=80000, num_grid=2000):
    """
    This preprocessing step will create alignment read count data from
    input directory dir_name was specified by a user. The results will
    be saved in directories which have same name with input bam files.
    e. g) /H3K4me3_1_K562.bam `s results is saved in /H3K4me3_1_K562/chrn_n_gridxxxx.csv

    :param dir_name:
    :param logger:
    :param bp_eps:
    :param searching_dist:
    :return:
    """

    PATH = os.path.abspath(dir_name)
    bam_files = glob.glob(PATH + '/*.bam')
    label_files = glob.glob(PATH + '/*.txt')

    MAX_CORE = cpu_count()
    processes = []

    logger.info("Creating small fragment files for training in input dir.")

    for bam_file in bam_files:
        for label_file in label_files:
            if is_same_target(bam_file, label_file) == True:
                logger.info("Find a matched pair between labeled data and an alignment file.")
                logger.info("Shared target :: " + bam_file.rsplit('/',1)[1].split('_')[0])
                label_data = loadLabel(label_file)
                cellType_string = bam_file[:-4].rsplit('_',1)[1]
                #label_data = filtering_label_with_cellType(label_data, cellType_string)

                logger.info("DBSACN clustering raw label data with base point eps:[ " + str(bp_eps) + " ]\n")
                clusteringLabels(label_data, bp_eps)

                logger.info("Making fragments for training with <searching distance, grid> : [ " \
                            + str(searching_dist) + ", "+ str(num_grid)+" ]\n")
                process = Process(target=makeTrainFrags, \
                                  args=(bam_file, label_data, searching_dist, num_grid, cellType_string ,logger,))
                parallel_execution(MAX_CORE - 1, process, processes)

    for proc in processes:
        proc.join()


def makeTrainFrags(bam_file, label_data_df, searching_dist, num_grid, cell_type, logger):
    """
    For a bam alignment file, it slices into small fragments
    with label regions. Length of sliced bam files is a sum of
    labeled regions and "searching_dist". If a group of labeled
    data start at chr1:300,000 and end with chr1:350,000, the
    length of a small fragment from bam_file is 50000 + searching dist.
    If (regions of the label cluster / 5) is lower than searchine dist,
    the bias is not a searching dist but (regions of the label cluster / 5).

    :param bam_files: bam_files MUST be a absPath.
    :param label_data_df:
    :param searching_dist:
    :param num_grid:
    :param logger:
    :return:
    """
    num_grid_label = num_grid // 5

    chr_list = set(label_data_df['chr'].tolist())
    if not os.path.isdir(bam_file[:-4]):
        os.makedirs(bam_file[:-4])

    if not os.path.isfile(bam_file + '.bai'):
        createBamIndex(bam_file)
        logger.info("Creating index file of [" + bam_file + "]")
    else:
        logger.info("[" + bam_file + "] already has index file.")

    bam_alignment = pysam.AlignmentFile(bam_file +'.sort', 'rb', index_filename=bam_file +'.sort.bai')

    for chr in chr_list:
        label_data_by_chr = label_data_df[label_data_df['chr'] == chr]
        class_list = set(label_data_by_chr['class'].tolist())

        for cls in class_list:
            label_data_by_class = label_data_by_chr[label_data_by_chr['class'] == cls]
            region_start = int(label_data_by_class.head(1)['start'])
            region_end = int(label_data_by_class.tail(1)['end'])
            region_size = region_end - region_start

            if region_size / 5 > searching_dist:
                left_dist = random.randint(0, int(region_size/5))
                right_dist = int(region_size/5) - left_dist
            else:
                left_dist = random.randint(0, searching_dist)  # Additional window is non-deterministic.
                right_dist = searching_dist - left_dist

            region_start -= left_dist
            region_end += right_dist
            region_size = region_end - region_start

            stride = region_size / num_grid             # that can be elimenated.
            stride_label = region_size / num_grid_label

            logger.debug("STRIDE :" + str(stride) + "           REGION SIZE :" + str(region_size))
            read_count_by_grid = pd.DataFrame(columns=['readCount'], dtype=int)

            for step in range(num_grid):
                count = bam_alignment.count(region=createRegionStr(chr, int(region_start + stride*step)))
                read_count_by_grid = read_count_by_grid.append({'readCount' : count}, ignore_index=True)

            output_count_file = bam_file[:-4] + "/" + str(chr) + "_" + str(cls) + "_grid" + str(num_grid)+".ct"
            output_label_file = bam_file[:-4] + "/label_" + str(chr) + "_" + str(cls) + "_grid" + str(num_grid)+".lb"

            output_label_df_bef = pd.DataFrame(columns=['startGrid','endGrid'])
            output_label_df_bef['startGrid'] = (label_data_by_class['start'] - region_start) / stride_label
            output_label_df_bef['endGrid'] = (label_data_by_class['end'] - region_start) / stride_label

            output_label_df = pd.DataFrame(columns=['peak','noPeak'], dtype=int, index=range(num_grid_label))
            output_label_df['peak'] = 0
            output_label_df['noPeak'] = 1

            index_count = 0
            for index, row in output_label_df_bef.iterrows():
                if cell_type in str(label_data_by_class['cellType'].iloc[index_count]):
                    output_label_df.loc[int(row['startGrid']):int(row['endGrid']), 'peak'] = 1
                    output_label_df.loc[int(row['startGrid']):int(row['endGrid']), 'noPeak'] = 0
                index_count += 1
            read_count_by_grid.to_csv(output_count_file)
            output_label_df.to_csv(output_label_file)

            logger.info("["+output_count_file+"] is created.")
            logger.info("["+output_label_file+"] is created.\n")


def clusteringLabels(label_data_df, bp_eps):
    """
    Clustering pandas label data with DBSCAN.
    It has hyperparameter that define maximum distance between
    cluster elements in the same group.

    :param label_data_df:
    :param bp_eps: size of base points

    :return: None. it will change the original data frame "label_data".
    """

    label_data_df['class'] = None
    chr_list = set(label_data_df['chr'].tolist())

    for chr in chr_list:
        chr_df = label_data_df.loc[label_data_df.chr == chr].copy()
        feature_df = chr_df[['start']]

        DBSCAN_model = DBSCAN(eps = bp_eps, min_samples=1)  ## It does not allow noisy element with minsample = 1
        predict = pd.DataFrame(DBSCAN_model.fit_predict(feature_df), columns=['class'])

        label_data_df.loc[label_data_df.chr == chr, 'class'] = predict['class'].tolist()


def loadLabel(label_file_name):
    """
    Given python list of labeled files, it return pandas dataframe
    including data from those files.

    :param label_files: Python list including set of label file names.
    :return: label_data_frame: Pandas dataframe with labeled data
    """

    label_col = ['chr', 'start', 'end', 'peakStat', 'cellType']
    label_data_frame = pd.DataFrame(columns=label_col)

    label_file = open(label_file_name, 'r')
    raw_label_data = label_file.readlines()

    label_data = []

    for peak in raw_label_data:
        if peak == "\r\n":
            raw_label_data.remove(peak)

    for peak in raw_label_data:
        if '#' not in peak and 'chr' in peak:
            peak = peak.rstrip('\r\n')
            label_data.append(peak)

    label_data.sort()

    for i in range(len(label_data)):
        label_data[i] = re.split(':|-| ', label_data[i], 4)

        for index in range(len(label_data[i])):
            if label_data[i][index] == '':
                label_data[i].pop(index)
        if len(label_data[i]) == 4:
            label_data[i].append('None')

        label_data[i][1] = int(label_data[i][1].replace(',',''))
        label_data[i][2] = int(label_data[i][2].replace(',',''))

    additional_data = pd.DataFrame(columns=label_col, data=label_data)
    label_data_frame = label_data_frame.append(additional_data, ignore_index=True)
    label_file.close()

    label_data_frame['chr'] = label_data_frame.chr.astype('category')
    label_data_frame['start'] = label_data_frame.start.astype(int)
    label_data_frame['end'] = label_data_frame.end.astype(int)
    label_data_frame['peakStat'] = label_data_frame.peakStat.astype('category')

    return label_data_frame


def createBamIndex(input_bam):
    """
    Python wrapper to use "bamtools" for creating index file of
    "input_bam".

    :param input_bam: A input bam file name.
    """
    #if not os.path.isfile(input_bam+".sort"):
    #    subprocess.call(['sudo bamtools sort -in ' + input_bam + ' -out '+ input_bam+'.sort'], shell=True)
    #else:
    #    return 0
    subprocess.call(['sudo bamtools index -in ' + input_bam], shell=True)



def createRegionStr(chr, start, end=None):
    """
    Creating "samtools"-style region string such as
    "chrN:zzz,zzz,zzz-yyy,yyy,yyy". If end is not specified,
    it will create "chrN:xxx,xxx,xxx-xxx,xxx,xxx".


    :param chr:
    :param start:
    :param end:

    :return: A string of samtool-style region
    """
    if end == None:
        return str(chr) + ":" + str(int(start)) + "-" + str(int(start))
    elif end is not None:
        return str(chr) + ":" + str(int(start)) + "-" + str(int(end))


def filtering_label_with_cellType(label_data_df, cellType):
    """

    :param label_data_df:
    :param cellType: A string of cell type
    :return:
    """
    label_data_df = label_data_df[label_data_df.cellType.str.contains(cellType)]
    return label_data_df


def is_same_target(bam_file_name, label_data_df):
    """

    :param bam_file_name:
    :param label_data_df:
    :return:
    """
    return bam_file_name.rsplit('/',1)[1].split('_')[0] == label_data_df.rsplit('/',1)[1].split('_')[0]


def parallel_execution(MAX_CORE, learning_process, learning_processes):
    """

    :param MAX_CORE:
    :param learning_process:
    :param learning_processes:
    :return:
    """
    if len(learning_processes) < MAX_CORE:
        learning_processes.append(learning_process)
        learning_process.start()
    else:
        keep_wait = True
        while True:
            time.sleep(0.1)
            if not (keep_wait is True):
                break
            else:
                for process in reversed(learning_processes):
                    if process.is_alive() is False:
                        learning_processes.remove(process)
                        learning_processes.append(learning_process)
                        learning_process.start()
                        keep_wait = False
                        break
