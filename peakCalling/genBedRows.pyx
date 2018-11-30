import math
import random
import numpy as np
cimport numpy as np

def predictionToBedString(np.ndarray prediction, str chromosome, int region_start, double stride,
        int num_grid, np.ndarray reads, int min_peak_size, int max_peak_num):
    """
    Python list "prediction" which has binary values will will be changed
    as bed-file string. There are two conditions to accept as peak for each
    prediction.

    1. Peak size must be higher than min_peak_size.
    2. The number of peak in the single window cannot be higher than max_peak_num.

    If "predicition" cannot satisfy these conditions, the function return empty list.

    :param prediction:
    :return:
    """
    cdef int peak_size = 0
    cdef list peaks = []
    cdef double end_point
    cdef double start_point
    
    cdef double z_score_mean
    cdef double mean
    cdef double std

    cdef double min_depth
    cdef double avg_depth
    cdef double max_depth

    cdef str random_name

    bed_template = "{}\t{}\t{}\t{}\t{:.2f}\t{}\t{:.2f}\t{}\n"

    for step in range(num_grid):
        if prediction[step] == 1:
            peak_size += 1
        else:
            if not (peak_size == 0):
                # Condition 1: peak size should be higher than min_peak_size.
                if peak_size > min_peak_size:
                    end_point = region_start + ( stride * step )
                    start_point = end_point - ( peak_size * stride )

                    std = math.sqrt(np.var(reads)+0.000001)
                    mean = np.mean(reads)
                    z_score_mean = float(np.mean((reads[step - peak_size : step] - mean)/std))
                    min_depth = np.amin(reads[step - peak_size : step])
                    avg_depth = np.mean(reads[step - peak_size : step])
                    max_depth = np.amax(reads[step - peak_size : step])

                    random_name = "{}_{:5}".format(chromosome,random.randint(0,100000))

                    peaks.append(bed_template.format(chromosome, int(start_point), int(end_point), random_name, z_score_mean, min_depth, avg_depth, max_depth))

                peak_size = 0


    # Condition 2 : The number of peaks must be lower than max_peak_num.
    if len(peaks) > max_peak_num:
        return []
    else:
        return peaks

