import math
import random
import numpy as np
import scipy.stats as st
import sys
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
    cdef double eps = sys.float_info.min
    
    cdef double end_point
    cdef double start_point
    
    cdef double threshold_pvalue = 1
    cdef double p_value
    cdef double p_value_mean
    cdef double p_value_var
    cdef double var
    cdef double mean
    cdef double score
    cdef double min_depth
    cdef double avg_depth
    cdef double max_depth

    cdef str random_name

    #bed_template = "{}\t{}\t{}\t{}\t{:.6f}\t{}\t{:.2f}\t{}\n"
    bed_template = "{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
    
    for step in range(num_grid):
        if prediction[step] > 0.5 :
            peak_size += 1
        else:
            if not (peak_size == 0):
                # Condition 1: peak size should be higher than min_peak_size.
                if peak_size > min_peak_size:
                    end_point = region_start + ( stride * step )
                    start_point = end_point - ( peak_size * stride )
                    
                    mean = np.mean(reads)
                    
                    min_depth = np.amin(reads[step - peak_size : step])
                    avg_depth = np.mean(reads[step - peak_size : step])
                    max_depth = np.amax(reads[step - peak_size : step])
                    
                    avg_sig = np.mean(prediction[step - peak_size : step])
                    
                    p_value_mean = st.poisson.cdf(mean, avg_depth)
                    p_value_max = st.poisson.cdf(mean, max_depth)
                    
                    score_narrow = -math.log10(p_value_max+eps)*avg_sig*100
                    score_broad = -math.log10(p_value_mean+eps)*avg_sig*100
                    
                    random_name = "{}_{}".format(chromosome,random.randint(0,100000))
                    if p_value < threshold_pvalue:
                        peaks.append(bed_template.format(chromosome, int(start_point), int(end_point), random_name, score_narrow, score_broad, avg_sig, p_value_mean, p_value_max))
                    else:
                        pass

                peak_size = 0

    # Condition 2 : The number of peaks must be lower than max_peak_num.
    if len(peaks) > max_peak_num:
        return []
    else:
        return peaks

