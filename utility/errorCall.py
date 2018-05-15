from utility.calculateError import run as calculateError
from utility.loadLabel import run as loadLabel
from utility.loadPeak import run as loadPeak

def run(input_bed_file, input_label_file, logger):
    peaks = loadPeak(input_bed_file)
    error = 0
    total = 0
    N = 0
    P = 0
    FN = 0
    FP = 0
    for i in range(22):
        chr_labels = loadLabel(input_label_file, input_chromosome="chr{}".format(i + 1))
        chr_peaks = list(filter(lambda peak: peak['chr'] == 'chr{}'.format(i + 1), peaks))
        logger.info("chr{}".format(i + 1))
        temp_x, temp_y, FNFP = calculateError(chr_peaks, chr_labels)
        if temp_x == 0 and temp_y == 0:
            continue
        error += temp_x
        total += temp_y
        N += FNFP['negativeNum']
        P += FNFP['positiveNum']
        FN += FNFP['FN']
        FP += FNFP['FP']
    logger.info("\n# of Negatives: {} , # of Positives: {}".format(N, P))
    logger.info("\nACC: {} , FN_Rate: {} , FP_Rate: {}".format(1. - error / total, FN / N, FP / P))
