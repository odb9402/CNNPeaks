import os


def calculate_error(peak_data, labeled_data):
    """
    calculate actual error by numbering to wrong label
    :param peak_data:
        python map is parsed and it is from result of peak calling algorithm
        like a MACS.
    :param labeled_data:
        python map is parsed and it is from labeled data file.
    :return:
        return python tuple ( number of incorrect label , number of whole label )
    """

    # sum of error label
    scores = 0.0

    # number of Error label about each error type.
    FP_error = 0.0
    FN_error = 0.0

    # number of label which can occur error about each error type.
    possible_FP = 0
    possible_FN = 0

    for label in labeled_data:
        if label['peakStat'] == 'peaks':
            possible_FN += 1
            state = is_peak(peak_data, label['regions'], weak_predict=True)

            if state == "False Negative":
                FN_error += 1
            else:
                scores += state

        elif (label['peakStat'] == 'peakStart') or (label['peakStat'] == 'peakEnd'):
            possible_FP += 1
            possible_FN += 1
            state = is_peak(peak_data, label['regions'])

            if state == "False Positive":
                FP_error += 1
            elif state == "False Negative":
                FN_error += 1
            else:
                scores += state

        elif label['peakStat'] == 'noPeak':
            possible_FP += 1
            state = is_noPeak(peak_data, label['regions'])
            # print state
            if not (state == True):
                FP_error += 1
                scores += state
            else:
                scores += 1

        else:
            print("label type error")
            exit()

    print("possible FN " + str(possible_FN), "possible FP " + str(possible_FP))
    print("FN_Error: " + str(FN_error), "FP_Error: " + str(FP_error))

    return len(labeled_data) - scores, len(labeled_data)


def is_peak(target, value, tolerance=500, weak_predict=False):
    """
    Checking the label is peak or not.
    :param target:
    :param value:
    :param tolerance:
    :param weak_predict:
    :return:
    """
    """this function will find to regions in target bed set by using binary search"""
    """the similarity allow the distance of bed file row between label area as long as own value"""

    index = len(target) / 2
    min_index = 0
    max_index = len(target)

    # if find correct one, return True
    while True:
        correct_ness = is_same(target, value, index, tolerance)
        # print index,min_index ,  max_index, len(target)
        if correct_ness is 'less':
            max_index = index
            index = (min_index + index) / 2
        elif correct_ness is 'upper':
            min_index = index
            index = (max_index + index) / 2
        # find correct regions
        else:
            if (weak_predict == True):
                return calculate_sum_of_weights(index, target, tolerance, value, mode='bonus')

            # find one peak
            else:
                if (index + 1) is not len(target) \
                        and is_same(target, value, index + 1, tolerance) is 'in' \
                        or is_same(target, value, index - 1, tolerance) is 'in':
                    return "False Positive"
                else:
                    return 1 + bonus_weight(value, target[index], 'peakStart')

        if max_index <= min_index + 1:
            if is_same(target, value, index, tolerance) is 'in':
                return 1 + bonus_weight(value, target[index], 'peakStart')
            else:
                return "False Negative"


def is_noPeak(target, value, tolerance=0):
    """
    :param target:
    :param value:
    :param tolerance:
    :return:
    """
    region_min = value[0]
    region_max = value[1]

    index = len(target) / 2
    min_index = 0
    max_index = len(target)
    steps = 1

    while True:
        find_matched = is_same(target, value, index, 0)

        if find_matched is 'less':
            max_index = index
            index = (min_index + index) / 2
        elif find_matched is 'upper':
            min_index = index
            index = (max_index + index) / 2
        # find correct regions , so it is fail
        else:
            return calculate_sum_of_weights(index, target, tolerance, value, mode='bias')

        if abs(float(target[index]['region_e']) - region_min) < 5 * steps or steps > 1000:
            break
        steps += 1

    # correct label ( no peak )
    if not (index + 1 >= len(target)):
        if float(target[index + 1]['region_s']) + tolerance > region_max \
                and float(target[index]['region_e']) + tolerance < region_min:
            return True
        else:
            return True

    # false negative no peak ( there is peak )
    else:
        return calculate_sum_of_weights(index, target, tolerance, value, mode='bias')


def calculate_sum_of_weights(index, target, tolerance, value, mode=None):
    peaks = []
    num_of_peaks = 1
    front_check = 1
    back_check = 1
    peaks.append(target[index])
    ## front seek
    while True:
        if index + front_check < len(target):
            if (is_same(target, value, index + front_check, tolerance) is 'in'):
                num_of_peaks += 1
                peaks.append(target[index + front_check])
            else:
                break
            front_check += 1
        else:
            break

    ## back seek
    while True:
        if index - back_check is not (-1):
            if (is_same(target, value, index - back_check, tolerance) is 'in'):
                num_of_peaks += 1
                peaks.append(target[index - back_check])
            else:
                break
            back_check += 1
        else:
            break

    if mode is 'bonus':
        return 1 + bonus_weight(value, peaks, 'peaks')
    elif mode is 'bias':
        return bonus_weight(value, peaks, 'nopeak')


def is_same(target, value, index, tolerance):
    """
    this function check label value whether bigger than index or lower than index
    :param target:
    :param value:
    :param index:
    :param tolerance:
    :return:
    """

    if value[1] + tolerance <= float(target[index]['region_s']):
        return 'less'
    elif value[0] - tolerance >= float(target[index]['region_e']):
        return 'upper'
    else:
        return 'in'


def bonus_weight(label, target, case):
    """
    label is raw of label data set.
    Target is dict or list of dict.

    :param label:
    :param target:
    :param case:
    :return:
    """
    length_label = (label[1] - label[0]) / 2
    center_label = label[1] + length_label

    if case == "peakStart":

        target['region_e'] = float(target['region_e'])
        target['region_s'] = float(target['region_s'])

        length_target = (target['region_e'] - target['region_s']) / 2
        center_target = target['region_s'] + length_target

        distance_c = abs(center_label - center_target)
        distance_l = abs(length_label - length_target)

        weight = (1.0 / (1 + distance_c / length_label)) * (1.0 / (1 + distance_l / length_target))

        return weight

    elif (case == "peaks") or (case == "nopeak"):

        weight_sum = 0

        for peak in target:
            length_peak = (float(peak['region_e']) - float(peak['region_s'])) / 2
            center_peak = float(peak['region_s']) + length_peak

            distance_c = abs(center_label - center_peak)
            distance_l = abs(length_label - length_peak)

            weight = (1.0 / (1 + distance_c / length_label)) * (1.0 / (1 + distance_l / length_peak))

            weight_sum += weight

        n = len(target)

        weight_mean = float(weight_sum) / float(n)
        penalty = 2 * (1.0 - ((n ** 0.5) / (1.0 + n ** 0.5)))

        if case == "peaks":
            return weight_mean * penalty
        elif case == "nopeak":
            return (-1) * (weight_mean * penalty)
        else:
            print("caseError")
            exit(0)


def run(input_peaks, input_labels):
    """
    This is the module for calculation Error by comparing between
    labeled data and the input file.
    :param input_file_name:
        This parameter is file name that result of peak detectors.
    :param input_labels:
        It is python map and it is already parsed which means
        having specific cell type and chromosome.
    :return:
        Accuracy of result file.
    """

    # case of input label size is 0, error num error rate is zero.
    if input_labels is -1:
        return 0, 0

    if input_peaks is -1:
        return 0, 0

    if len(input_peaks) is 0:
        return 0, 0

    if len(input_labels) is 0:
        return 0, 0

    error_num, total_label = calculate_error(input_peaks, input_labels)

    return error_num, total_label
