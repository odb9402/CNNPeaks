import glob
import ntpath


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


def extractChrClass(dir):
    """
    Extract a chromosome number and a class number from label file names.

    :param dir:
    :return:
    """

    chr_list = set()
    #a = glob.glob(dir + "*")
    #print(a)
    for ct_file in glob.glob(dir + "/*.ct"):
        #chr_list.add(ct_file.rsplit('/', 1)[1].split('_')[0])
        chr_list.add(path_leaf(ct_file).split('_')[0])

    data_direction = {}
    for chr in chr_list:
        cls_list = []
        for ct_file in glob.glob(dir + "/" + chr + "_*.ct"):
            #cls_list.append(ct_file.rsplit('/', 1)[1].split('_')[1])
            #b = path_leaf(ct_file).split('_')[1]
            cls_list.append(path_leaf(ct_file).split('_')[1])
        data_direction[chr] = cls_list

    return data_direction


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(path)