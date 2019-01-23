from utility.calculateError import run as calculateError
from utility.loadLabel import run as loadLabel
from utility.loadPeak import run as loadPeak
import argparse

def run(input_label_file=None, cell_type=None):
    cell_type = input_label_file.rsplit('_',1)[1].split('.')[0]
    labels = []
    for i in range(22):
        new_label = loadLabel(input_label_file, input_chromosome="chr{}".format(i + 1), input_cellType = cell_type)
        if new_label == -1:
            continue
        else:
            labels += new_label
        for label in new_label:
            print("chr{}\t{}\t{}\t{}".format(i+1, label['regions'][0], label['regions'][1], label['peakStat']))

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input')

args = parser.parse_args()

run(input_label_file=args.input)
