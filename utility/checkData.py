import os
import glob
import buildModel.buildModel as buildModel
import pandas as pd
import matplotlib.pyplot as plt

def run(dir_name, logger, num_grid=8000):
    PATH = os.path.abspath(dir_name)
    bam_files = glob.glob(PATH + '/*.bam')
    label_files = glob.glob(PATH + '/*.txt')

    dir_list = os.listdir(PATH)

    for dir in dir_list:
        dir = PATH + '/' + dir
        logger.info("DIRECTORY (TARGET) : <" + dir +">")

    input_list = {}
    for dir in dir_list:
        dir = PATH + '/' + dir
        input_list[dir] = buildModel.extractChrClass(dir)

    train_data_list = []
    train_label_list = []
    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = "{}/{}_{}_grid{}.ct".format(dir, chr, cls, num_grid)
                ref_file_name = "{}/ref_{}_{}_grid{}.ref".format(dir, chr, cls, num_grid)
                label_file_name = "{}/label_{}_{}_grid{}.lb".format(dir, chr, cls, num_grid)

                reads = (pd.read_csv(input_file_name))['readCount'].as_matrix().reshape(num_grid)
                label = (pd.read_csv(label_file_name))['peak'].as_matrix().transpose()
                label = buildModel.expandingPrediction(label)

                plt.plot(reads,'k')
                plt.plot(label,'r.')
                plt.title('{} {}-{}'.format(dir,chr,cls))
                plt.show()

                if input("save(1) or delete(0)  ::") == '0':
                    os.remove(input_file_name)
                    os.remove(label_file_name)
                    os.remove(ref_file_name)
