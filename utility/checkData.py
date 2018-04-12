import os
import glob
import buildModel.buildModel as buildModel
import pandas as pd
import matplotlib.pyplot as plt

def run(dir_name, logger, num_grid=8000):
    PATH = os.path.abspath(dir_name)
    bam_files = glob.glob(PATH + '/*.bam')
    label_files = glob.glob(PATH + '/*.txt')

    dir_list = []
    for bam_file in bam_files:
        dir_list.append(bam_file[:-4])

    for dir in dir_list:
        logger.info("DIRECTORY (TARGET) : <" + dir + ">")

    input_list = {}
    for dir in dir_list:
        input_list[dir] = buildModel.extractChrClass(dir)

    train_data_list = []
    train_label_list = []
    for dir in input_list:
        for chr in input_list[dir]:
            for cls in input_list[dir][chr]:
                input_file_name = (dir + "/" + chr + "_" + cls + "_grid" + str(num_grid) + ".ct")
                label_file_name = (dir + "/label_" + chr + "_" + cls + "_grid" + str(num_grid) + ".lb")
                reads = (pd.read_csv(input_file_name))['readCount'].as_matrix().reshape(num_grid)
                label = (pd.read_csv(label_file_name))['peak'].as_matrix().transpose()
                label = buildModel.expandingPrediction(label)

                plt.plot(reads,'k')
                plt.plot(label,'r.')
                plt.show()

                if input("save(s) or delete(d)  ::") == 'd':
                    os.remove(input_file_name)
                    os.remove(label_file_name)