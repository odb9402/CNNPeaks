"""
All annotations of these sources are written with poor english from writer.
Please understand if it will be so crappy. . . .
"""

import argparse
import sys
import logging

from buildModel.buildModel import run as buildModel
from preProcessing.preProcessing import run as preProcessing

def main():

    #################### Setting arguments ########################
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-mode","--runMode", choices=['preprocess','buildModel'] ,help="Select a mode.")
    arg_parser.add_argument("-i","--inputDir", help="Input directory including labeled data and bam alignment files.")
    arg_parser.add_argument("-m","--savedModel", help="A saved CNN model file.")
    arg_parser.add_argument("-g","--gridSize",default=4000, help="Define numbers of grid for each training data")
    arg_parser.add_argument("-s","--searchingDist", help="")
    arg_parser.add_argument("-eps","--basePointEPS",help="")

    args = arg_parser.parse_args()

    if args.inputDir == None:
        logger.error("'-i' : Input Directory was missed.")
        exit()
    ###############################################################

    if args.runMode == 'preprocess':
        preProcessing(args.inputDir, logger, num_grid=int(args.gridSize))
    elif args.runMode =='buildModel':
        buildModel(args.inputDir, logger, num_grid=int(args.gridSize))


if __name__ == '__main__':
    logger = logging.getLogger("ConvLog")
    logger.setLevel(logging.DEBUG)               # The logger object only output logs which have
                                                # upper level than INFO.
    log_format = logging.Formatter('%(asctime)s:%(message)s')

    stream_handler = logging.StreamHandler()    # Log output setting for the command line.
    stream_handler.setFormatter(log_format)     # The format of stream log will follow this format.
    logger.addHandler(stream_handler)

    #file_handler = logging.FileHandler()        # Log output setting for the file.
    #logger.addHandler(file_handler)

    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("i hope you complete. \n")
        sys.exit()