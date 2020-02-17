import subprocess as sp
import pandas as pd
import os
import argparse
import sys
import progressbar as pgb

def change_interval(bed_name, bam_name, output_name, interval_size=200):
    bed_file = open(bed_name, 'r')
    bed_lines = bed_file.readlines()
    
    new_bed_file = open(output_name, 'w')
    peak_num = len(bed_lines)
    bar = pgb.ProgressBar(max_value=peak_num)
    
    print("Refine the interval size of {}. . .".format(bed_name))
    print("Using alignment {} . . .".format(bam_name))
    
    i = 0
    new_bed_str = ""
    for b in bed_lines:
        b = b.rstrip('\n').split('\t')
        region_str = "{}:{}-{}".format(b[0],b[1],b[2])
        samtool_proc = sp.Popen(["samtools depth -r {} {}".format(region_str, bam_name)]
                                ,stdout=sp.PIPE
                                ,stderr=sp.DEVNULL
                                ,shell=True)
        out, _ = samtool_proc.communicate()
        
        strings = [peak.split('\t') for peak in out.decode("utf-8").split('\n')]
        strings.pop(len(strings)-1)
        
        max_depth = 0
        idx = 0
        max_idx = -1
        for s in strings:
            if int(s[2]) > max_depth:
                max_depth = int(s[2])
                max_idx = idx
#                print(max_idx, max_depth)
            idx += 1
        
        max_pos = int(strings[max_idx][1])
        
        new_bed = "{}\t{}\t{}\t{}\n".format(b[0],
                                        int(max_pos-interval_size/2),
                                        int(max_pos+interval_size/2),
                                        '\t'.join(b[3:]))
        new_bed_str += new_bed 
        i += 1
        bar.update(i)
        if i % 1000 == 0 and i != 0:
            new_bed_file.write(new_bed_str)
            new_bed_str = ""
    new_bed_file.write(new_bed_str)
    new_bed_file.close()
    

#################### Setting arguments ########################
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i","--inputBed", help="Input bed file to change interval size.")
arg_parser.add_argument("-b","--inputBam", help="Input bam file to change interval size.")
arg_parser.add_argument("-o","--outputBed", help="Output bed file name.")
arg_parser.add_argument("-s","--size", default=200, help="The interval size of the new bed file.")
args = arg_parser.parse_args()

args.size = int(args.size)

if args.inputBed == None:
    print("Input bed file must be provided.")
    args.print_help()
    exit()
elif args.inputBam == None:
    print("Input bam file must be provided.")
    args.print_help()
    exit()
elif args.outputBed == None:
    print("Output bed file must be provided.")
    args.print_help()
    exit()

change_interval(args.inputBed, args.inputBam, args.outputBed, args.size)