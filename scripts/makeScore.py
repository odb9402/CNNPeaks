import csv
import subprocess as sp
import argparse
import pandas as pd
import math
import subprocess as sp

input = ""

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i","--input")
arg_parser.add_argument("-o","--output")

args = arg_parser.parse_args()

col = ['chr','start','end','name','sigVal','pVal','maxpVal','maxRead']

bed = pd.read_csv(args.input, sep='\t', names=col)

score = []
score_max = []
for i in range(len(bed)):
    score.append(100 * -math.log10(bed['pVal'][i]) * bed['sigVal'][i])
    score_max.append(100 * -math.log10(bed['maxpVal'][i]) * bed['sigVal'][i])

bed = bed.assign(score=score)
bed = bed.assign(score_max=score_max)

print(bed)

bed[['chr','start','end','sigVal','pVal','maxpVal','score','score_max']].to_csv(args.output,sep='\t',header=False, index=False)

