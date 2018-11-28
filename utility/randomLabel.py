import pysam
import random
import os

def run(fileName, labelNumForChr = 20):
    """
    Generate random label point from
    """
    fileName = os.path.abspath(fileName)
    bam_alignment = pysam.AlignmentFile(fileName, 'rb', index_filename=fileName+'.bai')
    chr_length = bam_alignment.lengths
    chr_interval = []
    search_interval = 30000
    for i in range(len(chr_length)):
        chr_interval.append({'start':100001, 'end':chr_length[i]-search_interval})

    output = open(fileName.rsplit('.')[0] + '.txt' , "w")
    for i in range(22):
        label_count = labelNumForChr
        while label_count != 0 :
            start = random.randint(chr_interval[i]['start'], chr_interval[i]['end'])
            output.write("chr{}:{:,}-{:,} \t peaks\n".format(i+1, start, start + random.randint(70000,100000)))
            label_count -= 1
