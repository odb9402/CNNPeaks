import numpy as np
cimport numpy as np
import subprocess as sp
from cpython cimport array
from libc.stdlib cimport malloc, free
from cython.parallel import prange

def generateReadcounts(int region_start, int region_end, str chr_num, str file_name, int num_grid, int window_num):
    cdef float[:] read_count = np.empty(num_grid*window_num, dtype=np.float32)
    
    cdef double stride = (region_end - (region_start + 1))/float(num_grid * window_num)
    cdef list samtools_command

    if region_end == -1:
        samtools_command = ['samtools depth -aa -r {} {}'.format(chr_num, file_name)]
    else:
        samtools_command = ['samtools depth -aa -r {} {}'.format("{}:{}-{}".
            format(chr_num,region_start,(region_end - 1)), file_name)]
    
    samtools_call = sp.Popen(samtools_command, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    samtools_call.poll()
    
    cdef list samtools_lines = samtools_call.stdout.readlines()
    for step in range(num_grid * window_num):
        read_count[step] = float(str(samtools_lines[int(step * stride)])[:-3].rsplit('t',1)[1])
    
    return np.asarray(read_count)


def generateRefcounts(int region_start, int region_end, np.ndarray refGene, int num_grid):
    cdef float stride = (region_end - (region_start + 1)) / float(num_grid)
    cdef float[:] refGene_depth = np.empty(num_grid, dtype=np.float32)

    def searchRef(int bp, int s,int e):
        cdef int mid = int((e+s)/ 2)
        if mid == s:
            return mid
        else:
            if refGene[mid][0] < bp:
                return searchRef(bp, mid, e)
            elif refGene[mid][0] > bp:
                return searchRef(bp, s, mid)
            else:
                return mid
    
    cdef int L = len(refGene) - 1

    cdef int start_index = searchRef(region_start, 0, L)

    cdef int i = start_index
    cdef int end_index = start_index

    while L > i and region_end > refGene[i][0]:
        end_index = i
        i += 1

    cdef int location = 0 
    cdef char hit = 0
    
    for step in range(num_grid):
        location = int(region_start + stride * step)
        hit = 0

        for j in range(start_index, end_index + 1):
            if refGene[j][0] < location and location < refGene[j][1]:
                refGene_depth[step] = 1.
                hit = 1
                break
        if hit == 0:
            refGene_depth[step] = 0.

    return np.asarray(refGene_depth)
