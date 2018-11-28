import numpy as np
cimport numpy as np
import subprocess as sp

def generateReadcounts(int region_start, int region_end, str chr_num, str file_name, int num_grid):
    cdef list read_count_by_grid = []
    cdef int stride = (region_end - (region_start + 1)) / num_grid

    cdef list samtools_command = ['samtools depth -aa -r {} {}'.format(
        "{}:{}-{}".format(chr_num,region_start,(region_end - 1)), file_name)]
    samtools_call = sp.Popen(samtools_command, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    cdef list samtools_lines = samtools_call.stdout.readlines()
    samtools_call.poll()

    for i in range(len(samtools_lines)):
        samtools_lines[i] = int(str(samtools_lines[i])[:-3].rsplit('t',1)[1])

    for step in range(num_grid):
        read_count_by_grid.append(samtools_lines[(step * stride)])

    cdef np.ndarray read_count_nparr = np.array(read_count_by_grid, dtype=float)

    return read_count_nparr


def generateRefcounts(int region_start, int region_end, np.ndarray refGene, int num_grid):
    cdef int stride = (region_end - (region_start + 1)) / num_grid
    cdef list refGene_depth_list = []

    def searchRef(int bp, int s,int e):
        cdef int mid = ((e+s)/ 2)
        if mid == s:
            return mid
        else:
            if refGene[mid][0] < bp:
                return searchRef(bp, mid, e)
            elif refGene[mid][0] > bp:
                return searchRef(bp, s, mid)
            else:
                return mid

    cdef int L = len(refGene)

    cdef int start_index = searchRef(region_start, 0, L)

    cdef int i = start_index
    cdef int end_index = start_index

    while region_end > refGene[i][0]:
        end_index = i
        i += 1

    cdef int location = 0 
    cdef char hit = 0
    
    for step in range(num_grid):
        location = region_start + stride * step
        hit = 0

        for j in range(start_index, end_index + 1):
            if refGene[j][0] < location and location < refGene[j][1]:
                refGene_depth_list.append(1)
                hit = 1
                break
        if hit == 0:
            refGene_depth_list.append(0)

    cdef np.ndarray sub_refGene = np.array(refGene_depth_list, dtype=float)

    return sub_refGene

