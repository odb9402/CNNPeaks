cimport numpy as np
import numpy as np

cdef extern from "readbam.c":
    int* readbamMain(const char* input_file_name, const char* input_reg)
    
def genReadCount(file_name : bytes, reg : bytes, size : int):
    cdef int* read_counts = readbamMain(file_name, reg)
    cdef int[:] read_counts_np = np.empty(size, dtype=np.int32)
    
    for i in xrange(size):
        read_counts_np[i] = read_counts[i]
    for i in xrange(size):
        print(read_counts_np[i])
    
    return np.asarray(read_counts_np)