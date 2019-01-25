#ifndef __BEDIDX_H__
#define __BEDIDX_H__

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <zlib.h>

void *bed_read(const char *fn); // read a BED or position list file
void bed_destroy(void *_h);     // destroy the BED data structure
int bed_overlap(const void *_h, const char *chr, int beg, int end); // test if chr:beg-end overlaps
int *bed_index_core(int n, uint64_t *a, int *n_idx);


#endif /* !__BEDIDX_H__ */

