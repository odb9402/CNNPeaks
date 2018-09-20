/*  bam2depth.c -- depth subcommand.

    Copyright (C) 2011, 2012 Broad Institute.
    Copyright (C) 2012-2014 Genome Research Ltd.

    Author: Heng Li <lh3@sanger.ac.uk>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.  */

/* This program demonstrates how to generate pileup from multiple BAMs
 * simutaneously, to achieve random access and to use the BED interface.
 * To compile this program separately, you may:
 *
 *   gcc -g -O2 -Wall -o bam2depth -D_MAIN_BAM2DEPTH bam2depth.c -lhts -lz
 */

#include <python.h>
#include <config.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <unistd.h>
#include "htslib/sam.h"
#include "samtools.h"
#include "sam_opts.h"

static PyObject *

typedef struct {     // auxiliary data structure
    samFile *fp;     // the file handle
    bam_hdr_t *hdr;  // the file header
    hts_itr_t *iter; // NULL if a region not specified
    int min_mapQ, min_len; // mapQ filter; length filter
} aux_t;

// This function reads a BAM alignment from one BAM file.
static int read_bam(void *data, bam1_t *b) // read level filters better go here to avoid pileup
{
    aux_t *aux = (aux_t*)data; // data in fact is a pointer to an auxiliary structure
    int ret;
    while (1)
    {
        ret = aux->iter? sam_itr_next(aux->fp, aux->iter, b) : sam_read1(aux->fp, aux->hdr, b);
        if ( ret<0 ) break;
        if ( b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FQCFAIL | BAM_FDUP) ) continue;
        if ( (int)b->core.qual < aux->min_mapQ ) continue;
        if ( aux->min_len && bam_cigar2qlen(b->core.n_cigar, bam_get_cigar(b)) < aux->min_len ) continue;
        break;
    }
    return ret;
}

int read_file_list(const char *file_list,int *n,char **argv[]);

bamRead_main(PyObject *self, PyObject *args)
{
    int i, n, tid, reg_tid, beg, end, pos, *n_plp, baseQ = 0, mapQ = 0, min_len = 0;
    int all = 1,  nfiles, max_depth = -1;
    const bam_pileup1_t **plp;
    char *reg = 0; // specified region
    char *file_list = NULL, **fn = NULL;
    bam_hdr_t *h = NULL; // BAM header of the 1st input
    aux_t *data;
    bam_mplp_t mplp;
    int last_pos = -1, last_tid = -1, ret;
    
    PyObject *py_depth;
    int depth_len*;
    int depth_result*;
    int depth_index = 0;

    depth_result = (int)malloc((*depth_len) * sizeof(int))

    if(!PyArg_ParseTuple(args,  "ssi" , &reg, &file_list, &depth_len))
        return NULL;

    sam_global_args ga = SAM_GLOBAL_ARGS_INIT;
    static const struct option lopts[] = {
        SAM_OPT_GLOBAL_OPTIONS('-', 0, '-', '-', 0, '-'),
        { NULL, 0, NULL, 0 }
    };

    // initialize the auxiliary data structures
    if (file_list)
    {
        if ( read_file_list(file_list,&nfiles,&fn) ) return 1;
        n = nfiles;
        argv = fn;
        optind = 0;
    }
    else
        n = argc - optind; // the number of BAMs on the command line
    
    reg_tid = 0; beg = 0; end = INT_MAX;  // set the default region
    int rf;
    data = calloc(1, sizeof(aux_t));
    data->fp = sam_open_format(argv[optind+i], "r", &ga.in); // open BAM
    if (data[i]->fp == NULL) {
        print_error_errno("depth", "Could not open \"%s\"", argv[optind+i]);
        goto depth_end;
    }
    rf = SAM_FLAG | SAM_RNAME | SAM_POS | SAM_MAPQ | SAM_CIGAR | SAM_SEQ;
    if (baseQ)
        rf |= SAM_QUAL;
    if (hts_set_opt(data->fp, CRAM_OPT_REQUIRED_FIELDS, rf)) {
        fprintf(stderr, "Failed to set CRAM_OPT_REQUIRED_FIELDS value\n");
        return 1;
    }
    if (hts_set_opt(data->fp, CRAM_OPT_DECODE_MD, 0)) {
        fprintf(stderr, "Failed to set CRAM_OPT_DECODE_MD value\n");
        return 1;
    }
    data->min_mapQ = mapQ;                    // set the mapQ filter
    data->min_len  = min_len;                 // set the qlen filter
    data->hdr = sam_hdr_read(data->fp);    // read the BAM header
    if (data->hdr == NULL) {
        fprintf(stderr, "Couldn't read header for \"%s\"\n",
                argv[optind+i]);
        goto depth_end;
    }
    if (reg) { // if a region is specified
        hts_idx_t *idx = sam_index_load(data->fp, argv[optind+i]);  // load the index
        if (idx == NULL) {
            print_error("depth", "can't load index for \"%s\"", argv[optind+i]);
            goto depth_end;
        }
        data->iter = sam_itr_querys(idx, data->hdr, reg); // set the iterator
        hts_idx_destroy(idx); // the index is not needed any more; free the memory
        if (data->iter == NULL) {
            print_error("depth", "can't parse region \"%s\"", reg);
            goto depth_end;
        }
    }

    h = data->hdr; // easy access to the header of the 1st BAM
    if (reg) {
        beg = data->iter->beg; // and to the parsed region coordinates
        end = data->iter->end;
        reg_tid = data->iter->tid;
    }

    // the core multi-pileup loop
    mplp = bam_mplp_init(n, read_bam, (void*)data); // initialization
    bam_mplp_set_maxcnt(mplp,INT_MAX);

    n_plp = calloc(n, sizeof(int)); // n_plp[i] is the number of covering reads from the i-th BAM
    plp = calloc(n, sizeof(bam_pileup1_t*)); // plp[i] points to the array of covering reads (internal in mplp)
    while ((ret=bam_mplp_auto(mplp, &tid, &pos, n_plp, plp)) > 0) { // come to the next covered position
        if (pos < beg || pos >= end) continue; // out of range; skip
        if (tid >= h->n_targets) continue;     // diff number of @SQ lines per file?
        if (all) {
            while (tid > last_tid) {
                if (last_tid >= 0 && !reg) {
                    // Deal with remainder or entirety of last tid.
                    while (++last_pos < h->target_len[last_tid]) {
                        fputs(h->target_name[last_tid], stdout);
                        printf("\t%d", last_pos+1);
                        depth_result[depth_index] = last_pos + 1;
                        depth_index++;
                        for (i = 0; i < n; i++)
                            putchar('\t'), putchar('0');
                            depth_result[depth_index] = 0;
                            depth_index++;
                        putchar('\n');
                    }
                }
                last_tid++;
                last_pos = -1;
                if (all < 2)
                    break;
            }

            // Deal with missing portion of current tid
            while (++last_pos < pos) {
                if (last_pos < beg) continue; // out of range; skip
                fputs(h->target_name[tid], stdout); printf("\t%d", last_pos+1);
                depth_result[depth_index] = last_pos + 1;
                depth_index++;
                for (i = 0; i < n; i++){
                    depth_result[depth_index] = 0;
                    depth_index++;
                    putchar('\t');
                    putchar('0');
                }
                putchar('\n');
            }

            last_tid = tid;
            last_pos = pos;
        }
        fputs(h->target_name[tid], stdout); printf("\t%d", pos+1); // a customized printf() would be faster
        depth_result[depth_index] = last_pos + 1;
        depth_index++;
        for (i = 0; i < n; ++i) { // base level filters have to go here
            int j, m = 0;
            for (j = 0; j < n_plp[i]; ++j) {
                const bam_pileup1_t *p = plp[i] + j; // DON'T modfity plp[][] unless you really know
                if (p->is_del || p->is_refskip) ++m; // having dels or refskips at tid:pos
                else if (bam_get_qual(p->b)[p->qpos] < baseQ) ++m; // low base quality
            }
            printf("\t%d", n_plp[i] - m); // this the depth to output
            depth_result[depth_index] = n_pip[i] - m;
            depth_index++;
        }
        putchar('\n');
    }
    free(n_plp); free(plp);
    bam_mplp_destroy(mplp);

    if (all) {
        // Handle terminating region
        if (last_tid < 0 && reg && all > 1) {
            last_tid = reg_tid;
            last_pos = beg-1;
        }
        while (last_tid >= 0 && last_tid < h->n_targets) {
            while (++last_pos < h->target_len[last_tid]) {
                if (last_pos >= end) break;
                fputs(h->target_name[last_tid], stdout); printf("\t%d", last_pos+1);
                depth_result[depth_index] = last_pos + 1;
                depth_index++;
                for (i = 0; i < n; i++)
                    depth_result[depth_index] = 0;
                    depth_index++;
                    putchar('\t'), putchar('0');
                putchar('\n');
            }
            last_tid++;
            last_pos = -1;
            if (all < 2 || reg)
                break;
        }
    }

depth_end:
    bam_hdr_destroy(data->hdr);
    if (data->fp)
        sam_close(data->fp);
    hts_itr_destroy(data->iter);
    free(data);
    free(reg);
    free(fn);
    sam_global_args_free(&ga);

    for(i = 0; i < *depth_len; i ++){
        if(!PyList_SetItem(py_depth, i, PyLong_FromLong(depth_result[i])))
            printf("PyList Object Set Error.\n");
    }
    return py_depth;
}

static PyMethodDef bamDepthCount[] = {
    {"bamDepth", bamRead_main, METH_VARARGS, "main function of bam count extracted from samtools1.8."},
    {NULL, NULL, 0, NULL}
}

PyMODINIT_FUNC
initbamdepth(void){
    (void) Py_InitModule("bamDepth", bamDepthCount);
}

int main(int argc, char *argv[]){
    Py_SetProgramName(argv[0]);
    Py_Initalize();
    
}