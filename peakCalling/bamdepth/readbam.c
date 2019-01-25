#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <htslib/bgzf.h>
#include <htslib/sam.h>

typedef BGZF* bamFile;
typedef hts_itr_t *bam_iter_t;
typedef bam_hdr_t bam_header_t;
typedef hts_idx_t bam_index_t;

typedef struct {     // auxiliary data structure
    bamFile fp;      // the file handler
    bam_iter_t iter; // NULL if a region not specified
    int min_mapQ, min_len; // mapQ filter; length filter
} aux_t;

int bam_iter_read(bamFile fp, bam_iter_t iter, bam1_t *b)
{
    return iter? hts_itr_next(fp, iter, b, 0) : bam_read1(fp, b);
}

// This function reads a BAM alignment from one BAM file.
static int read_bam(void *data, bam1_t *b) // read level filters better go here to avoid pileup
{
  aux_t *aux = (aux_t*)data; // data in fact is a pointer to an auxiliary structure
  int ret;
  while (1)
  {
    ret = aux->iter? bam_iter_read(aux->fp, aux->iter, b) : bam_read1(aux->fp, b);
    if ( ret<0 ) break;
    if ( b->core.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FQCFAIL | BAM_FDUP) ) continue;
    if ( (int)b->core.qual < aux->min_mapQ ) continue;
    if ( aux->min_len && bam_cigar2qlen(&b->core, bam_get_cigar(b)) < aux->min_len ) continue;
    break;
  }
  return ret;

}

int bam_parse_region(bam_header_t *header, const char *str, int *ref_id, int *beg, int *end)
{
    const char *name_lim = hts_parse_reg(str, beg, end);
    char *name = malloc(name_lim - str + 1);
    memcpy(name, str, name_lim - str);
    name[name_lim - str] = '\0';
    *ref_id = bam_name2id(header, name);
    free(name);
    if (*ref_id == -1) return -1;
    return *beg <= *end? 0 : -1;
}

int* readbamMain(const char* input_file_name, const char* input_reg)
{
    int i, n, tid, beg, end, pos, *n_plp, baseQ = 0, mapQ = 0, min_len = 0;
    const bam_pileup1_t **plp;
    char *file_name = 0;
    char *reg = 0; // specified region
    bam_header_t *h = 0; // BAM header of the 1st input
    aux_t **data;
    bam_mplp_t mplp;
    int prev_tid = -1;  // the id of the previous positions tid
    int prev_pos = -1;
    int *output_depths; // Result depths will be stored here.
    int output_idx = 0;
    
    file_name = input_file_name;
    reg = input_reg;

    // initialize the auxiliary data structures
    n = 1; // the number of BAMs on the command line
    data = calloc(n, sizeof(void*)); // data[i] for the i-th input
    beg = 0; end = 1<<30; tid = -1;  // set the default region
    
    for (i = 0; i < n; ++i) {
        bam_header_t *htmp;
        data[i] = calloc(1, sizeof(aux_t));
        data[i]->fp = bgzf_open(file_name, "r"); // open BAM
        data[i]->min_mapQ = mapQ;                    // set the mapQ filter
        data[i]->min_len  = min_len;                 // set the qlen filter
        htmp = bam_hdr_read(data[i]->fp);         // read the BAM header
        if (i == 0) {
            h = htmp; // keep the header of the 1st BAM
            if (reg) bam_parse_region(h, reg, &tid, &beg, &end); // also parse the region
        } else bam_hdr_destroy(htmp); // if not the 1st BAM, trash the header
        if (tid >= 0) { // if a region is specified and parsed successfully
            bam_index_t *idx = bam_index_load(file_name);  // load the index
            data[i]->iter = bam_itr_queryi(idx, tid, beg, end); // set the iterator
            hts_idx_destroy(idx); // the index is not needed any more; phase out of the memory
        }
    }
    
    output_depths = calloc(end-beg+1, sizeof(int));
    
    // the core multi-pileup loop
    mplp = bam_mplp_init(n, read_bam, (void**)data); // initialization
    n_plp = calloc(n, sizeof(int)); // n_plp[i] is the number of covering reads from the i-th BAM
    plp = calloc(n, sizeof(void*)); // plp[i] points to the array of covering reads (internal in mplp)
    
    while (bam_mplp_auto(mplp, &tid, &pos, n_plp, plp) > 0){ // come to the next covered position
        if (pos < beg || pos >= end)
            continue; // out of range; skip
        
        // The skipped region which has 0 coverage is processed in here. (When chomosome[TID] was changed.)
        while(tid > prev_tid){
            if(prev_tid >= 0 && !reg){
                while(++prev_pos < h->target_len[prev_tid]){
                    printf("pos0 : %d\t0\n",prev_pos);
                    output_depths[output_idx++] = 0;
                }
            }
            prev_tid++;
            prev_pos = -1;
        }
        
        // Processing missing portion of current tid.
        while (++prev_pos < pos) {
            if (prev_pos < beg)
                continue;
            printf("pos1 : %d\t0\n",prev_pos);
            output_depths[output_idx++] = 0;
        }
        
        prev_tid = tid;
        prev_pos = pos;
        
        for (i = 0; i < n; ++i) { // base level filters have to go here
            int j, m = 0;
            for (j = 0; j < n_plp[i]; ++j) {
                const bam_pileup1_t *p = plp[i] + j; // DON'T modfity plp[][] unless you really know
                if (p->is_del || p->is_refskip) ++m; // having dels or refskips at tid:pos
                else if (bam_get_qual(p->b)[p->qpos] < baseQ) ++m; // low base quality
            }
            printf("pos2 : %d\t%d", pos,n_plp[i] - m); // Read depth output (Only for depth>=1) 
            output_depths[output_idx++] = n_plp[i]-m;
        }
        putchar('\n');
    }
    free(n_plp);
    free(plp);
    bam_mplp_destroy(mplp);

    // If there is absolutly no read in the regions or..., the routine start here.
    // Remained regions will be all "0" for their depth.
    if (prev_tid < 0 && reg) {
        prev_tid = tid;
        prev_pos = beg-1;
    }
    while (prev_tid >= 0 && prev_tid < h->n_targets) {
        while (++prev_pos < h->target_len[prev_tid]) {
            if (prev_pos >= end)
                break;
            printf("pos3 : %d\t",prev_pos);
            printf("0\n");
            output_depths[output_idx++] = 0;
        }
        prev_tid++;
        prev_pos = -1;
        if(reg)
            break;
    }
    
    bam_hdr_destroy(h);
    for (i = 0; i < n; ++i) {
        bgzf_close(data[i]->fp);
        if (data[i]->iter) bam_itr_destroy(data[i]->iter);
        free(data[i]);
    }
    free(data); 
    
    return output_depths;
}
/*
static PyMethodDef readbamMethods[] = {
    {"gen_count", readbamMain, METH_VARARGS, "return read mapping counts."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef readbammodule = {
    PyModuleDef_HEAD_INIT,
    "readbam", // name of module
    NULL,      // documentation
    -1,        // size of per-interpreter
    readbamMethods
};

PyMODINIT_FUNC PyInit_readbam(void){
    return PyModule_Create(&readbammodule);
}*/