/*
 * viterbi.cu
 * GPU-accelerated Viterbi decoder for Gene Prediction using GHMMs
 * Includes CPU reference implementation for benchmarking.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_75 viterbi.cu -o viterbi -lm
 *   (change sm_75 to match your GPU: sm_86 for RTX 30xx, sm_89 for RTX 40xx)
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Usage 1 — random synthetic sequence (original behaviour):
 *   ./viterbi <sequence_length>
 *
 * Usage 2 — single FASTA record (first record in file):
 *   ./viterbi <input.fa>
 *
 * Usage 3 — specific transcript from a multi-FASTA file:
 *   ./viterbi <input.fa> <TRANSCRIPT_ID>
 *
 * Usage 4 — with ground-truth accuracy comparison (pre-built label file):
 *   ./viterbi <input.fa> <TRANSCRIPT_ID> <labels.txt>  [output.txt]
 *
 * Usage 5 — pass a GTF directly; gtf_to_labels.py is called automatically:
 *   ./viterbi <input.fa> <TRANSCRIPT_ID> <annotation.gtf>  [output.txt]
 *
 * ──────────────────────────────────────────────────────────────────────────
 * FASTA format expected (UCSC-style per-transcript records):
 *   >hg38_knownGene_ENST00000852538.1 range=chr22:19969892-20016780 strand=- ...
 *
 * To pre-build label files from a GTF (recommended for repeated runs):
 *   python3 gtf_to_labels.py  Test_Sequence.fa  Test_Sequence_Labelled.gtf \
 *           -t ENST00000852538.1  -o ENST00000852538.1_labels.txt
 *   ./viterbi Test_Sequence.fa ENST00000852538.1 ENST00000852538.1_labels.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <ctype.h>
#include <assert.h>
#include <cuda_runtime.h>

/* ─── Constants ─────────────────────────────────────────────────────────── */
#define NUM_SYMBOLS   4
#define BLOCK_SIZE    9
#define LOG_ZERO     -1e30f
#define NUM_STATES    9
#define S             9

/* State indices */
#define ST_INTERGENIC 0
#define ST_START      1
#define ST_EXON_0     2
#define ST_EXON_1     3
#define ST_EXON_2     4
#define ST_DONOR      5
#define ST_INTRON     6
#define ST_ACCEPTOR   7
#define ST_STOP       8

/* Symbol encoding */
#define SYM_A 0
#define SYM_T 1
#define SYM_G 2
#define SYM_C 3

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 1 — FASTA I/O
   ═══════════════════════════════════════════════════════════════════════════ */

/*
 * nucleotide_to_symbol
 * IUPAC → 4-symbol alphabet, case-insensitive.
 * Returns -1 for characters that should be skipped.
 */
static int nucleotide_to_symbol(char c) {
    switch (toupper((unsigned char)c)) {
        case 'A':                                              return SYM_A;
        case 'T': case 'U':                                    return SYM_T;
        case 'G': case 'R': case 'S': case 'K':
        case 'B': case 'D':                                    return SYM_G;
        case 'C': case 'Y': case 'M': case 'H': case 'V':     return SYM_C;
        case 'W': case 'N':                                    return SYM_A;
        default:                                               return -1;
    }
}

/*
 * read_fasta_record
 *
 * Reads one record from a (possibly multi-) FASTA file.
 * If target_id is non-NULL and non-empty, searches for a record whose header
 * contains target_id; otherwise reads the first record.
 *
 * On success: *obs is heap-allocated (caller must free), *T is set.
 * Returns 0 on success, non-zero on error.
 */
int read_fasta_record(const char* filename,
                      const char* target_id,
                      int**       obs,
                      int*        T)
{
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open FASTA file '%s'\n", filename);
        return 1;
    }

    int   want_first = (!target_id || target_id[0] == '\0');
    int   in_target  = 0;
    long  seq_len    = 0;
    char  line[8192];

    /* ── Pass 1: locate record and count its sequence characters ── */
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '>') {
            if (in_target) break;
            if (want_first || strstr(line, target_id)) {
                in_target = 1;
                char* nl = strchr(line, '\n');
                if (nl) *nl = '\0';
                printf("[FASTA] Using record: %s\n", line + 1);
            }
            continue;
        }
        if (!in_target) continue;
        for (int i = 0; line[i] && line[i] != '\n' && line[i] != '\r'; i++)
            if (nucleotide_to_symbol(line[i]) >= 0) seq_len++;
    }

    if (!in_target || seq_len == 0) {
        if (target_id && target_id[0])
            fprintf(stderr, "Error: transcript '%s' not found in '%s'\n",
                    target_id, filename);
        else
            fprintf(stderr, "Error: no sequence data in '%s'\n", filename);
        fclose(fp);
        return 1;
    }

    *obs = (int*)malloc(seq_len * sizeof(int));
    if (!*obs) {
        fprintf(stderr, "Error: malloc failed (%ld bases)\n", seq_len);
        fclose(fp);
        return 1;
    }

    /* ── Pass 2: fill obs array ── */
    rewind(fp);
    in_target   = 0;
    long idx    = 0;
    int  warned = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '>') {
            if (in_target) break;
            if (want_first || strstr(line, target_id)) in_target = 1;
            continue;
        }
        if (!in_target) continue;
        for (int i = 0; line[i] && line[i] != '\n' && line[i] != '\r'; i++) {
            int sym = nucleotide_to_symbol(line[i]);
            if (sym < 0) {
                if (!warned && !isspace((unsigned char)line[i])) {
                    fprintf(stderr,
                            "Warning: unrecognised character '%c' near "
                            "position %ld — skipped.\n", line[i], idx);
                    warned = 1;
                }
                continue;
            }
            (*obs)[idx++] = sym;
        }
    }

    fclose(fp);
    *T = (int)idx;
    printf("[FASTA] Sequence length: %d nucleotides\n\n", *T);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 2 — PATH OUTPUT
   ═══════════════════════════════════════════════════════════════════════════ */

/*
 * write_path
 * Writes the decoded state path (one integer per line) with a comment header.
 */
int write_path(const char* filename,
               const int*  path,
               int         T,
               const char* transcript_id)
{
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open output file '%s'\n", filename);
        return 1;
    }
    fprintf(fp, "# Viterbi decoded state path\n");
    if (transcript_id && transcript_id[0])
        fprintf(fp, "# Transcript: %s\n", transcript_id);
    fprintf(fp, "# T=%d\n", T);
    fprintf(fp, "# State: 0=Intergenic 1=Start 2=Exon_0 3=Exon_1 4=Exon_2 "
                "5=Donor 6=Intron 7=Acceptor 8=Stop\n");
    for (int t = 0; t < T; t++)
        fprintf(fp, "%d\n", path[t]);
    fclose(fp);
    printf("[OUTPUT] Decoded path written to '%s'\n", filename);
    return 0;
}

/*
 * build_output_filename
 * Produces:  <fasta_basename>_<tid>_viterbi_path.txt
 */
static void build_output_filename(const char* fasta_path,
                                  const char* tid,
                                  char*       buf,
                                  size_t      buf_size)
{
    const char* base = strrchr(fasta_path, '/');
#ifdef _WIN32
    const char* b2 = strrchr(fasta_path, '\\');
    if (!base || (b2 && b2 > base)) base = b2;
#endif
    base = base ? base + 1 : fasta_path;
    strncpy(buf, base, buf_size - 1);
    buf[buf_size - 1] = '\0';
    char* dot = strrchr(buf, '.');
    if (dot) *dot = '\0';
    if (tid && tid[0]) {
        strncat(buf, "_",  buf_size - strlen(buf) - 1);
        strncat(buf, tid,  buf_size - strlen(buf) - 1);
    }
    strncat(buf, "_viterbi_path.txt", buf_size - strlen(buf) - 1);
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 3 — GROUND-TRUTH ACCURACY
   ═══════════════════════════════════════════════════════════════════════════ */

/* Returns 1 if filename ends with ext (case-insensitive). */
static int has_extension(const char* filename, const char* ext) {
    size_t fn = strlen(filename), fe = strlen(ext);
    if (fn < fe) return 0;
    const char* tail = filename + fn - fe;
    for (size_t i = 0; i < fe; i++)
        if (tolower((unsigned char)tail[i]) != tolower((unsigned char)ext[i]))
            return 0;
    return 1;
}

/*
 * invoke_gtf_to_labels
 * Calls  python3 gtf_to_labels.py <fasta> <gtf> [-t <tid>] -o <out>
 * Writes the generated label-file path into label_path buffer.
 */
static int invoke_gtf_to_labels(const char* fasta_path,
                                 const char* gtf_path,
                                 const char* tid,
                                 char*       label_path,
                                 size_t      label_path_size)
{
    if (tid && tid[0])
        snprintf(label_path, label_path_size, "%s_labels.txt", tid);
    else
        snprintf(label_path, label_path_size, "viterbi_gt_labels.txt");

    char cmd[4096];
    if (tid && tid[0])
        snprintf(cmd, sizeof(cmd),
                 "python3 gtf_to_labels.py \"%s\" \"%s\" -t \"%s\" -o \"%s\"",
                 fasta_path, gtf_path, tid, label_path);
    else
        snprintf(cmd, sizeof(cmd),
                 "python3 gtf_to_labels.py \"%s\" \"%s\" -o \"%s\"",
                 fasta_path, gtf_path, label_path);

    printf("[GTF] Invoking: %s\n\n", cmd);
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr,
                "Error: gtf_to_labels.py failed (exit %d).\n"
                "  Ensure gtf_to_labels.py is in the working directory or on PATH.\n",
                ret);
        return 1;
    }
    return 0;
}

/*
 * read_ground_truth
 * Reads whitespace/comma/newline-separated integers from a file.
 * Lines starting with '#' are comments.
 * Allocates *gt (caller must free).
 */
int read_ground_truth(const char* filename, int** gt, int* gt_len) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open ground-truth file '%s'\n", filename);
        return 1;
    }

    int  count = 0;
    char line[65536];

    /* Pass 1: count tokens */
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;
        char* p = line;
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == ',' ||
                   *p == '\n' || *p == '\r') p++;
            if (*p == '\0' || *p == '#') break;
            if (isdigit((unsigned char)*p) || *p == '-') {
                strtol(p, &p, 10); count++;
            } else p++;
        }
    }
    if (count == 0) {
        fprintf(stderr, "Error: no integer tokens in '%s'\n", filename);
        fclose(fp); return 1;
    }

    *gt = (int*)malloc(count * sizeof(int));
    if (!*gt) {
        fprintf(stderr, "Error: malloc failed for ground-truth\n");
        fclose(fp); return 1;
    }

    /* Pass 2: fill */
    rewind(fp);
    int idx = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') continue;
        char* p = line;
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == ',' ||
                   *p == '\n' || *p == '\r') p++;
            if (*p == '\0' || *p == '#') break;
            if (isdigit((unsigned char)*p) || *p == '-')
                (*gt)[idx++] = (int)strtol(p, &p, 10);
            else p++;
        }
    }
    fclose(fp);
    *gt_len = idx;
    return 0;
}

static const char* state_short_name(int s) {
    switch (s) {
        case ST_INTERGENIC: return "Intergenic";
        case ST_START:      return "StartCodon";
        case ST_EXON_0:     return "Exon_0(f0)";
        case ST_EXON_1:     return "Exon_1(f1)";
        case ST_EXON_2:     return "Exon_2(f2)";
        case ST_DONOR:      return "SpDonor   ";
        case ST_INTRON:     return "Intron    ";
        case ST_ACCEPTOR:   return "SpAcceptor";
        case ST_STOP:       return "StopCodon ";
        default:            return "Unknown   ";
    }
}

/*
 * compute_accuracy
 * Prints overall per-base accuracy, per-state precision/recall/F1,
 * and a full confusion matrix.
 */
void compute_accuracy(const int* pred, const int* truth, int T) {
    long conf[NUM_STATES][NUM_STATES];
    memset(conf, 0, sizeof(conf));
    long correct = 0;

    for (int t = 0; t < T; t++) {
        int p = (pred[t]  < 0 || pred[t]  >= NUM_STATES) ? 0 : pred[t];
        int g = (truth[t] < 0 || truth[t] >= NUM_STATES) ? 0 : truth[t];
        conf[g][p]++;
        if (p == g) correct++;
    }

    printf("\n================================================\n");
    printf("  Accuracy Report  (GPU prediction vs ground truth)\n");
    printf("================================================\n");
    printf("  Overall per-base accuracy: %.4f%%  (%ld / %d)\n\n",
           100.0 * correct / T, correct, T);

    printf("  %-12s  %8s  %8s  %9s  %10s  %10s\n",
           "State", "TP", "Prec(%)", "Recall(%)", "F1(%)", "Support");
    printf("  %-12s  %8s  %8s  %9s  %10s  %10s\n",
           "------------", "--------", "--------", "---------",
           "----------", "----------");

    for (int s = 0; s < NUM_STATES; s++) {
        long tp = conf[s][s], fp = 0, fn = 0;
        for (int k = 0; k < NUM_STATES; k++) {
            if (k != s) { fp += conf[k][s]; fn += conf[s][k]; }
        }
        long   support = tp + fn;
        double prec    = (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0.0;
        double recall  = support   > 0 ? 100.0 * tp / support   : 0.0;
        double f1      = (prec + recall) > 0
                         ? 2.0 * prec * recall / (prec + recall) : 0.0;
        printf("  %-12s  %8ld  %7.2f%%  %8.2f%%  %9.2f%%  %10ld\n",
               state_short_name(s), tp, prec, recall, f1, support);
    }

    printf("\n  Confusion matrix (rows = truth, cols = predicted):\n");
    printf("  %12s", "");
    for (int j = 0; j < NUM_STATES; j++)
        printf("  %10s", state_short_name(j));
    printf("\n");
    for (int i = 0; i < NUM_STATES; i++) {
        printf("  %-12s", state_short_name(i));
        for (int j = 0; j < NUM_STATES; j++)
            printf("  %10ld", conf[i][j]);
        printf("\n");
    }
    printf("================================================\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 4 — CPU VITERBI
   ═══════════════════════════════════════════════════════════════════════════ */
/*
 * Impliments the viterbi algorithm using the cpu
 *
 *  trans  [S x S]            log transition probabilities  trans[i*S + j]
 *  emit   [S x NUM_SYMBOLS]  log emission  probabilities   emit[s*NUM_SYMBOLS + obs]
 *  init   [S]                log initial state probabilities
 *  obs    [T]                observed symbol sequence (0..NUM_SYMBOLS-1)
 *  delta  [T x S]            DP table (output, caller allocates)
 *  psi    [T x S]            backpointer table (output, caller allocates)
 *  path   [T]                decoded state sequence (output, caller allocates)
 */
void cpu_viterbi(const float* trans,
                 const float* emit,
                 const float* init,
                 const int*   obs,
                 float*       delta,
                 int*         psi,
                 int*         path,
                 int          T)
{
    for (int s = 0; s < S; s++) {
        delta[s] = init[s] + emit[s * NUM_SYMBOLS + obs[0]];
        psi[s]   = -1;
    }
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < S; j++) {
            float best_val = LOG_ZERO; int best_i = 0;
            for (int i = 0; i < S; i++) {
                float v = delta[(t-1)*S + i] + trans[i*S + j];
                if (v > best_val) { best_val = v; best_i = i; }
            }
            delta[t*S + j] = best_val + emit[j * NUM_SYMBOLS + obs[t]];
            psi[t*S + j]   = best_i;
        }
    }
    float best_val = LOG_ZERO; int best_s = 0;
    for (int s = 0; s < S; s++)
        if (delta[(T-1)*S + s] > best_val) { best_val = delta[(T-1)*S + s]; best_s = s; }
    path[T-1] = best_s;
    for (int t = T-2; t >= 0; t--)
        path[t] = psi[(t+1)*S + path[t+1]];
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 5 — GPU KERNELS
   ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Fills the first row of the DP table (t = 0).
 * Uses Map parallelism, one thread per state j.
 */
__global__ void viterbi_init_kernel(float*       delta,
                                    int*         psi,
                                    const float* init,
                                    const float* emit,
                                    const int*   obs)
{
    int j = threadIdx.x;
    if (j >= S) return;
    delta[j] = init[j] + emit[j * NUM_SYMBOLS + obs[0]];
    psi[j]   = -1;
}

/*
 * Fills out the DP Table
 * Uses Map parallelism, one thread per state j and Shared memory
 */
__global__ void viterbi_step_kernel(float*       delta,
                                    int*         psi,
                                    const float* trans,
                                    const float* emit,
                                    const int*   obs,
                                    int          T)
{
    __shared__ float s_prev[NUM_STATES];
    int j = threadIdx.x;
    for (int t = 1; t < T; t++) {
        if (j < NUM_STATES) s_prev[j] = delta[(t-1)*NUM_STATES + j];
        __syncthreads();
        if (j < NUM_STATES) {
            float best_val = LOG_ZERO; int best_i = 0;
            for (int i = 0; i < NUM_STATES; i++) {
                float v = s_prev[i] + trans[i*NUM_STATES + j];
                if (v > best_val) { best_val = v; best_i = i; }
            }
            delta[t*NUM_STATES + j] = best_val + emit[j * NUM_SYMBOLS + obs[t]];
            psi[t*NUM_STATES + j]   = best_i;
        }
        __syncthreads();
    }
}

/*
 * Finds the argmax over the final row of delta (reduction over S).
 * Uses parallel reduction
 */
__global__ void viterbi_termination_kernel(const float* delta,
                                           int*         best_state,
                                           int          T)
{
    __shared__ float s_val[NUM_STATES];
    __shared__ int   s_idx[NUM_STATES];
    int tid = threadIdx.x;
    s_val[tid] = delta[(T-1)*NUM_STATES + tid];
    s_idx[tid] = tid;
    __syncthreads();
    for (int stride = NUM_STATES / 2; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < NUM_STATES)
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        __syncthreads();
    }
    if (tid == 0) *best_state = s_idx[0];
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 6 — GPU VITERBI HOST WRAPPER
   ═══════════════════════════════════════════════════════════════════════════ */

void gpu_viterbi(const float* h_trans,
                 const float* h_emit,
                 const float* h_init,
                 const int*   h_obs,
                 int*         h_path,
                 int          T,
                 float*       kernel_ms)
{
    size_t sz_delta = (size_t)T * S * sizeof(float);
    size_t sz_psi   = (size_t)T * S * sizeof(int);
    size_t sz_trans = (size_t)S * S * sizeof(float);
    size_t sz_emit  = (size_t)S * NUM_SYMBOLS * sizeof(float);
    size_t sz_init  = (size_t)S * sizeof(float);
    size_t sz_obs   = (size_t)T * sizeof(int);

    float *d_delta, *d_trans, *d_emit, *d_init;
    int   *d_psi, *d_obs, *d_best;

    cudaMalloc(&d_delta, sz_delta); cudaMalloc(&d_psi,   sz_psi);
    cudaMalloc(&d_trans, sz_trans); cudaMalloc(&d_emit,  sz_emit);
    cudaMalloc(&d_init,  sz_init);  cudaMalloc(&d_obs,   sz_obs);
    cudaMalloc(&d_best,  sizeof(int));

    cudaMemcpy(d_trans, h_trans, sz_trans, cudaMemcpyHostToDevice);
    cudaMemcpy(d_emit,  h_emit,  sz_emit,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_init,  h_init,  sz_init,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs,   h_obs,   sz_obs,   cudaMemcpyHostToDevice);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start);

    viterbi_init_kernel<<<1, BLOCK_SIZE>>>(d_delta, d_psi, d_init, d_emit, d_obs);
    cudaGetLastError();
    viterbi_step_kernel<<<1, BLOCK_SIZE>>>(d_delta, d_psi, d_trans, d_emit, d_obs, T);
    viterbi_termination_kernel<<<1, BLOCK_SIZE>>>(d_delta, d_best, T);
    cudaGetLastError();

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(kernel_ms, ev_start, ev_stop);

    int* h_psi = (int*)malloc(sz_psi);
    if (!h_psi) { fprintf(stderr, "malloc psi failed\n"); exit(1); }
    cudaMemcpy(h_psi, d_psi, sz_psi, cudaMemcpyDeviceToHost);

    int h_best;
    cudaMemcpy(&h_best, d_best, sizeof(int), cudaMemcpyDeviceToHost);

    h_path[T-1] = h_best;
    for (int t = T-2; t >= 0; t--)
        h_path[t] = h_psi[(t+1)*S + h_path[t+1]];

    free(h_psi);
    cudaFree(d_delta); cudaFree(d_psi);  cudaFree(d_trans);
    cudaFree(d_emit);  cudaFree(d_init); cudaFree(d_obs);
    cudaFree(d_best);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 7 — GENE MODEL
   ═══════════════════════════════════════════════════════════════════════════ */

static void build_gene_model(float* trans, float* emit, float* init) {
    for (int i = 0; i < S * S;           i++) trans[i] = LOG_ZERO;
    for (int i = 0; i < S * NUM_SYMBOLS; i++) emit[i]  = LOG_ZERO;
    for (int i = 0; i < S;               i++) init[i]  = LOG_ZERO;

#define LT(p) logf(p)
    trans[ST_INTERGENIC*S + ST_INTERGENIC] = LT(0.990f);
    trans[ST_INTERGENIC*S + ST_START]      = LT(0.010f);
    trans[ST_START*S      + ST_EXON_0]     = LT(1.000f);
    trans[ST_EXON_0*S     + ST_EXON_1]     = LT(0.940f);
    trans[ST_EXON_0*S     + ST_DONOR]      = LT(0.050f);
    trans[ST_EXON_0*S     + ST_STOP]       = LT(0.010f);
    trans[ST_EXON_1*S     + ST_EXON_2]     = LT(0.950f);
    trans[ST_EXON_1*S     + ST_DONOR]      = LT(0.050f);
    trans[ST_EXON_2*S     + ST_EXON_0]     = LT(0.950f);
    trans[ST_EXON_2*S     + ST_DONOR]      = LT(0.050f);
    trans[ST_DONOR*S      + ST_INTRON]     = LT(1.000f);
    trans[ST_INTRON*S     + ST_INTRON]     = LT(0.970f);
    trans[ST_INTRON*S     + ST_ACCEPTOR]   = LT(0.030f);
    trans[ST_ACCEPTOR*S   + ST_EXON_0]     = LT(1.000f);
    trans[ST_STOP*S       + ST_INTERGENIC] = LT(1.000f);

    emit[ST_INTERGENIC*NUM_SYMBOLS + SYM_A] = LT(0.26f);
    emit[ST_INTERGENIC*NUM_SYMBOLS + SYM_T] = LT(0.24f);
    emit[ST_INTERGENIC*NUM_SYMBOLS + SYM_G] = LT(0.26f);
    emit[ST_INTERGENIC*NUM_SYMBOLS + SYM_C] = LT(0.24f);

    emit[ST_START*NUM_SYMBOLS + SYM_A] = LT(0.60f);
    emit[ST_START*NUM_SYMBOLS + SYM_T] = LT(0.15f);
    emit[ST_START*NUM_SYMBOLS + SYM_G] = LT(0.15f);
    emit[ST_START*NUM_SYMBOLS + SYM_C] = LT(0.10f);

    emit[ST_EXON_0*NUM_SYMBOLS + SYM_A] = LT(0.22f);
    emit[ST_EXON_0*NUM_SYMBOLS + SYM_T] = LT(0.18f);
    emit[ST_EXON_0*NUM_SYMBOLS + SYM_G] = LT(0.30f);
    emit[ST_EXON_0*NUM_SYMBOLS + SYM_C] = LT(0.30f);

    emit[ST_EXON_1*NUM_SYMBOLS + SYM_A] = LT(0.25f);
    emit[ST_EXON_1*NUM_SYMBOLS + SYM_T] = LT(0.20f);
    emit[ST_EXON_1*NUM_SYMBOLS + SYM_G] = LT(0.30f);
    emit[ST_EXON_1*NUM_SYMBOLS + SYM_C] = LT(0.25f);

    emit[ST_EXON_2*NUM_SYMBOLS + SYM_A] = LT(0.20f);
    emit[ST_EXON_2*NUM_SYMBOLS + SYM_T] = LT(0.20f);
    emit[ST_EXON_2*NUM_SYMBOLS + SYM_G] = LT(0.30f);
    emit[ST_EXON_2*NUM_SYMBOLS + SYM_C] = LT(0.30f);

    emit[ST_DONOR*NUM_SYMBOLS + SYM_A] = LT(0.05f);
    emit[ST_DONOR*NUM_SYMBOLS + SYM_T] = LT(0.10f);
    emit[ST_DONOR*NUM_SYMBOLS + SYM_G] = LT(0.80f);
    emit[ST_DONOR*NUM_SYMBOLS + SYM_C] = LT(0.05f);

    emit[ST_INTRON*NUM_SYMBOLS + SYM_A] = LT(0.35f);
    emit[ST_INTRON*NUM_SYMBOLS + SYM_T] = LT(0.35f);
    emit[ST_INTRON*NUM_SYMBOLS + SYM_G] = LT(0.15f);
    emit[ST_INTRON*NUM_SYMBOLS + SYM_C] = LT(0.15f);

    emit[ST_ACCEPTOR*NUM_SYMBOLS + SYM_A] = LT(0.10f);
    emit[ST_ACCEPTOR*NUM_SYMBOLS + SYM_T] = LT(0.05f);
    emit[ST_ACCEPTOR*NUM_SYMBOLS + SYM_G] = LT(0.80f);
    emit[ST_ACCEPTOR*NUM_SYMBOLS + SYM_C] = LT(0.05f);

    emit[ST_STOP*NUM_SYMBOLS + SYM_A] = LT(0.40f);
    emit[ST_STOP*NUM_SYMBOLS + SYM_T] = LT(0.45f);
    emit[ST_STOP*NUM_SYMBOLS + SYM_G] = LT(0.10f);
    emit[ST_STOP*NUM_SYMBOLS + SYM_C] = LT(0.05f);

    init[ST_INTERGENIC] = LT(0.98f);
    init[ST_START]      = LT(0.02f);
#undef LT
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 8 — UTILITIES
   ═══════════════════════════════════════════════════════════════════════════ */

void random_obs(int* obs, int T, unsigned* seed) {
    for (int t = 0; t < T; t++) obs[t] = rand_r(seed) % NUM_SYMBOLS;
}

int paths_match(const int* a, const int* b, int T) {
    for (int t = 0; t < T; t++) if (a[t] != b[t]) return 0;
    return 1;
}


static void print_usage(const char* prog) {
    fprintf(stderr,
        "\nUsage:\n"
        "  %s <length>                                      (random sequence)\n"
        "  %s <input.fa>                                    (first FASTA record)\n"
        "  %s <input.fa> <TRANSCRIPT_ID>                    (named record)\n"
        "  %s <input.fa> <TRANSCRIPT_ID> <labels.txt>       (with accuracy)\n"
        "  %s <input.fa> <TRANSCRIPT_ID> <annot.gtf>        (GTF auto-convert)\n"
        "  %s <input.fa> <TRANSCRIPT_ID> <gt_source> <out>  (explicit output)\n"
        "\n"
        "  gtf_to_labels.py must be in the working directory or on PATH\n"
        "  when passing a .gtf file as the ground-truth source.\n",
        prog, prog, prog, prog, prog, prog);
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION 9 — MAIN
   ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {

    if (argc < 2) { print_usage(argv[0]); return 1; }

    /* ── Parse arguments ──────────────────────────────────────────────────── */
    int  use_fasta  = 0;
    int  T          = 10000;
    char fasta_path[1024]  = {0};
    char tid[512]          = {0};    /* transcript ID                         */
    char gt_source[1024]   = {0};    /* labels.txt or annotation.gtf          */
    char out_path[1024]    = {0};    /* output decoded-path file              */
    char auto_labels[1024] = {0};    /* auto-generated label file from GTF   */

    {
        /* Detect mode: integer → random; otherwise → FASTA file */
        int is_int = 1;
        for (int i = 0; argv[1][i]; i++)
            if (!isdigit((unsigned char)argv[1][i])) { is_int = 0; break; }

        if (is_int) {
            T = atoi(argv[1]);
            if (T <= 0) { fprintf(stderr, "Error: length must be > 0\n"); return 1; }
        } else {
            use_fasta = 1;
            strncpy(fasta_path, argv[1], sizeof(fasta_path) - 1);

            /* argv[2]: transcript ID (if it doesn't look like a file path) */
            if (argc >= 3) {
                int looks_like_file = has_extension(argv[2], ".gtf")  ||
                                      has_extension(argv[2], ".gff")  ||
                                      has_extension(argv[2], ".txt")  ||
                                      has_extension(argv[2], ".fa")   ||
                                      has_extension(argv[2], ".fasta");
                if (!looks_like_file) {
                    strncpy(tid, argv[2], sizeof(tid) - 1);
                } else {
                    /* argv[2] is already the gt_source */
                    strncpy(gt_source, argv[2], sizeof(gt_source) - 1);
                }
            }

            /* argv[3]: ground-truth source */
            if (argc >= 4 && gt_source[0] == '\0')
                strncpy(gt_source, argv[3], sizeof(gt_source) - 1);

            /* argv[4]: explicit output path */
            if (argc >= 5)
                strncpy(out_path, argv[4], sizeof(out_path) - 1);
        }
    }

    if (use_fasta && out_path[0] == '\0')
        build_output_filename(fasta_path, tid, out_path, sizeof(out_path));

    /* ── Banner ───────────────────────────────────────────────────────────── */
    printf("================================================\n");
    printf(" GPU Viterbi for Gene Prediction\n");
    printf(" 9-state GHMM:  Intergenic → Start → Exon_0\n");
    printf("                → Exon_1 → Exon_2 → Donor\n");
    printf("                → Intron → Acceptor → [Exon_0]\n");
    printf("                → Stop → Intergenic\n");
    if (use_fasta) {
        printf(" Input FASTA  : %s\n", fasta_path);
        if (tid[0])       printf(" Transcript   : %s\n", tid);
        if (gt_source[0]) printf(" Ground truth : %s\n", gt_source);
        printf(" Output file  : %s\n", out_path);
    }
    printf("================================================\n\n");

    /* ── Build gene model ─────────────────────────────────────────────────── */
    float* trans = (float*)malloc(S * S           * sizeof(float));
    float* emit  = (float*)malloc(S * NUM_SYMBOLS * sizeof(float));
    float* init  = (float*)malloc(S               * sizeof(float));
    if (!trans || !emit || !init) {
        fprintf(stderr, "malloc failed (model)\n"); return 1;
    }
    build_gene_model(trans, emit, init);

    /* ── Load sequence ────────────────────────────────────────────────────── */
    int* obs = NULL;
    if (use_fasta) {
        if (read_fasta_record(fasta_path, tid[0] ? tid : NULL, &obs, &T) != 0)
            return 1;
    } else {
        obs = (int*)malloc(T * sizeof(int));
        if (!obs) { fprintf(stderr, "malloc obs failed\n"); return 1; }
        unsigned seed = 42;
        random_obs(obs, T, &seed);
        printf("[INFO] Random sequence, T = %d\n\n", T);
    }

    printf(" Sequence length T = %d\n\n", T);

    /* ── Allocate DP tables ───────────────────────────────────────────────── */
    float* cpu_delta = (float*)malloc((size_t)T * S * sizeof(float));
    int*   cpu_psi   = (int*)  malloc((size_t)T * S * sizeof(int));
    int*   cpu_path  = (int*)  malloc(T * sizeof(int));
    int*   gpu_path  = (int*)  malloc(T * sizeof(int));
    if (!cpu_delta || !cpu_psi || !cpu_path || !gpu_path) {
        fprintf(stderr, "DP table malloc failed\n"); return 1;
    }

    /* ════════════════════════════════
       STEP A — CPU Viterbi
       ════════════════════════════════ */
    printf("[CPU] Running Viterbi...\n");
    struct timespec ts, te;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    cpu_viterbi(trans, emit, init, obs, cpu_delta, cpu_psi, cpu_path, T);
    clock_gettime(CLOCK_MONOTONIC, &te);
    double cpu_ms = (te.tv_sec - ts.tv_sec) * 1e3
                  + (te.tv_nsec - ts.tv_nsec) * 1e-6;
    printf("[CPU] Done. Time = %.3f ms\n\n", cpu_ms);

    /* ════════════════════════════════
       STEP B — GPU Viterbi
       ════════════════════════════════ */
    printf("[GPU] Running Viterbi (parallel reduction)...\n");
    float gpu_ms = 0.0f;
    gpu_viterbi(trans, emit, init, obs, gpu_path, T, &gpu_ms);
    printf("[GPU] Done. Kernel time = %.3f ms\n", gpu_ms);
    printf("[GPU] Speedup over CPU  = %.2fx\n\n", (float)cpu_ms / gpu_ms);

    /* ════════════════════════════════
       STEP C — CPU vs GPU check
       ════════════════════════════════ */
    if (paths_match(cpu_path, gpu_path, T))
        printf("[CORRECTNESS] ✓ GPU path matches CPU path exactly.\n\n");
    else
        printf("[CORRECTNESS] ✗ MISMATCH between GPU and CPU paths.\n\n");

    /* ════════════════════════════════
       STEP D — Write output file
       ════════════════════════════════ */
    if (use_fasta)
        write_path(out_path, gpu_path, T, tid[0] ? tid : NULL);

    /* ════════════════════════════════
       STEP E — Ground-truth accuracy
       ════════════════════════════════ */
    if (gt_source[0]) {
        const char* label_file = gt_source;

        /* Auto-convert GTF → label file */
        if (has_extension(gt_source, ".gtf") || has_extension(gt_source, ".gff")) {
            if (invoke_gtf_to_labels(fasta_path, gt_source,
                                     tid[0] ? tid : NULL,
                                     auto_labels, sizeof(auto_labels)) != 0) {
                fprintf(stderr, "Warning: GTF conversion failed — skipping accuracy.\n");
                goto skip_accuracy;
            }
            label_file = auto_labels;
        }

        int* gt     = NULL;
        int  gt_len = 0;
        printf("\n[ACCURACY] Loading labels from '%s'...\n", label_file);
        if (read_ground_truth(label_file, &gt, &gt_len) == 0) {
            if (gt_len != T)
                fprintf(stderr,
                        "Warning: label count (%d) != sequence length (%d). "
                        "Comparing first %d positions.\n",
                        gt_len, T, (gt_len < T ? gt_len : T));
            int cmp_len = (gt_len < T) ? gt_len : T;
            printf("[ACCURACY] Comparing %d positions...\n", cmp_len);
            compute_accuracy(gpu_path, gt, cmp_len);
            free(gt);
        }
    }
    skip_accuracy:;

    /* ════════════════════════════════
       STEP F — Summary table
       ════════════════════════════════ */
    printf("\n================================================\n");
    printf("  Benchmark Summary\n");
    printf("================================================\n");
    printf("  %-28s %10.3f ms\n",          "CPU Viterbi", cpu_ms);
    printf("  %-28s %10.3f ms  (%.1fx)\n", "GPU Viterbi", gpu_ms,
           (float)cpu_ms / gpu_ms);
    if (use_fasta)
        printf("  %-28s %s\n", "Output path file", out_path);
    if (gt_source[0])
        printf("  %-28s %s\n", "Ground-truth source", gt_source);
    printf("================================================\n");

    free(trans); free(emit); free(init); free(obs);
    free(cpu_delta); free(cpu_psi); free(cpu_path); free(gpu_path);
    return 0;
}
