# GPU-Accelerated Viterbi Gene Prediction

A CUDA implementation of the Viterbi algorithm for *ab initio* gene prediction using a 9-state Generalized Hidden Markov Model (GHMM). Includes a CPU reference implementation for correctness verification and benchmarking.

Built for ECE 213 — Wi26, UCSD.

---

## How It Works

The decoder models gene structure as a sequence of hidden states inferred from raw DNA. Each nucleotide is assigned one of 9 biological states:

| State | Index | Description |
|---|---|---|
| Intergenic | 0 | Non-coding background |
| Start Codon | 1 | ATG translation start |
| Exon\_0 | 2 | Coding exon, reading frame 0 |
| Exon\_1 | 3 | Coding exon, reading frame 1 |
| Exon\_2 | 4 | Coding exon, reading frame 2 |
| Splice Donor | 5 | GT dinucleotide at intron start |
| Intron | 6 | Non-coding intronic region |
| Splice Acceptor | 7 | AG dinucleotide at intron end |
| Stop Codon | 8 | TAA / TAG / TGA |

The GPU parallelises across states at each timestep using shared memory, with a parallel reduction for the termination step. Backtracking runs on the CPU after copying the `ψ` table back from device memory.

---

## Repository Structure

```
.all transcripts
├── GHMMGenePredictor/
    └── viterbi.cu                 # GPU + CPU Viterbi decoder (CUDA)
    └── extract_all_labels.py      # Data Preprocessing script
    └── run_viterbi_all.sh         # Batch runner for entrypoint
    └── Test_Sequence.fa           # hg38 chr22 test sequences (1145 transcripts)
    └── Test_Sequence_Labelled.gtf # Ground-truth annotations
```

---

## Requirements

- CUDA toolkit (nvcc)
- Python 3.6+
- A GPU (tested on NVIDIA A30)

---

## Quick Start

### 1. Compile

```bash
nvcc -O3 -arch=sm_75 viterbi.cu -o viterbi -lm
```

> Change `-arch` to match your GPU: `sm_86` for RTX 30xx, `sm_89` for RTX 40xx.

### 2. Prepare ground-truth labels

The label extractor reads the provided FASTA and GTF files and writes one integer-per-line label file per transcript into an output directory.

```bash
python3 extract_all_labels.py Test_Sequence.fa Test_Sequence_Labelled.gtf -o labels/
```

A test dataset from **hg38 chromosome 22** (1145 transcripts, ~21M nucleotides) is preloaded in this repository and can be run directly with the command above.

**Options:**

```bash
# Validate splice sites and codon sequences, print state distribution
python3 extract_all_labels.py Test_Sequence.fa Test_Sequence_Labelled.gtf -o labels/ --validate --summary

# Process specific transcripts only
python3 extract_all_labels.py Test_Sequence.fa Test_Sequence_Labelled.gtf -o labels/ -t ENST00000852538.1,ENST00000263207.8

# Parallel extraction (e.g. 8 cores)
python3 extract_all_labels.py Test_Sequence.fa Test_Sequence_Labelled.gtf -o labels/ -j 8
```

This produces:
- `labels/<TRANSCRIPT_ID>_labels.txt` — state label file per transcript
- `labels/manifest.tsv` — index of all transcripts with lengths and file paths

### 3. Run Viterbi on all sequences

#### On DSMLP (recommended)

```bash
/opt/launch-sh/bin/launch.sh -v a30 -c 8 -g 1 -m 8 -i yatisht/ece213-wi26:latest -f ./ECE213FinalProject/GHMMGenePredictor/run_viterbi_all.sh
```

This submits a job to the cluster, runs the decoder across all 1145 transcripts, and reports per-base accuracy against the GTF ground truth.

#### Locally

```bash
chmod +x run_viterbi_all.sh
./run_viterbi_all.sh \
    -f Test_Sequence.fa \
    -m labels/manifest.tsv \
    -v ./viterbi \
    -o viterbi_output/
```

### 4. Single-transcript run

```bash
# First record in the FASTA
./viterbi Test_Sequence.fa

# Specific transcript
./viterbi Test_Sequence.fa ENST00000852538.1

# With ground-truth accuracy report
./viterbi Test_Sequence.fa ENST00000852538.1 labels/ENST00000852538.1_labels.txt

# Pass the GTF directly — label file is generated automatically
./viterbi Test_Sequence.fa ENST00000852538.1 Test_Sequence_Labelled.gtf
```

---

## Output

| File | Description |
|---|---|
| `viterbi_output/<TID>_viterbi_path.txt` | Decoded state sequence, one integer per line |
| `viterbi_summary.csv` | Per-transcript: GPU ms, CPU ms, speedup, CPU-match, accuracy % |
| `viterbi_run.log` | Full stdout from every viterbi call |

The accuracy report printed per transcript includes overall per-base accuracy, per-state precision / recall / F1, and a full 9×9 confusion matrix comparing the GPU-decoded path against the GTF ground truth.

---

## Data Preparation (custom dataset)

To run on your own sequences you need:

1. **A multi-FASTA file** — one record per transcript, strand-corrected, with UCSC-style headers:
   ```
   >hg38_knownGene_ENST00000852538.1 range=chr22:19969892-20016780 strand=-
   ATGCGT...
   ```
   These can be generated with UCSC's `getFastaFromBed` or `hgGetAnn`.

2. **A GTF annotation file** — standard format with `exon`, `CDS`, `start_codon`, and `stop_codon` features, using 1-based genomic coordinates and matching `transcript_id` attributes.

Then run steps 2–3 from the Quick Start above.
