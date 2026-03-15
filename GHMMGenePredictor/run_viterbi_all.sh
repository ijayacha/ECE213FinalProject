#!/usr/bin/env bash
# =============================================================================
# run_viterbi_all.sh
# Run the GPU Viterbi decoder on every transcript listed in manifest.tsv.
#
# Usage:
#   ./run_viterbi_all.sh [options]
#
# Options:
#   -f  FASTA        Multi-FASTA input file          (default: Test_Sequence.fa)
#   -m  MANIFEST     manifest.tsv from extract_all_labels.py
#                                                     (default: labels/manifest.tsv)
#   -v  VITERBI      Path to the compiled viterbi binary
#                                                     (default: ./viterbi)
#   -o  OUTDIR       Directory for decoded path files (default: viterbi_output/)
#   -l  LOGFILE      Per-run log file                 (default: viterbi_run.log)
#   -j  JOBS         Parallel jobs (transcripts at once; each still uses 1 GPU)
#                                                     (default: 1)
#   -s  SUMMARY      Summary CSV written after all runs
#                                                     (default: viterbi_summary.csv)
#   --skip-existing  Skip transcripts whose output file already exists
#   --dry-run        Print commands without executing them
#
# Output per transcript:
#   <outdir>/<TRANSCRIPT_ID>_viterbi_path.txt   decoded state path
#
# Summary CSV columns:
#   transcript_id, fasta_length, gpu_ms, cpu_ms, speedup, match_cpu,
#   accuracy_pct, label_file, path_file, status
# =============================================================================

set -euo pipefail

cd /home/ijayacha/ECE213FinalProject/GHMMGenePredictor   # make sure we're in the right directory, update path to your working directory

# Compile
nvcc -O3 -arch=sm_80 viterbi.cu -o viterbi -lm

# ── Defaults ──────────────────────────────────────────────────────────────────
FASTA="Test_Sequence.fa"
MANIFEST="labels/manifest.tsv"
VITERBI="./viterbi"
OUTDIR="viterbi_output"
LOGFILE="viterbi_run.log"
JOBS=1
SUMMARY="viterbi_summary.csv"
SKIP_EXISTING=0
DRY_RUN=0

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,2\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f) FASTA="$2";        shift 2 ;;
        -m) MANIFEST="$2";     shift 2 ;;
        -v) VITERBI="$2";      shift 2 ;;
        -o) OUTDIR="$2";       shift 2 ;;
        -l) LOGFILE="$2";      shift 2 ;;
        -j) JOBS="$2";         shift 2 ;;
        -s) SUMMARY="$2";      shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --dry-run)       DRY_RUN=1;       shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
[[ -f "$FASTA"    ]] || { echo "ERROR: FASTA file not found: $FASTA";    exit 1; }
[[ -f "$MANIFEST" ]] || { echo "ERROR: Manifest not found: $MANIFEST";   exit 1; }
[[ -f "$VITERBI"  ]] || { echo "ERROR: Viterbi binary not found: $VITERBI. Compile with:"; \
                           echo "       nvcc -O3 -arch=sm_75 viterbi.cu -o viterbi -lm"; exit 1; }

mkdir -p "$OUTDIR"

# ── Count transcripts ─────────────────────────────────────────────────────────
TOTAL=$(grep -v '^#' "$MANIFEST" | wc -l)
echo "================================================"
echo " Batch Viterbi Gene Prediction"
echo "================================================"
echo " FASTA    : $FASTA"
echo " Manifest : $MANIFEST  ($TOTAL transcripts)"
echo " Output   : $OUTDIR/"
echo " Log      : $LOGFILE"
echo " Jobs     : $JOBS"
[[ $SKIP_EXISTING -eq 1 ]] && echo " Mode     : skip existing outputs"
[[ $DRY_RUN       -eq 1 ]] && echo " Mode     : DRY RUN (no execution)"
echo "================================================"
echo ""

# ── Write summary CSV header ──────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
    echo "transcript_id,fasta_length,gpu_ms,cpu_ms,speedup,match_cpu,accuracy_pct,label_file,path_file,status" \
        > "$SUMMARY"
fi

# ── Initialise log ────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
    {
        echo "# Viterbi batch run — $(date)"
        echo "# FASTA=$FASTA  MANIFEST=$MANIFEST  OUTDIR=$OUTDIR  JOBS=$JOBS"
    } > "$LOGFILE"
fi

# ── Worker function ───────────────────────────────────────────────────────────
# Called once per transcript. Parses viterbi stdout to extract timing and
# accuracy, then appends a row to the summary CSV.
#
# Arguments: $1=transcript_id  $2=label_file  $3=fasta_length
run_one() {
    local TID="$1"
    local LABEL_FILE="$2"
    local FASTA_LEN="$3"

    local PATH_FILE="${OUTDIR}/${TID}_viterbi_path.txt"

    # -- Skip if output exists and --skip-existing is set --
    if [[ $SKIP_EXISTING -eq 1 && -f "$PATH_FILE" ]]; then
        echo "  [SKIP] $TID"
        return 0
    fi

    # -- Build command --
    local CMD="$VITERBI $FASTA $TID $LABEL_FILE $PATH_FILE"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY]  $CMD"
        return 0
    fi

    # -- Run and capture stdout --
    local OUTPUT
    local STATUS="OK"
    if ! OUTPUT=$("$VITERBI" "$FASTA" "$TID" "$LABEL_FILE" "$PATH_FILE" 2>&1); then
        STATUS="FAILED"
    fi

    # -- Append raw output to log --
    {
        echo ""
        echo "### $TID  $(date +%T)"
        echo "$OUTPUT"
    } >> "$LOGFILE"

    # -- Parse timing and accuracy from stdout --
    local GPU_MS CPU_MS SPEEDUP MATCH_CPU ACC_PCT
    GPU_MS=$(   echo "$OUTPUT" | grep -oP 'Kernel time\s*=\s*\K[0-9.]+' || echo "N/A")
    CPU_MS=$(   echo "$OUTPUT" | grep -oP 'Time\s*=\s*\K[0-9.]+(?= ms)' | head -1 || echo "N/A")
    SPEEDUP=$(  echo "$OUTPUT" | grep -oP 'Speedup over CPU\s*=\s*\K[0-9.]+' || echo "N/A")
    MATCH_CPU=$(echo "$OUTPUT" | grep -oP '(?<=CORRECTNESS\] )[✓✗]' || echo "N/A")
    ACC_PCT=$(  echo "$OUTPUT" | grep -oP 'Overall per-base accuracy:\s*\K[0-9.]+' || echo "N/A")

    # -- Append to summary CSV (atomic via temp file trick) --
    local ROW="${TID},${FASTA_LEN},${GPU_MS},${CPU_MS},${SPEEDUP},${MATCH_CPU},${ACC_PCT},${LABEL_FILE},${PATH_FILE},${STATUS}"
    echo "$ROW" >> "$SUMMARY"

    # -- Print one-liner progress to stdout --
    printf "  %-45s  T=%-7s  GPU=%6s ms  Acc=%s%%  %s\n" \
        "$TID" "$FASTA_LEN" "$GPU_MS" "$ACC_PCT" "$STATUS"
}

export -f run_one
export FASTA VITERBI OUTDIR LOGFILE SUMMARY SKIP_EXISTING DRY_RUN

# ── Main loop ─────────────────────────────────────────────────────────────────
START_TIME=$(date +%s)

if [[ $JOBS -gt 1 ]]; then
    # Parallel mode: use GNU parallel (or xargs -P as fallback)
    if command -v parallel &>/dev/null; then
        grep -v '^#' "$MANIFEST" \
            | awk -F'\t' '{print $1, $3, $2}' \
            | parallel -j "$JOBS" --colsep ' ' run_one {1} {2} {3}
    else
        echo "NOTE: GNU parallel not found, falling back to xargs -P $JOBS"
        grep -v '^#' "$MANIFEST" \
            | awk -F'\t' '{print $1"\t"$3"\t"$2}' \
            | xargs -P "$JOBS" -I{} bash -c '
                IFS=$'"'"'\t'"'"' read -r TID LABEL_FILE FASTA_LEN <<< "{}"
                run_one "$TID" "$LABEL_FILE" "$FASTA_LEN"
            '
    fi
else
    # Serial mode
    DONE=0
    while IFS=$'\t' read -r TID FASTA_LEN LABEL_FILE _VALID _WARN; do
        [[ "$TID" == \#* ]] && continue
        DONE=$(( DONE + 1 ))
        printf "[%4d/%4d] " "$DONE" "$TOTAL"
        run_one "$TID" "$LABEL_FILE" "$FASTA_LEN"
    done < "$MANIFEST"
fi

# ── Final stats ───────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

if [[ $DRY_RUN -eq 0 ]]; then
    SUCCEEDED=$(grep -v '^transcript_id' "$SUMMARY" | grep -c ',OK$'   || true)
    FAILED=$(   grep -v '^transcript_id' "$SUMMARY" | grep -c ',FAILED$' || true)

    echo ""
    echo "================================================"
    echo " Batch complete"
    echo "================================================"
    echo " Total      : $TOTAL"
    echo " Succeeded  : $SUCCEEDED"
    echo " Failed     : $FAILED"
    printf " Wall time  : %dm %ds\n" $(( ELAPSED/60 )) $(( ELAPSED%60 ))
    echo " Summary    : $SUMMARY"
    echo " Log        : $LOGFILE"
    echo "================================================"
fi
