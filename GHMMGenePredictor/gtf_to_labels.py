#!/usr/bin/env python3
"""
gtf_to_labels.py
~~~~~~~~~~~~~~~~
Convert a UCSC-style multi-FASTA + GTF annotation file into a per-nucleotide
state-label file suitable for use with the Viterbi gene-prediction program.

Usage
-----
    python3 gtf_to_labels.py <input.fa> <annotation.gtf> [options]

Options
    -t, --transcript  TRANSCRIPT_ID   Which transcript to process (default: first record)
    -o, --output      FILE            Output label file (default: <transcript_id>_labels.txt)
    --list                            List all transcript IDs in the FASTA and exit

State encoding (matches viterbi.cu HMM)
    0  Intergenic
    1  Start codon   (ATG, 3 nt)
    2  Exon_0        (coding, reading frame 0)
    3  Exon_1        (coding, reading frame 1)
    4  Exon_2        (coding, reading frame 2)
    5  Splice donor  (first 2 nt of intron = GT)
    6  Intron
    7  Splice acceptor (last 2 nt of intron = AG)
    8  Stop codon    (TAA/TAG/TGA, 3 nt)

Notes
-----
- The FASTA file is expected to contain one record per transcript, already
  strand-corrected (as produced by UCSC's getFastaFromBed --strand or
  hgGetAnn), with headers of the form:
      >hg38_knownGene_<TRANSCRIPT_ID> range=<chrom>:<start>-<end> strand=+/-
- The GTF file uses 1-based genomic coordinates.
- For minus-strand transcripts the coordinate transform is:
      fasta_pos = genome_end - gtf_pos   (0-based)
- For plus-strand transcripts:
      fasta_pos = gtf_pos - genome_start  (0-based)
- Non-coding exonic regions (UTR) are labelled Intergenic (state 0) because
  the 9-state HMM has no explicit UTR state.
- Only the requested transcript's own annotation is used; overlapping features
  from other transcripts are ignored.
"""

import argparse
import re
import sys
from collections import defaultdict, Counter

# ── State indices (must match viterbi.cu) ────────────────────────────────────
ST_INTERGENIC = 0
ST_START      = 1
ST_EXON_0     = 2
ST_EXON_1     = 3
ST_EXON_2     = 4
ST_DONOR      = 5
ST_INTRON     = 6
ST_ACCEPTOR   = 7
ST_STOP       = 8

STATE_NAMES = [
    'Intergenic', 'Start', 'Exon_0(f0)', 'Exon_1(f1)', 'Exon_2(f2)',
    'Donor', 'Intron', 'Acceptor', 'Stop'
]

# Splice-site widths (nt)
DONOR_LEN    = 2   # GT dinucleotide
ACCEPTOR_LEN = 2   # AG dinucleotide


# ─────────────────────────────────────────────────────────────────────────────
# FASTA parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_fasta_header(header_line):
    """
    Parse a UCSC-style FASTA header and return a metadata dict.
    Expected format:
        >hg38_knownGene_ENST00000852538.1 range=chr22:19969892-20016780 strand=- ...
    Returns dict with keys: id, chrom, start (1-based), end (1-based), strand.
    """
    m_id     = re.search(r'>hg38_knownGene_(\S+)', header_line)
    m_range  = re.search(r'range=(\w+):(\d+)-(\d+)', header_line)
    m_strand = re.search(r'strand=([+-])', header_line)

    if not m_id or not m_range or not m_strand:
        # Fallback: use the whole thing after '>'
        tid = header_line.lstrip('>').split()[0]
        return {'id': tid, 'chrom': None, 'start': 0, 'end': 0, 'strand': '+'}

    return {
        'id':     m_id.group(1),
        'chrom':  m_range.group(1),
        'start':  int(m_range.group(2)),   # 1-based
        'end':    int(m_range.group(3)),   # 1-based
        'strand': m_strand.group(1),
    }


def read_fasta_index(fasta_path):
    """
    Scan the FASTA file and return an ordered list of (transcript_id, meta, byte_offset)
    so individual records can be fetched without loading the whole file into RAM.
    """
    index = []
    with open(fasta_path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(b'>'):
                meta = parse_fasta_header(line.decode())
                index.append((meta['id'], meta))
    return index


def read_fasta_record(fasta_path, target_tid):
    """
    Read a single FASTA record (by transcript ID) and return (sequence_str, meta).
    Returns (None, None) if not found.
    """
    found = False
    seq_parts = []
    meta = None

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if found:
                    break           # finished reading the target record
                m = parse_fasta_header(line)
                if m['id'] == target_tid:
                    found = True
                    meta  = m
            elif found:
                seq_parts.append(line.lower())

    if not found:
        return None, None
    return ''.join(seq_parts), meta


# ─────────────────────────────────────────────────────────────────────────────
# GTF parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gtf(gtf_path, target_tid):
    """
    Parse the GTF file and return all features for *target_tid*.
    Returns a dict:
        {
          'exon':         [(gtf_start, gtf_end), ...],
          'CDS':          [(gtf_start, gtf_end, frame), ...],
          'start_codon':  [(gtf_start, gtf_end), ...],
          'stop_codon':   [(gtf_start, gtf_end), ...],
        }
    All coordinates are 1-based inclusive genomic.
    """
    features = defaultdict(list)

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue

            feat_type = parts[2]
            if feat_type not in ('exon', 'CDS', 'start_codon', 'stop_codon'):
                continue

            # Extract transcript_id from attribute field
            m = re.search(r'transcript_id "([^"]+)"', parts[8])
            if not m:
                continue
            tid = m.group(1)
            if tid != target_tid:
                continue

            gtf_s    = int(parts[3])
            gtf_e    = int(parts[4])
            frame_s  = parts[7]
            frame    = int(frame_s) if frame_s in ('0', '1', '2') else 0

            if feat_type == 'CDS':
                features['CDS'].append((gtf_s, gtf_e, frame))
            else:
                features[feat_type].append((gtf_s, gtf_e))

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Label builder
# ─────────────────────────────────────────────────────────────────────────────

def build_labels(seq, meta, features):
    """
    Produce a per-nucleotide state label array (list of ints, length = len(seq)).

    Labelling priority (higher overwrites lower):
        1 = transcript body → Intron
        2 = exon            → Exon_0 placeholder (UTR gets reset to Intergenic later)
        3 = CDS             → Exon_0/1/2 based on reading frame propagation
        4 = splice donor/acceptor
        5 = start_codon / stop_codon
    """
    T      = len(seq)
    strand = meta['strand']
    g_s    = meta['start']   # 1-based genomic start of the FASTA record
    g_e    = meta['end']     # 1-based genomic end

    def to_fasta(gtf_s, gtf_e):
        """
        Convert 1-based inclusive GTF coordinates to 0-based inclusive FASTA coords.
        Returns (fa_start, fa_end) clamped to [0, T-1].
        """
        if strand == '-':
            fa_s = g_e - gtf_e
            fa_e = g_e - gtf_s
        else:
            fa_s = gtf_s - g_s
            fa_e = gtf_e - g_s
        return max(0, fa_s), min(T - 1, fa_e)

    labels   = [ST_INTERGENIC] * T
    priority = [0] * T

    exons  = features.get('exon',        [])
    cdss   = features.get('CDS',         [])
    starts = features.get('start_codon', [])
    stops  = features.get('stop_codon',  [])

    if not exons:
        print(f"  Warning: no exon features found — all bases labelled Intergenic.",
              file=sys.stderr)
        return labels

    # Sort exons into FASTA order (ascending FASTA coordinate)
    if strand == '-':
        exons_fa_order = sorted(exons, key=lambda x: g_e - x[1])
    else:
        exons_fa_order = sorted(exons, key=lambda x: x[0])

    # ── Step 1: transcript body → Intron ──────────────────────────────────────
    tx_fa_s = to_fasta(min(e[0] for e in exons), max(e[1] for e in exons))
    for i in range(tx_fa_s[0], tx_fa_s[1] + 1):
        labels[i]   = ST_INTRON
        priority[i] = 1

    # ── Step 2: exonic positions → placeholder (Exon_0) ──────────────────────
    for (gs, ge) in exons:
        fa_s, fa_e = to_fasta(gs, ge)
        for i in range(fa_s, fa_e + 1):
            if priority[i] < 2:
                labels[i]   = ST_EXON_0   # will be refined or reset below
                priority[i] = 2

    # ── Step 3: CDS with reading-frame propagation ────────────────────────────
    # Sort CDS regions in FASTA 5'→3' order so the running frame is correct.
    if strand == '-':
        cds_sorted = sorted(cdss, key=lambda x: g_e - x[1])
    else:
        cds_sorted = sorted(cdss, key=lambda x: x[0])

    for (gs, ge, frame) in cds_sorted:
        fa_s, fa_e = to_fasta(gs, ge)
        for i in range(fa_e - fa_s + 1):
            fi        = fa_s + i
            codon_pos = (i + frame) % 3
            state     = (ST_EXON_0, ST_EXON_1, ST_EXON_2)[codon_pos]
            if priority[fi] <= 2:
                labels[fi]   = state
                priority[fi] = 3

    # ── Step 4: Non-coding exon positions (UTR) → Intergenic ─────────────────
    # Positions inside an exon but not inside any CDS have priority == 2;
    # they are UTR and the 9-state HMM has no UTR state, so reset to Intergenic.
    for i in range(T):
        if priority[i] == 2:
            labels[i]   = ST_INTERGENIC
            priority[i] = 0

    # ── Step 5: Splice donor / acceptor ──────────────────────────────────────
    for k in range(len(exons_fa_order) - 1):
        ex_curr = exons_fa_order[k]
        ex_next = exons_fa_order[k + 1]

        curr_fa_s, curr_fa_e = to_fasta(*ex_curr)
        next_fa_s, _         = to_fasta(*ex_next)

        # Donor: first DONOR_LEN nt of intron immediately after ex_curr
        for i in range(curr_fa_e + 1,
                       min(curr_fa_e + 1 + DONOR_LEN, T)):
            if labels[i] == ST_INTRON:
                labels[i]   = ST_DONOR
                priority[i] = 4

        # Acceptor: last ACCEPTOR_LEN nt of intron immediately before ex_next
        for i in range(max(0, next_fa_s - ACCEPTOR_LEN), next_fa_s):
            if labels[i] == ST_INTRON:
                labels[i]   = ST_ACCEPTOR
                priority[i] = 4

    # ── Step 6: start_codon ──────────────────────────────────────────────────
    for (gs, ge) in starts:
        fa_s, fa_e = to_fasta(gs, ge)
        for i in range(fa_s, fa_e + 1):
            labels[i]   = ST_START
            priority[i] = 5

    # ── Step 7: stop_codon ───────────────────────────────────────────────────
    for (gs, ge) in stops:
        fa_s, fa_e = to_fasta(gs, ge)
        for i in range(fa_s, fa_e + 1):
            labels[i]   = ST_STOP
            priority[i] = 5

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_labels(seq, labels):
    """
    Run sanity checks and print a summary.  Returns True if all checks pass.
    """
    T = len(seq)
    ok = True

    # Check start codons
    start_pos = [i for i, l in enumerate(labels) if l == ST_START]
    if start_pos:
        codon = seq[start_pos[0]: start_pos[-1] + 1]
        if codon.lower() != 'atg':
            print(f"  [WARN] Start codon sequence = '{codon}' (expected 'atg')",
                  file=sys.stderr)
            ok = False
        else:
            print(f"  [OK] Start codon @ pos {start_pos[0]}: '{codon}'")

    # Check stop codons
    stop_pos = [i for i, l in enumerate(labels) if l == ST_STOP]
    if stop_pos:
        codon = seq[stop_pos[0]: stop_pos[-1] + 1]
        if codon.lower() not in ('taa', 'tag', 'tga'):
            print(f"  [WARN] Stop codon sequence = '{codon}' (expected taa/tag/tga)",
                  file=sys.stderr)
            ok = False
        else:
            print(f"  [OK] Stop codon  @ pos {stop_pos[0]}: '{codon}'")

    # Check donor dinucleotides (GT)
    donor_pos = [i for i, l in enumerate(labels) if l == ST_DONOR]
    bad_donors = [i for i in donor_pos if seq[i].lower() != 'g' and seq[i].lower() != 't']
    # Donors come in pairs: pos i='g', pos i+1='t'
    donor_starts = donor_pos[::2] if donor_pos else []
    bad_gt = [i for i in donor_starts
              if i + 1 < T and seq[i:i+2].lower() != 'gt']
    if bad_gt:
        print(f"  [WARN] {len(bad_gt)} donor sites do not start with 'gt': "
              f"e.g. pos {bad_gt[0]}: '{seq[bad_gt[0]:bad_gt[0]+2]}'",
              file=sys.stderr)
    elif donor_starts:
        print(f"  [OK] All {len(donor_starts)} donor sites begin with 'gt'")

    # Check acceptor dinucleotides (AG)
    accept_pos = [i for i, l in enumerate(labels) if l == ST_ACCEPTOR]
    accept_starts = accept_pos[::2] if accept_pos else []
    bad_ag = [i for i in accept_starts
              if i + 1 < T and seq[i:i+2].lower() != 'ag']
    if bad_ag:
        print(f"  [WARN] {len(bad_ag)} acceptor sites do not end with 'ag': "
              f"e.g. pos {bad_ag[0]}: '{seq[bad_ag[0]:bad_ag[0]+2]}'",
              file=sys.stderr)
    elif accept_starts:
        print(f"  [OK] All {len(accept_starts)} acceptor sites end with 'ag'")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_labels(path, labels, tid, meta):
    with open(path, 'w') as f:
        f.write('# Ground-truth state labels\n')
        f.write(f'# Transcript: {tid}\n')
        f.write(f'# Genomic range: {meta["chrom"]}:{meta["start"]}-{meta["end"]}'
                f'  strand={meta["strand"]}\n')
        f.write(f'# T={len(labels)}\n')
        f.write('# State: 0=Intergenic 1=Start 2=Exon_0 3=Exon_1 4=Exon_2 '
                '5=Donor 6=Intron 7=Acceptor 8=Stop\n')
        for lbl in labels:
            f.write(f'{lbl}\n')
    print(f"[OUTPUT] Labels written to '{path}'  ({len(labels)} positions)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert FASTA + GTF annotation to per-nucleotide HMM state labels.')
    parser.add_argument('fasta',  help='Multi-FASTA file (one record per transcript)')
    parser.add_argument('gtf',    help='GTF annotation file')
    parser.add_argument('-t', '--transcript', default=None,
                        help='Transcript ID to process (default: first record in FASTA)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output label file (default: <transcript_id>_labels.txt)')
    parser.add_argument('--list', action='store_true',
                        help='List all transcript IDs in the FASTA and exit')
    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        print("Transcript IDs in FASTA:")
        index = read_fasta_index(args.fasta)
        for tid, meta in index:
            print(f"  {tid:40s}  {meta['chrom']}:{meta['start']}-{meta['end']}"
                  f"  strand={meta['strand']}")
        return

    # ── Determine target transcript ──
    if args.transcript:
        target_tid = args.transcript
    else:
        index = read_fasta_index(args.fasta)
        if not index:
            print("Error: no records found in FASTA", file=sys.stderr)
            sys.exit(1)
        target_tid = index[0][0]
        print(f"[INFO] No transcript specified, using first record: {target_tid}")

    # ── Read FASTA record ──
    print(f"[FASTA] Reading record '{target_tid}'...")
    seq, meta = read_fasta_record(args.fasta, target_tid)
    if seq is None:
        print(f"Error: transcript '{target_tid}' not found in '{args.fasta}'",
              file=sys.stderr)
        sys.exit(1)
    print(f"[FASTA] Length = {len(seq)} nt  "
          f"({meta['chrom']}:{meta['start']}-{meta['end']} strand={meta['strand']})")

    # ── Parse GTF ──
    print(f"[GTF]   Parsing features for '{target_tid}'...")
    features = parse_gtf(args.gtf, target_tid)
    counts = {k: len(v) for k, v in features.items()}
    print(f"[GTF]   Found: {counts}")

    # ── Build labels ──
    print("[LABEL] Building per-nucleotide labels...")
    labels = build_labels(seq, meta, features)

    # ── Print distribution ──
    c = Counter(labels)
    print("\n  Label distribution:")
    for s in range(9):
        pct = 100.0 * c[s] / len(labels)
        print(f"    {s}  {STATE_NAMES[s]:12s}: {c[s]:7d}  ({pct:.2f}%)")

    # ── Validate ──
    print("\n  Validation:")
    validate_labels(seq, labels)

    # ── Write output ──
    if args.output:
        out_path = args.output
    else:
        out_path = f"{target_tid}_labels.txt"

    write_labels(out_path, labels, target_tid, meta)


if __name__ == '__main__':
    main()
