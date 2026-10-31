# Importing MVSA-Single and MVSA-Multiple

The datasets cannot be downloaded automatically (they are distributed by the authors on request / via
their project page), so import is manual. **Do not edit the raw files; the pipeline only reads them.**

1. Obtain MVSA-Single and/or MVSA-Multiple from the original authors (Niu et al., MMM 2016, Beihang/Xi'an
   Jiaotong University) and check their licence/terms — academic use only; do not redistribute.
2. Place each dataset so it looks like:

       data/raw/MVSA_Single/labelResultAll.txt
       data/raw/MVSA_Single/data/<ID>.jpg  and  <ID>.txt
       data/raw/MVSA_Multiple/labelResultAll.txt
       data/raw/MVSA_Multiple/data/<ID>.jpg  and  <ID>.txt

3. Validate and preprocess:

       python scripts/import_mvsa.py --variant single   --raw data/raw/MVSA_Single
       python scripts/import_mvsa.py --variant multiple --raw data/raw/MVSA_Multiple

Outputs go to `data/processed/mvsa_<variant>/`: `samples.csv`, `report.json`, `processing_log.jsonl`.

## What the pipeline does
validate files and images -> resolve labels -> detect duplicates -> clean text -> stratified, group-aware split
(images/texts that are identical are forced into one split) -> leakage assertion -> export.

Label rule (standard for MVSA): majority vote per modality over annotators (MVSA-Multiple has three; ties are
dropped); equal text/image labels are kept; neutral + X becomes X; positive vs negative is dropped.
Class ids: negative=0, neutral=1, positive=2.
