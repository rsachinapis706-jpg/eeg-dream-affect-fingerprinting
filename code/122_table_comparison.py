"""
Table 21 (comparison with published work), round 3 (Reviewer 2, comment 8).

* Every reported value names its metric: published accuracies are labelled
  "acc.", this work's 0.3224 "bal. acc.".
* The CBraMod row is labelled as evaluated here, not as a published value,
  and gives both input scalings (microvolts; units of 100 uV, the
  pretraining convention), read from 49's two result files.
* ReliefF placement: the paper describes selection before classification but
  not whether it was fitted before or within cross-validation, so the entry
  reads "placement not stated" instead of "before split".
* \\footnotetext inside a table* does not render in ieeeaccess.cls; the notes
  are set as lettered lines directly below the tables instead.
Idempotent: the table is rebuilt from its label to its \\end{table*}.
"""
import io

import pandas as pd

TEX = r"E:\Dream_Analysis\accesstemp.tex"
OUT = r"E:\Dream GAN\analysis\out"
# 49_cbramod_expanded.py with the windows in microvolts and, with
# CBRAMOD_SCALE=uv100, in units of 100 uV (the pretraining convention)
cx = {k: pd.read_csv(OUT + rf"\cbramod_expanded{k}.csv").set_index("model")
      for k in ("", "_uv100")}
sp = cx[""].loc["spectral_logreg", "balanced"]
assert sp == cx["_uv100"].loc["spectral_logreg", "balanced"]
cb_uv, cb_100 = (cx[k].loc["cbramod", "balanced"] for k in ("", "_uv100"))
s = io.open(TEX, encoding="utf-8", newline="").read()
lab = s.index("\\label{tab:comparison}")
a = s.index("\\footnotesize", lab)
b = s.index("\\end{table*}", lab)
old = s[a:b]
if "\\footnotetext[3]{" in old:          # first run: the original footnote
    fn3 = old[old.index("\\footnotetext[3]{") + len("\\footnotetext[3]{"):]
    depth, i = 1, 0
    while depth:
        depth += {"{": 1, "}": -1}.get(fn3[i], 0)
        i += 1
    fn3 = fn3[:i - 1]                   # note 3 text, kept as written
else:                                   # re-run: note c as already set
    k = old.index("\\textsuperscript{c}\\,") + len("\\textsuperscript{c}\\,")
    fn3 = old[k:old.index("\\\\", k)].strip()
R = "\\raggedright\\arraybackslash"
cols = "".join(f">{{{R}}}p{{{w}}}" for w in
               ("2.35cm", "1.5cm", "1.35cm", "1.5cm", "1.75cm", "1.6cm",
                "1.4cm", "1.5cm", "1.15cm"))
sup = lambda x: f"\\textsuperscript{{{x}}}"
new = (
    "\\footnotesize\n"
    f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n"
    "\\toprule\n"
    "\\multicolumn{9}{@{}l}{\\textit{(a) Dream affect on DEED: reported "
    "performance and every leakage-relevant choice}} \\\\\n"
    "\\addlinespace[2pt]\n"
    "Work & Validation unit & Participants separated & Window overlap & "
    "Feature selection & Resampling & Permutation null & Positive control & "
    "Reported (metric) \\\\\n"
    "\\midrule\n"
    "Liu et al.\\ \\cite{liu2022deed}\n"
    f"& Window & Not stated & Not stated & Not stated & Not stated & No & No "
    f"& ---{sup('a')} \\\\\n"
    "\\addlinespace[2pt]\n"
    "Mahmoudi et al.\\ \\cite{mahmoudi2025emotion}\n"
    "& Window & Not stated & Not stated & Not stated & Not stated & No & No "
    "& 93.83\\% acc. \\\\\n"
    "\\addlinespace[2pt]\n"
    "Sayad Mojdehbar et al.\\ \\cite{mojdehbar2026automatic}\n"
    f"& Window & Not stated & Not stated & ReliefF; placement not "
    f"stated{sup('b')} & Not stated & No & No & 92.33\\% acc. \\\\\n"
    "\\midrule\n"
    "\\addlinespace[2pt]\n"
    "\\textbf{This work}\n"
    "& \\textbf{Participant} & \\textbf{Yes} & \\textbf{0\\%, swept to 90\\%} "
    "& \\textbf{None; nested where studied} & \\textbf{None; nested where "
    "studied} & \\textbf{Yes} & \\textbf{Yes} & \\textbf{0.3224} bal.\\ "
    f"acc.{sup('c')} \\\\\n"
    "\\botrule\n"
    "\\end{tabular}\n\n"
    "\\vspace{6pt}\n"
    "\\begin{tabular}{@{}p{3.4cm}p{3.6cm}p{4.4cm}p{2.6cm}@{}}\n"
    "\\toprule\n"
    "\\multicolumn{4}{@{}l}{\\textit{(b) Sleep staging, the positive "
    "control}} \\\\\n"
    "\\addlinespace[2pt]\n"
    "Work & Method & Cohort & Score (metric) \\\\\n"
    "\\midrule\n"
    "Xu et al.\\ \\cite{xu2026electroencephalography}, as published & "
    "State-space sequence model & Sleep-EDF-78 & 0.862 acc."
    f"{sup('d')} \\\\\n"
    "CBraMod \\cite{wang2025cbramod}, evaluated here & Pretrained "
    "transformer, 9{,}000\\,h & Sleep-EDF Expanded, 100 part. & "
    f"{cb_uv:.4f} / {cb_100:.4f} bal.\\ acc.{sup('e')} \\\\\n"
    "\\textbf{This work} & Spectral, single window & Sleep-EDF Expanded, "
    f"100 part. & \\textbf{{{sp:.4f}}} bal.\\ acc.{sup('e')} \\\\\n"
    "\\botrule\n"
    "\\end{tabular}\n\n"
    "\\vspace{4pt}\n"
    "\\parbox{\\textwidth}{\\scriptsize\\raggedright\n"
    f"{sup('a')}\\,The dataset paper reports descriptive analyses and "
    "demonstrates classification, but does not report a headline three-class "
    "accuracy for dream affect.\\\\\n"
    f"{sup('b')}\\,Described as applied to the feature matrix before "
    "classification; whether it was fitted before or within cross-validation "
    "is not stated.\\\\\n"
    f"{sup('c')}\\,{fn3}\\\\\n"
    f"{sup('d')}\\,Five-class accuracy as reported by the authors, not "
    "balanced accuracy, obtained with a sequence model that observes "
    "neighbouring epochs; not commensurable with the two rows below, which "
    "classify a single window in isolation.\\\\\n"
    f"{sup('e')}\\,Both computed in this study on the identical cohort, "
    "windows and folds, so the two rows differ only in the representation; "
    "CBraMod uses the authors' published pretrained weights, fine-tuned here "
    "with the windows in microvolts (first value) and in units of "
    "100\\,$\\mu$V, the convention of its pretraining (second value); the "
    "training schedule was not retuned for either. "
    "Figures obtained on the 8-recording Sleep-EDF 1.0.0 release are not "
    "comparable with this cohort and are not reported.}\n")
s = s[:a] + new + s[b:]
io.open(TEX, "w", encoding="utf-8", newline="").write(s)
print("Table 21 rebuilt; note c kept verbatim:", fn3[:80], "...")
