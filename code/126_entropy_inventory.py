"""
Inventory of the 240 re-implemented entropy features (round 3, Reviewer 2,
comment 9), and the corrected description in the manuscript.

The features are those of 08_replicate_published.py (epoch_nonlinear and
entropy_block). Their order, by which the columns f0..f239 were written, is:
for each of the 6 channels, the 6 coefficient sets of a five-level db4
discrete wavelet transform (A5, D5, D4, D3, D2, D1, in pywt.wavedec order),
then 4 intrinsic mode functions of empirical mode decomposition; for each of
these 10 components, 4 measures: sample entropy, permutation entropy,
approximate entropy, Higuchi fractal dimension. 6 x 10 x 4 = 240.

The previous description listed three measures and omitted the Higuchi
fractal dimension, the truncation to 1,024 samples and the zero fill; it is
replaced here. Writes out/supplement/entropy_feature_inventory.csv.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "08_replicate_published.py")
OUT = os.path.join(HERE, "out", "supplement", "entropy_feature_inventory.csv")
TEX = r"E:\Dream_Analysis\accesstemp.tex"

# the inventory is derived from constants in the extraction script itself
code = io.open(SRC, encoding="utf-8").read()
level = int(re.search(r"^DWT_LEVEL = (\d+)", code, re.M).group(1))
n_imf = int(re.search(r"^N_IMF = (\d+)", code, re.M).group(1))
wavelet = re.search(r'^WAVELET = "(\w+)"', code, re.M).group(1)
assert "ant.higuchi_fd" in code and "x[:1024]" in code and "sig[::2]" in code
MEASURES = ["sample entropy (m=2, r=0.2 SD)", "permutation entropy (order 3, "
            "normalised)", "approximate entropy (m=2, r=0.2 SD)",
            "Higuchi fractal dimension (kmax=10)"]
comps = ([("DWT", f"A{level}")] + [("DWT", f"D{k}") for k in range(level, 0, -1)]
         + [("EMD", f"IMF{k + 1}") for k in range(n_imf)])
rows, i = [], 0
for ch in range(6):
    for dec, comp in comps:
        for meas in MEASURES:
            rows.append(f"f{i},{ch + 1},{dec},{comp},\"{meas}\"")
            i += 1
assert i == 240, i
os.makedirs(os.path.dirname(OUT), exist_ok=True)
io.open(OUT, "w", encoding="utf-8").write(
    "column,channel,decomposition,component,measure\n" + "\n".join(rows) + "\n")

s = io.open(TEX, encoding="utf-8", newline="").read()
old = ("The re-implemented entropy family of the published DEED studies "
       "decomposes each channel by a five-level \\texttt{db4} discrete wavelet "
       "transform and by empirical mode decomposition into four intrinsic mode "
       "functions (computed at 100\\,Hz), and computes sample entropy (embedding "
       "dimension 2, tolerance 0.2 of the standard deviation), permutation "
       "entropy (order 3, normalised) and approximate entropy (dimension 2) on "
       "each standardised component, 240 features in all.")
new = ("The re-implemented entropy family of the published DEED studies, a "
       "sensitivity experiment rather than a reproduction of their pipelines, "
       "band-passes each of the six channels at 1--45\\,Hz and decomposes it "
       f"into ten components: the six coefficient sets of a {level}-level "
       f"\\texttt{{{wavelet}}} discrete wavelet transform (A5 and D5--D1) and "
       f"{n_imf} intrinsic mode functions of empirical mode decomposition, "
       "computed at 100\\,Hz. Each standardised component, truncated to its "
       "first 1{,}024 samples, gives four measures: sample entropy and "
       "approximate entropy (embedding dimension 2, tolerance 0.2 of the "
       "standard deviation), permutation entropy (order 3, normalised) and "
       "Higuchi fractal dimension. A component shorter than 20 samples, a "
       "constant component or a failed estimate is set to zero. Six channels "
       "$\\times$ ten components $\\times$ four measures give the 240 "
       "features; the inventory is in the supplementary material.")
if old in s:
    s = s.replace(old, new)
assert new in s
io.open(TEX, "w", encoding="utf-8", newline="").write(s)
print(f"inventory written ({i} features); manuscript description replaced")
