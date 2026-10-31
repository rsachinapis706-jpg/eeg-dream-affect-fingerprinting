"""
Remaining wording corrections for round 3 (Access-2026-50879, Reviewer 2).

* Comment 3: a within-participant permutation of recording labels keeps each
  participant's label composition over recordings, so participant identity is
  exactly as predictive of the recording-level label as in the data; because
  recordings differ in length, the window-level class proportions of a
  participant can change. The statement now names the level.
Idempotent.
"""
import io

TEX = r"E:\Dream_Analysis\accesstemp.tex"
EDITS = [
    ("labels permuted among each participant's own recordings, which keeps "
     "every participant's label composition and so keeps participant identity "
     "exactly as predictive of the label as in the data;",
     "labels permuted among each participant's own recordings, which keeps "
     "every participant's label composition over recordings and so keeps "
     "participant identity exactly as predictive of the recording-level label "
     "as in the data (because recordings differ in length, a participant's "
     "class proportions over windows can change);"),
]
s = io.open(TEX, encoding="utf-8", newline="").read()
for old, new in EDITS:
    if old in s:
        s = s.replace(old, new)
    assert new in s, new[:70]
io.open(TEX, "w", encoding="utf-8", newline="").write(s)
print(f"{len(EDITS)} wording corrections applied")
