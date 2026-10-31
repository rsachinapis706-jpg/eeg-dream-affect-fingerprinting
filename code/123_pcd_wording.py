"""
PCD statements kept within the guarantee (round 3, Reviewer 2, comments 1, 5).

* Equation (5) states invariance conditional on the inputs and on the grouping
  of inputs into recordings and participants, which the proof conditions on.
* "Under the alternative its margins are stochastically larger" and "rejecting
  licenses a prediction" are replaced: larger margins under an alternative is
  an assumption about the learner and the score, not a consequence of the
  alternative, and the rule controls how often it predicts under the null, not
  whether a retained prediction is correct.
* Pooling is described as a trade-off between resolution and heterogeneity of
  the reference, not as a saving in model fits: the same null decoders also
  give each input its own reference.
Idempotent.
"""
import io

TEX = r"E:\Dream_Analysis\accesstemp.tex"
s = io.open(TEX, encoding="utf-8", newline="").read()
EDITS = [
    ("H_0:\\ \\pi(\\mathbf{y}) \\stackrel{d}{=} \\mathbf{y} \\quad "
     "\\text{for every } \\pi \\in \\mathcal{G},",
     "H_0:\\ \\big(\\pi(\\mathbf{y}) \\mid \\mathbf{X}, \\mathbf{g}\\big) "
     "\\stackrel{d}{=} \\big(\\mathbf{y} \\mid \\mathbf{X}, \\mathbf{g}\\big) "
     "\\quad \\text{for every } \\pi \\in \\mathcal{G},"),
    ("where $\\mathcal{G}$ is the set of permutations that move labels only "
     "between units at which the label varies and only within a participant. "
     "Under $H_0$ the decoder fitted to the observed labels is one draw from "
     "the same distribution as decoders fitted to permuted labels; under the "
     "alternative its margins are stochastically larger. Rejecting at an input "
     "licenses a prediction there; failing to reject licenses abstention.",
     "where $\\mathbf{X}$ are the inputs, $\\mathbf{g}$ assigns each input to "
     "its recording and participant, and $\\mathcal{G}$ is the set of "
     "permutations that move labels only between units at which the label "
     "varies and only within a participant. Under $H_0$ the decoder fitted to "
     "the observed labels is one draw from the same distribution as decoders "
     "fitted to permuted labels. Under an alternative in which the decoder "
     "learns the association, its margins tend to exceed those of the null "
     "decoders, but that depends on the learner and the score and does not "
     "follow from the alternative alone. The rule predicts where the test "
     "rejects and abstains where it does not; what this controls is how often "
     "it predicts under the null, not whether a retained prediction is "
     "correct."),
    ("Pooling across test inputs rather than forming a separate reference per "
     "input is what makes the procedure feasible at $B$ in the low hundreds: a "
     "per-input reference would need $B$ draws each and would be "
     "correspondingly noisier at the same computational cost.",
     "Pooling across test inputs is a trade-off, not a saving in model fits: "
     "the same $B$ null decoders also give each input its own reference "
     "(Proposition~\\ref{prop:strat} with a single input as the stratum), but "
     "that reference has only $B+1$ values, so its $p$-value moves in steps of "
     "$1/(B+1)$, whereas the pooled reference has $(B+1)\\,n$ values."),
    ("Under the alternative, when the decoder extracts genuine information, "
     "its margins are stochastically larger than those of the null decoders.",
     "When the decoder does extract information about the label, its margins "
     "are expected to exceed those of the null decoders; this is an assumption "
     "about the learner and the margin, not a consequence of the alternative, "
     "so coverage under an alternative is measured below rather than derived."),
]
for old, new in EDITS:
    if old in s:
        s = s.replace(old, new)
    assert new in s, f"edit not applied: {new[:70]}"
io.open(TEX, "w", encoding="utf-8", newline="").write(s)
print(f"{len(EDITS)} PCD statements revised")
