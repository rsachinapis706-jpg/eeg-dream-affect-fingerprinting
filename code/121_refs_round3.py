"""
Reference changes for round 3 (Access-2026-50879, Reviewer 2, comment 11).

1. Removes the citations Reviewer 2 found tangential, with the sentences that
   carried them: Li et al. (OSAMamba, cited only for architectural lineage),
   Hu et al. (EEGDTF), and the two montage-reconstruction papers (Zhuang et
   al.; Lomoio et al.). The generative-model subsection is shortened to the
   one argument it supports: a synthetic sample derived from test data leaks
   as an oversampled one does.
2. Drops those four entries from references.bib, so no uncited entry remains.
3. Makes DOIs visible in the rendered bibliography. IEEEtran.bst (v1.14, the
   version on Overleaf) ignores the doi field, so each entry with a DOI gets
   note = {doi: \\nolinkurl{...}}, printed at the end of the reference in
   IEEE's own form. Entries that already have a note are left alone.
Idempotent.
"""
import io
import re

TEX = r"E:\Dream_Analysis\accesstemp.tex"
BIB = r"E:\Dream_Analysis\references.bib"
DROP = ["li2025osamamba", "hu2026eegdtf", "zhuang2026attention",
        "lomoio2026bidirectional"]

s = io.open(TEX, encoding="utf-8", newline="").read()
edits = [
    (" Li et al. \\cite{li2025osamamba} apply the same class of bidirectional "
     "state-space model to apnoea detection; it is cited for that architectural "
     "lineage, not for a comparison of results.", ""),
    ("\\subsection{Generative and Reconstruction Models for EEG}",
     "\\subsection{Generative Models for EEG}"),
    ("Two strands of this literature bear directly on the analyses here. "
     "Generative models are increasingly used to augment EEG training data, by "
     "diffusion regulated with reinforcement learning \\cite{an2026enhancing} or "
     "by adversarial networks \\cite{liang2026generative}; a synthetic sample "
     "derived from test data leaks in the same way as an oversampled one, so "
     "the placement rule of Section~\\ref{sec3} applies to it unchanged. Hu et "
     "al. \\cite{hu2026eegdtf} parameterise the periodic and aperiodic spectral "
     "components separately; the aperiodic component is one of the feature "
     "families that carries participant identity in Section~\\ref{sec4}. "
     "Montage reconstruction from sparse inputs "
     "\\cite{zhuang2026attention,lomoio2026bidirectional} approaches the "
     "question of the channel-density analysis in "
     "Section~\\ref{sec:replication} from the opposite direction, by asking "
     "whether information absent at low density can be recovered.",
     "Generative models are increasingly used to augment EEG training data, by "
     "diffusion regulated with reinforcement learning \\cite{an2026enhancing} or "
     "by adversarial networks \\cite{liang2026generative}. They bear on this "
     "work at one point only: a synthetic sample derived from test data leaks "
     "in the same way as an oversampled one, so a generative model, like "
     "oversampling, must be fitted within the training fold, and the placement "
     "rule of Section~\\ref{sec3} applies to it unchanged."),
]
for old, new in edits:
    if old in s:
        s = s.replace(old, new)
    elif new:                     # a deletion leaves nothing to find again
        assert new in s, f"neither old nor new text found: {old[:60]}"
for k in DROP:
    assert f"{{{k}}}" not in s and f"{k}," not in s and f",{k}" not in s, k
io.open(TEX, "w", encoding="utf-8", newline="").write(s)

b = io.open(BIB, encoding="utf-8", newline="").read()
entries = re.split(r"(?=^@)", b, flags=re.M)
out, dropped, noted = [], 0, 0
for e in entries:
    m = re.match(r"@\w+\{([^,\s]+),", e)
    if m and m.group(1) in DROP:
        dropped += 1
        continue
    d = re.search(r"^\s*doi\s*=\s*\{([^}]+)\}", e, re.M | re.I)
    if m and d and not re.search(r"^\s*note\s*=", e, re.M | re.I):
        # the note follows the doi; if more fields follow, it keeps the comma
        e = re.sub(r"(^\s*doi\s*=\s*\{[^}]+\})(,?)",
                   lambda x: x.group(1) + ",\n  note={doi: "
                   f"\\nolinkurl{{{d.group(1).strip()}}}}}" + x.group(2), e,
                   count=1, flags=re.M | re.I)
        noted += 1
    out.append(e)
io.open(BIB, "w", encoding="utf-8", newline="").write("".join(out))
print(f"text edited; {dropped} entries dropped; DOI note added to {noted}")
