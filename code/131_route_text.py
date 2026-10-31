"""
Route decomposition stated as protocol contrasts (round 3, Reviewer 2,
comment 4), with the paired intervals of 124_route_contrasts.py.

The differences window-shuffled minus recording-grouped and recording-grouped
minus participant-separated are contrasts between evaluation protocols:
changing the partition also changes the training data and the generalisation
problem, so they are consistent with recognition of the recording and of the
participant but do not identify those contributions separately. On Tononi the
second contrast is negative. The identity rule and the forest reach similar
balanced accuracy but predict the same class on only part of the windows.

Every number is read from out/route_contrasts.csv. Table 13's Tononi
contrasts were differences of rounded means (0.2653, 0.2772); the exact
differences are used. Idempotent.
"""
import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = r"E:\Dream_Analysis\accesstemp.tex"
rc = pd.read_csv(os.path.join(HERE, "out", "route_contrasts.csv"))


def get(ds, key):
    r = rc[(rc.dataset == ds) & rc.contrast.str.startswith(key)].iloc[0]
    return r


def f(x, signed=False):
    return f"{x:+.4f}".replace("+", "$+$").replace("-", "$-$") if signed \
        else f"{x:.4f}".replace("-", "$-$")


def ci(r):
    lo, hi = r.lo, r.hi
    fmt = lambda v: (f"$-${abs(v):.4f}" if v < 0 else f"{v:.4f}")
    return f"{fmt(lo)} to {fmt(hi)}"


D_rec, D_par, D_tot = (get("DEED", "window-shuffled minus recording"),
                       get("DEED", "recording-grouped minus"),
                       get("DEED", "window-shuffled minus participant"))
D_fr, D_ag = (get("DEED", "forest minus identity"), get("DEED", "agreement"))
T_rec, T_par, T_tot = (get("Tononi", "window-shuffled minus recording"),
                       get("Tononi", "recording-grouped minus"),
                       get("Tononi", "window-shuffled minus participant"))
pct = lambda v: f"{100 * v:.0f}\\%"
assert T_par.estimate < 0, "the Tononi participant contrast is no longer negative"

E = [
    # contribution list
    ("\\item \\textbf{Identification and quantification of the underlying "
     "mechanism.} Window-level partitioning lets a classifier recognise where a "
     "window came from rather than what it contains, by two routes. Splitting a "
     "recording's windows across the partition accounts for 0.1374 of the "
     "0.1685 by which it raises balanced accuracy on DEED; sharing participants "
     "across the partition accounts for the remaining 0.0311, through "
     "participant identity, decodable at 0.8040 against a chance level of "
     "0.0294, combined with a participant--label association. A rule that uses "
     "no EEG and knows only who the participant is matches the window-shuffled "
     "classifier, 0.4915 against 0.4909 balanced accuracy, and falls to chance, "
     "0.3314, when it cannot draw on the held-out recording.",
     "\\item \\textbf{Protocol contrasts that locate the window-level "
     "advantage.} Window-level partitioning raises balanced accuracy on DEED by "
     f"{f(D_tot.estimate)} ({ci(D_tot)}). Keeping each recording's windows "
     f"together removes {f(D_rec.estimate)} ({ci(D_rec)}) of it, and separating "
     f"participants as well a further {f(D_par.estimate)} ({ci(D_par)}). These "
     "are differences between evaluation protocols, consistent with recognition "
     "of the recording and of the participant rather than separately identified "
     "contributions of each. Participant identity is decodable at 0.8040 "
     "against a chance level of 0.0294. A rule that uses no EEG and knows only "
     "who the participant is reaches a balanced accuracy similar to that of "
     "the window-shuffled classifier, 0.4915 against 0.4909, although the two "
     f"predict the same class on only {pct(D_ag.estimate)} of windows, and it "
     "falls to chance, 0.3314, when it cannot draw on the held-out recording."),
    # Section IV, mechanism
    ("Of the 0.1685 by which shuffling windows raises balanced accuracy over "
     "participant separation, 0.1374 is removed by keeping the windows of each "
     "recording together, and the remaining 0.0311 by separating participants "
     "as well. Each step corresponds to a route by which a classifier can score "
     "above chance by recognising where a window came from rather than what it "
     "contains.",
     "Of the 0.1685 by which shuffling windows raises balanced accuracy over "
     f"participant separation, 0.1374 (paired 95\\% participant-bootstrap "
     f"interval {ci(D_rec)}) is removed by keeping the windows of each "
     f"recording together, and the remaining 0.0311 ({ci(D_par)}) by "
     "separating participants as well. These are contrasts between evaluation "
     "protocols, not isolated mechanisms: changing the partition also changes "
     "the training data and what the model must generalise to. They are "
     "consistent with a classifier scoring above chance by recognising where a "
     "window came from rather than what it contains, a reading the identity "
     "rule below tests directly, and they need not both be positive: on the "
     f"Tononi corpus the second is {f(T_par.estimate)} ({ci(T_par)}). The "
     "intervals resample participants with predictions held fixed, over the "
     "ten fold assignments."),
    ("On DEED the recording route carries 0.1374 of the 0.1685; on Tononi it "
     "carries all of it.",
     "On DEED the recording contrast is 0.1374 of the 0.1685; on Tononi it is "
     f"{f(T_rec.estimate)} of {f(T_tot.estimate)}, the participant contrast "
     f"being {f(T_par.estimate)} ({ci(T_par)})."),
    ("it bears on the participant route only, which on DEED accounts for "
     "0.0311 of the 0.1685 by which window-level partitioning raises balanced "
     "accuracy.",
     "it bears only on the participant contrast, which on DEED is 0.0311 "
     f"({ci(D_par)}) of the 0.1685 by which window-level partitioning raises "
     "balanced accuracy."),
    # Section VII, third finding
    ("\\emph{Third, window-level partitioning is exploited by recognising the "
     "source of a window, and this is not specific to EEG.} On DEED, splitting "
     "recordings across the partition accounts for 0.1374 of the 0.1685 by "
     "which window-level partitioning raises balanced accuracy, and sharing "
     "participants for the remaining 0.0311. A rule that uses no EEG and knows "
     "only the participant matches the window-shuffled classifier, 0.4915 "
     "against 0.4909 balanced accuracy,",
     "\\emph{Third, the window-level advantage is consistent with recognising "
     "the source of a window, and this is not specific to EEG.} On DEED, "
     "keeping each recording's windows together removes 0.1374 of the 0.1685 "
     "by which window-level partitioning raises balanced accuracy, and "
     "separating participants a further 0.0311; these are protocol contrasts, "
     "not separately identified mechanisms. A rule that uses no EEG and knows "
     "only the participant reaches a balanced accuracy similar to that of the "
     "window-shuffled classifier, 0.4915 against 0.4909, while predicting the "
     f"same class on only {pct(D_ag.estimate)} of windows,"),
    ("where it is larger, 0.2653 against 0.1685 in balanced accuracy, and runs "
     "entirely through the recording route.",
     f"where it is larger, {f(T_tot.estimate)} against 0.1685 in balanced "
     "accuracy, and appears entirely at the step that keeps recordings "
     "together."),
    ("where window-level partitioning raises balanced accuracy by 0.2653 "
     "against DEED's 0.1685, entirely through the recording route.",
     "where window-level partitioning raises balanced accuracy by "
     f"{f(T_tot.estimate)} against DEED's 0.1685, all of it removed by keeping "
     "recordings together."),
    ("The inflation, 0.2653 in balanced accuracy, exceeds DEED's 0.1685,",
     f"The inflation, {f(T_tot.estimate)} in balanced accuracy, exceeds "
     "DEED's 0.1685,"),
    ("The observed inflation was larger, 0.2653 against 0.1685 in balanced "
     "accuracy.",
     f"The observed inflation was larger, {f(T_tot.estimate)} against 0.1685 "
     "in balanced accuracy."),
    ("Section~\\ref{sec:mechanism} separates a recording route, which requires "
     "only that a recording contribute several windows, from a participant "
     "route, which requires an association between participant and label.",
     "Section~\\ref{sec:mechanism} distinguishes the contrast obtained by "
     "keeping recordings together, which needs only that a recording "
     "contribute several windows, from the contrast obtained by separating "
     "participants, which needs an association between participant and "
     "label."),
    ("and on Tononi it runs entirely through the recording route of "
     "Section~\\ref{sec:mechanism}.",
     "and on Tononi all of it is removed by keeping recordings together "
     "(Section~\\ref{sec:mechanism}), the participant contrast being "
     f"{f(T_par.estimate)} ({ci(T_par)})."),
    # Table 13
    ("& $\\mathbf{+0.1685}$ & $\\mathbf{+0.2653}$ \\\\",
     f"& $\\mathbf{{+0.1685}}$ & $\\mathbf{{+{T_tot.estimate:.4f}}}$ \\\\"),
    ("\\quad removed by grouping recordings & 0.1374 & 0.2772 \\\\",
     f"\\quad removed by grouping recordings & 0.1374 & {T_rec.estimate:.4f} \\\\\n"
     f"\\quad removed by also separating participants & 0.0311 & "
     f"$-${abs(T_par.estimate):.4f} \\\\"),
]

s = io.open(TEX, encoding="utf-8", newline="").read()
for old, new in E:
    if old in s:
        assert s.count(old) == 1, old[:60]
        s = s.replace(old, new)
    assert new in s, f"not applied: {new[:70]}"
io.open(TEX, "w", encoding="utf-8", newline="").write(s)
print(f"{len(E)} passages rewritten as protocol contrasts")
