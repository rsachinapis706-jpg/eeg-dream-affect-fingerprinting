"""
Writes the foundation-model part of Section IV-J, its diagnostics table, and
the matched-comparison paragraphs of Section V, from the result files
(Reviewer 1, comment 9; Reviewer 2 on foundation-model scope).

Superseded for the table note and the control comparison by
145_cbramod_input.py: the note here first said the input followed the
pretraining convention, which was wrong (corrected 21 Oct 2026; see 145).

Sources: cbramod_diag_summary.csv and cbramod_diag_pretraining_effect.csv
(88), test39_cbramod_expanded.txt values as already quoted (49), the settings
of 14_cbramod_finetune.py and 76_cbramod_diagnostics.py.
"""
import io
import shutil

import pandas as pd

OUT = r"E:\Dream GAN\analysis\out"
TEX = r"E:\Dream_Analysis\accesstemp.tex"


def main():
    s = pd.read_csv(OUT + r"\cbramod_diag_summary.csv").set_index("config")
    pr = pd.read_csv(OUT + r"\cbramod_diag_pretraining_effect.csv"
                     ).set_index("pretrained")
    for c in ("published", "scratch", "revised", "scratch_revised"):
        assert c in s.index and s.loc[c, "folds"] == 34, c
    assert set(pr.index) == {"published", "revised"}

    def ci(c):
        r = s.loc[c]
        return f"{r.pooled_bal:.4f} [{r.bal_lo:.4f}, {r.bal_hi:.4f}]"

    def d(c):
        r = pr.loc[c]
        return f"{r['diff']:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]"

    all_include_chance = all(s.loc[c, "bal_lo"] < 1 / 3 < s.loc[c, "bal_hi"]
                             for c in s.index)
    pub, rev = s.loc["published"], s.loc["revised"]

    rows = []
    for c, sett, pre in (("published", "one participant, lr $10^{-3}$, batch 32, $\\le$8 epochs", "yes"),
                         ("scratch", "as above", "no"),
                         ("revised", "a fifth of participants, lr $10^{-4}$, batch 64, $\\le$15 epochs", "yes"),
                         ("scratch_revised", "as above", "no")):
        r = s.loc[c]
        rows.append(f"{sett} & {pre} & {ci(c)} & {r.pooled_acc:.4f} & "
                    f"{int(r.constant_folds)} & {int(r.chosen_epoch_1)} & "
                    f"{r.best_val_bal_median:.4f} \\\\")
    table = "\n".join([
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Fine-tuning CBraMod on DEED dream valence, leave-one-participant-out on the 6{,}144 windows, under four configurations. Early stopping monitors validation balanced accuracy on held-out training participants -- one, or a fifth of them -- and keeps the best epoch. Pooled balanced accuracy is computed over all held-out windows, with a 95\% participant-bootstrap interval (2{,}000 resamples of the 34 folds). Constant folds: held-out participants for whom every window receives the same class. Epoch-1 folds: folds in which early stopping kept the first epoch.}",
        r"\label{tab:cbramod}",
        r"\small",
        r"\begin{tabular}{@{}p{4.6cm}cccccc@{}}",
        r"\toprule",
        r"Validation and optimisation & Pretrained & Pooled bal.\ acc.\ [95\% CI] & Pooled acc. & Constant folds & Epoch-1 folds & Median best val.\ bal. \\",
        r"\midrule",
    ] + rows + [
        r"\botrule",
        r"\end{tabular}",
        r"\footnotetext{All configurations: AdamW, weight decay 0.01, backbone learning rate one tenth of the head's, class-weighted cross-entropy, gradient norm clipped at 1.0, patience 3 (first two rows) or 4 (last two). Each window enters as six channels of twenty 1\,s patches at 200\,Hz, the backbone's patch size, and the head averages over channels; each window and channel is standardised (median absolute value 0.627), not expressed in units of 100\,$\mu$V as in pretraining (median absolute value 0.040 on these windows); the text reports the second settings in that unit. Pretrained minus randomly initialised: "
        + f"{d('published')} with the first settings, {d('revised')} with the second.}}",
        r"\normalsize",
        r"\end{table*}",
    ])

    ft = (f"The model was also fine-tuned end to end, leave-one-participant-out on the identical 6{{,}}144 windows, and Table~\\ref{{tab:cbramod}} reports training diagnostics under four configurations; confusion matrices, fold-level class distributions and training and validation curves are in the supplementary material. With the settings first used -- early stopping with patience 3 on a single held-out training participant, a learning rate of $10^{{-3}}$ and batches of 32 -- validation balanced accuracy never rose above 0.3333, so early stopping kept the first-epoch model in {int(pub.chosen_epoch_1)} of 34 folds and {int(pub.constant_folds)} folds predicted a single class; pooled balanced accuracy is {ci('published')}. That configuration measures a failure of training rather than the representation. "
          f"With validation on a fifth of the training participants, a learning rate of $10^{{-4}}$, batches of 64 and up to 15 epochs -- the kind of procedure used for the positive control -- {int(rev.constant_folds)} of 34 folds predict a single class, the median chosen epoch is {rev.chosen_epoch_median:.0f}, and pooled balanced accuracy is {ci('revised')}. "
          f"The same architecture without pretraining reaches {ci('scratch')} under the first settings and {ci('scratch_revised')} under the second; pretrained minus randomly initialised is {d('published')} and {d('revised')}. "
          + ("Every interval includes the 0.3333 chance level." if all_include_chance
             else "Not every interval includes chance; see Table~\\ref{tab:cbramod}."))

    concl = ("Two conclusions follow, both limited to this model and this task. First, under every training configuration tried, fine-tuned CBraMod does not exceed chance under participant separation with any confidence, and pretraining on four orders of magnitude more EEG shows no detectable advantage over the same architecture trained from scratch; the frozen embeddings likewise show no within-participant effect. This does not bear on EEG foundation models as a class, on other training procedures or on other tasks. Second, whatever within-participant signal is available to personalisation or test-time adaptation is small: neither representation reaches significance, and the positive spectral estimate is modest and imprecise.")

    t = io.open(TEX, encoding="utf-8", newline="").read()
    shutil.copyfile(TEX, TEX + ".pre_fm")

    def swap(old, new):
        nonlocal t
        assert t.count(old) == 1, (t.count(old), old[:70])
        t = t.replace(old, new)

    i = t.index("The model was also fine-tuned end to end, so the finding does not rest on frozen representations alone.")
    j = t.index("\n", i)
    t = t[:i] + ft + t[j:]
    i = t.index("The corresponding positive control is reported in Section~\\ref{sec5}")
    j = t.index("\n", i)
    t = t[:i] + ("The corresponding positive control is reported in Section~\\ref{sec5}, where the model is compared with the spectral pipeline on the identical cohort, windows and folds: 0.7906 against 0.7799, a difference of $+0.0107$.") + t[j:]
    i = t.index("Two conclusions follow. First, the finding does not depend on handcrafted features")
    j = t.index("\n", i)
    t = t[:i] + concl + "\n\n" + table + t[j:]

    swap("the same 100 participants, the same 34{,}163 windows and the same five participant-separated folds are given to CBraMod and to the 108-feature spectral pipeline, so the two differ only in the representation. CBraMod reaches 0.7906 balanced accuracy and the spectral pipeline 0.7799, a difference of $+0.0107$. On dream affect the same model returns 0.3147.",
         f"the same 100 participants, the same 34{{,}}163 windows and the same five participant-separated folds are given to CBraMod and to the spectral pipeline, whose feature definitions give 36 features from the two EEG channels, so the two differ only in the representation. CBraMod reaches 0.7906 balanced accuracy and the spectral pipeline 0.7799, a difference of $+0.0107$. On dream affect, fine-tuned with a procedure of the same kind, it reaches {ci('revised')} (Section~\\ref{{sec:within}}).")
    i = t.index("Two aspects of that comparison deserve emphasis because they strengthen the argument rather than weakening it.")
    j = t.index("\n", i)
    t = t[:i] + ("One aspect of that comparison requires care. CBraMod received a tuned training schedule -- twenty epochs, early stopping on a fifth of the training participants -- while the spectral pipeline used hyperparameters fixed in advance. The one-point margin therefore compares a pretrained transformer with spectral features under unequal tuning; it does not measure what pretraining contributes, which requires the same architecture without pretraining. That comparison was made on dream affect (Section~\\ref{sec:within}), where pretraining showed no detectable advantage, and was not made on sleep staging.") + t[j:]
    io.open(TEX, "w", encoding="utf-8", newline="").write(t)
    print("Section IV-J (foundation model), Table cbramod and Section V updated")


if __name__ == "__main__":
    main()
