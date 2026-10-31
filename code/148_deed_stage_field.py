"""
The stage field of DEED's file names, as the dataset paper defines it (round
3, Reviewer 2, comment 3).

Reviewer 2 asks for authoritative documentation of what the analysed
intervals represent, since the ratings workbook calls each interval the REM
period before the awakening while the file-name stage tag reads REM for only
18 recordings. The dataset paper (Liu et al., bioRxiv 2022, doi
10.1101/2022.09.19.508475, Data Records) gives the naming pattern
EEG_{subject+experiment}_{Movie}_{Emotion}_{R+order of REM}_{sleep stage
before REM}, and its protocol wakes participants by a tone five minutes after
the onset of REM. The tag is therefore the stage before the REM period, not
the stage of the interval, and the two sources do not conflict. The paper
does not say how the extracted intervals were bounded. Idempotent.
"""
import io

TEX = r"E:\Dream_Analysis\accesstemp.tex"
OLD = ("The workbook labels that interval as the REM period preceding the "
       "awakening, whereas the stage tag in the file names reads REM for only "
       "18 recordings, N1 for 219, N2 for 143, W for 103 and N3 for 2, with 48 "
       "untagged; the interval is used as distributed and is not re-segmented.")
NEW = ("The workbook labels that interval as the REM period preceding the "
       "awakening, in agreement with the protocol of the dataset paper, in "
       "which participants were woken by a tone five minutes after the onset "
       "of REM \\cite{liu2022deed}. The stage field of the file names does not "
       "contradict this: the dataset paper defines it as the sleep stage "
       "before REM \\cite{liu2022deed}, and it reads REM for 18 recordings, N1 "
       "for 219, N2 for 143, W for 103 and N3 for 2, with 48 untagged. The "
       "dataset paper does not state how the extracted intervals were "
       "bounded, so for the 175 recordings without a workbook match the "
       "interval cannot be confirmed independently; it is used as distributed "
       "and is not re-segmented.")


def main():
    s = io.open(TEX, encoding="utf-8", newline="").read()
    if OLD in s:
        assert s.count(OLD) == 1
        s = s.replace(OLD, NEW)
    assert s.count(NEW) == 1
    io.open(TEX, "w", encoding="utf-8", newline="").write(s)
    print("stage field described as defined by the dataset paper")


if __name__ == "__main__":
    main()
