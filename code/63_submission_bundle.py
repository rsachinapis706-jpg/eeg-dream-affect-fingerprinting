"""
Assemble the LaTeX archive for the "Main Manuscript" upload.

The portal accepts a single archive containing the source, the bibliography,
the figures, and the classes and packages the document needs. An archive is not
merely convenient here, it is necessary: the class file has been modified to
drop the placeholder volume footer, so a build against the stock ieeeaccess.cls
would reintroduce it.

One correction is applied while packing. ieeeaccess.cls loads Logo.png and
notaglineLogo.png with capital letters, but the template ships logo.png and
notaglinelogo.png in lower case. That difference is invisible on Windows and
fatal on the Linux systems that typeset the article, so both spellings go into
the archive.
"""
import io
import os
import re
import zipfile

SRC = os.path.join("E:" + os.sep, "Dream_Analysis")
TPL = os.path.join(SRC, "IEEE_Access_LaTeX_template__1_")
TEX = "accesstemp.tex"
OUT = os.path.join(SRC, "Access-2026-50879_revised_LaTeX_source.zip")  # never overwrite a submitted archive

# Images the class itself pulls in, with the capitalisation it asks for.
CLASS_ART = ["Logo.png", "notaglineLogo.png", "bullet.png"]
SUPPORT = ["ieeeaccess.cls", "IEEEtran.cls"]
EXTS = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")

# The biography photographs are on disk under different names from the ones the
# source asks for. The source is left alone -- it matches the project the
# authors compile in -- and the files are supplied under the expected names
# instead. Only the first needs a real format conversion; .jpg and .jpeg are
# the same format, so those are copies.
PHOTO_ALIAS = {
    "sachin.png": "sachin.jpg",
    "jagan1.jpeg": "jaganath.jpg",
    "yeshwanth.jpeg": "yeshwanth.jpg",
    "abishek.jpeg": "abhishek.jpg",
}


def find(name):
    """Locate a file by name, case-insensitively, in either directory."""
    for d in (SRC, TPL):
        for ext in EXTS:
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return p
        try:
            entries = os.listdir(d)
        except OSError:
            continue
        for ext in EXTS:
            want = (name + ext).lower()
            for e in entries:
                if e.lower() == want:
                    return os.path.join(d, e)
    return None


def main():
    tex = io.open(os.path.join(SRC, TEX), encoding="utf-8").read()

    # Materialise the photographs under the names the source expects.
    from PIL import Image
    for want, have in PHOTO_ALIAS.items():
        dst, src = os.path.join(SRC, want), os.path.join(SRC, have)
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        if os.path.splitext(want)[1].lower() == ".png":
            Image.open(src).save(dst, "PNG")
        else:
            io.open(dst, "wb").write(io.open(src, "rb").read())
        print(f"  prepared {want} from {have}")

    wanted, missing = [], []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        n = m.group(1)
        p = find(n)
        (wanted if p else missing).append((n, p))

    for n in CLASS_ART + SUPPORT:
        p = find(n)
        (wanted if p else missing).append((n, p))

    bib = re.search(r"\\bibliography\{([^}]+)\}", tex)
    if bib:
        p = find(bib.group(1) + ".bib") or find(bib.group(1))
        (wanted if p else missing).append((bib.group(1) + ".bib", p))

    if missing:
        for n, _ in missing:
            print(f"  MISSING  {n}")
        raise SystemExit("archive not written; resolve the missing files first")

    seen = set()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(SRC, TEX), TEX)
        for name, path in wanted:
            # store under the name the source asks for, whatever the disk says
            arc = os.path.basename(path)
            if not os.path.splitext(name)[1]:
                arc = name + os.path.splitext(path)[1]
            elif name in CLASS_ART:
                arc = name           # the capitalisation the class expects
            if arc in seen:
                continue
            seen.add(arc)
            z.write(path, arc)

    print(f"archive : {OUT}")
    print(f"entries : {len(seen) + 1}")
    for a in sorted(seen):
        print("   ", a)


if __name__ == "__main__":
    main()
