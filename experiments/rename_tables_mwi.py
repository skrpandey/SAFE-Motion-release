"""Rename the archived label "SE" (shortcut effect) to "MWI" (matched-world interaction) in
paper/tables/*.tex. results/ CSV and JSON fields keep "SE". Run after any table generator."""
import glob, re
n = 0
for p in glob.glob("paper/tables/*.tex"):
    s = open(p).read()
    t = re.sub(r"(?<![A-Za-z])SE(?![A-Za-z])", "MWI", s).replace("shortcut effect", "matched-world interaction")
    if t != s:
        n += 1; open(p, "w").write(t)
print("tables renamed:", n)
