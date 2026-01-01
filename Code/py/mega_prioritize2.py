import math
import os
import pickle
import sys
import tracemalloc
import psutil  # noqa: F401  (you import it, not used in this snippet)

# These must exist in your environment:
import competitors
import fast
import metric2  # <-- renamed from metric
from pympler import muppy, summary  # noqa: F401  (imported but not used here)

usage = """USAGE:
This script runs bounding-box (bbox) or white-box (line/function/branch) prioritization
across multiple algorithms. Typically:
  STR, I-TSD => bounding box only
  ART-D, ART-F, GT, GA, GA-S => white box only
  FAST-* => can do both, depending on arguments.

We store results in ../Results/compareresults/{prog}_{v}/
and read coverage/fault data from ../Input_Data/input_adjlist/{prog}_{v}.
"""

###############################################################################
# Bounding-Box Prioritization (STR, I-TSD, FAST variants)
###############################################################################
def bboxPrioritization(name, prog, v, ctype, k, n, r, b, repeats, selsize):
    """
    BBox algorithms:
      - STR
      - I-TSD
      - FAST-pw, FAST-one, FAST-log, FAST-sqrt, FAST-all
    """
    print(f"starting bboxPrioritization ... (algorithm={name}, entity={ctype})")
    javaFlag = True if v == "v0" else False

    # Input coverage file
    fin = f"/Users/Desktop/vs-projects/good algorithem/AGA-master/Input_Data/input_adjlist/{prog}_{v}/{prog}-{ctype}.txt"

    # Fault matrix
    if javaFlag:
        fault_matrix = f"/Users/Desktop/vs-projects/good algorithem/AGA-master/Input_Data/input_adjlist/{prog}_{v}/fault_matrix.pickle"
    else:
        fault_matrix = f"/Users/Desktop/vs-projects/good algorithem/AGA-master/Input_Data/input_adjlist/{prog}_{v}/fault_matrix_key_tc.pickle"

    outpath = f"/Users/Desktop/test3/{prog}_{v}/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    ppath = os.path.join(outpath, "prioritized")
    if not os.path.exists(ppath):
        os.makedirs(ppath)

    # Collect times & metrics
    ptimes, stimes = [], []
    apfds, napfds, pfds = [], [], []

    # ---- STR ----
    if name == "STR":
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  BBox STR run {run+1}")
                stime, ptime, prioritization = competitors.str_(fin)
                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    # ---- I-TSD ----
    elif name == "I-TSD":
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  BBox I-TSD run {run+1}")
                stime, ptime, prioritization = competitors.i_tsd(fin)
                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    # ---- FAST-* (BBox) ----
    elif name.startswith("FAST-"):
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  BBox {name} run {run+1}")

                if name == "FAST-pw":
                    if javaFlag:
                        stime, ptime, prioritization = fast.fast_pw(fin, r, b, bbox=True, k=k, memory=False)
                    else:
                        stime, ptime, prioritization = fast.fast_pw(fin, r, b, bbox=True, k=k, memory=True)

                else:
                    if name == "FAST-one":
                        def one_(x): return 1
                        sel_fun = one_
                    elif name == "FAST-log":
                        def log_(x): return int(math.log(x, 2)) + 1
                        sel_fun = log_
                    elif name == "FAST-sqrt":
                        def sqrt_(x): return int(math.sqrt(x)) + 1
                        sel_fun = sqrt_
                    elif name == "FAST-all":
                        def all_(x): return x
                        sel_fun = all_
                    else:
                        def pw(x): return x
                        sel_fun = pw

                    if javaFlag:
                        stime, ptime, prioritization = fast.fast_(
                            fin, sel_fun, r=r, b=b, bbox=True, k=k, memory=False
                        )
                    else:
                        stime, ptime, prioritization = fast.fast_(
                            fin, sel_fun, r=r, b=b, bbox=True, k=k, memory=True
                        )

                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    else:
        print("Wrong input or not a bounding-box algorithm:", name)
        print(usage)


###############################################################################
# White-Box Prioritization (GA, GA-S, ART-D, ART-F, GT, FAST variants)
###############################################################################
def wboxPrioritization(name, prog, v, ctype, n, r, b, repeats, selsize):
    """
    White-box algorithms:
      - ART-D, ART-F, GT, GA, GA-S
      - FAST-* variants in WB mode
    """
    print(f"starting wboxPrioritization ... (algorithm={name}, entity={ctype})")
    javaFlag = True if v == "v0" else False

    fin = f"/Users/Desktop/vs-projects/good algorithem/AGA-master/Input_Data/input_adjlist/{prog}_{v}/{prog}-{ctype}.txt"
    if javaFlag:
        fault_matrix = f"/Users/Desktop/vs-projects/good algorithem/AGA-master/Input_Data/input_adjlist/{prog}_{v}/fault_matrix.pickle"
    else:
        fault_matrix = f"/Users/Desktop/vs-projects/good algorithem/AGA-master/Input_Data/input_adjlist/{prog}_{v}/fault_matrix_key_tc.pickle"

    outpath = f"/Users/Desktop/ALLALGO METHOD OUTPUT/{prog}_{v}/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    ppath = os.path.join(outpath, "prioritized")
    if not os.path.exists(ppath):
        os.makedirs(ppath)

    ptimes, stimes = [], []
    apfds, napfds, pfds = [], [], []

    if name == "ART-D":
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  WB ART-D run {run+1}")
                stime, ptime, prioritization = competitors.artd(fin)
                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    elif name == "ART-F":
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  WB ART-F run {run+1}")
                stime, ptime, prioritization = competitors.artf(fin)
                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    elif name == "GT":
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  WB GT run {run+1}")
                stime, ptime, prioritization = competitors.gt(fin)
                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    elif name == "GA":
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  WB GA run {run+1}")

                tracemalloc.start()
                stime, ptime, prioritization = competitors.ga(fin)

                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    elif name == "GA-S":
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  WB GA-S run {run+1}")
                stime, ptime, prioritization = competitors.ga_s(fin)
                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    elif name.startswith("FAST-"):
        if (f"{name}-{ctype}.tsv") not in set(os.listdir(outpath)):
            for run in range(repeats):
                print(f"  WB {name} run {run+1}")

                if name == "FAST-pw":
                    if javaFlag:
                        stime, ptime, prioritization = fast.fast_pw(fin, r, b, memory=False)
                    else:
                        stime, ptime, prioritization = fast.fast_pw(fin, r, b, memory=True)
                else:
                    if name == "FAST-one":
                        def one_(x): return 1
                        sel_fun = one_
                    elif name == "FAST-log":
                        def log_(x): return int(math.log(x, 2)) + 1
                        sel_fun = log_
                    elif name == "FAST-sqrt":
                        def sqrt_(x): return int(math.sqrt(x)) + 1
                        sel_fun = sqrt_
                    elif name == "FAST-all":
                        def all_(x): return x
                        sel_fun = all_
                    else:
                        def default_sel(x): return x
                        sel_fun = default_sel

                    if javaFlag:
                        stime, ptime, prioritization = fast.fast_(fin, sel_fun, r=r, b=b, memory=False)
                    else:
                        stime, ptime, prioritization = fast.fast_(fin, sel_fun, r=r, b=b, memory=True)

                writePrioritization(ppath, name, ctype, run, prioritization)

                apfd_val = metric2.apfd(prioritization, fault_matrix, javaFlag)
                napfd_val = metric2.napfd(prioritization, fault_matrix, javaFlag)
                pfd_val = metric2.pfd(prioritization, fault_matrix, javaFlag, ratio=0.25)

                stimes.append(stime)
                ptimes.append(ptime)
                apfds.append(apfd_val)
                napfds.append(napfd_val)
                pfds.append(pfd_val)

                print(f"    APFD:  {apfd_val}")
                print(f"    NAPFD: {napfd_val}")
                print(f"    PFD@25%: {pfd_val}")

            rep = (name, stimes, ptimes, apfds, napfds, pfds)
            writeOutput(outpath, ctype, rep, javaFlag)
        else:
            print(name, "already run.")

    else:
        print("Wrong input or not a white-box algorithm:", name)
        print(usage)


###############################################################################
# Writers for output
###############################################################################
def writePrioritization(path, name, ctype, run, prioritization):
    """Writes the prioritization (an ordering) to a .pickle file."""
    if not os.path.exists(path):
        os.makedirs(path)
    fout = os.path.join(path, f"{name}-{ctype}-{run+1}.pickle")
    with open(fout, "wb") as f:
        pickle.dump(prioritization, f)


def _emit_rows(fout, st, pt, apfd_val, napfd_val, pfd_val, javaFlag):
    """
    Helper to write rows in a consistent way.
    - For javaFlag=True, metric functions can return lists (per version).
      We write one row per version (keeping same st/pt).
    - For javaFlag=False, they are floats, write one row.
    """
    if javaFlag and isinstance(apfd_val, (list, tuple)):
        # If list lengths differ, we write up to the minimum length to keep columns aligned.
        if isinstance(napfd_val, (list, tuple)):
            L1 = len(apfd_val)
            L2 = len(napfd_val)
        else:
            L1 = len(apfd_val)
            L2 = L1

        if isinstance(pfd_val, (list, tuple)):
            L3 = len(pfd_val)
        else:
            L3 = L1

        L = min(L1, L2, L3)

        for i in range(L):
            fout.write(f"{st}\t{pt}\t{apfd_val[i]}\t{napfd_val[i]}\t{pfd_val[i]}\n")
    else:
        fout.write(f"{st}\t{pt}\t{apfd_val}\t{napfd_val}\t{pfd_val}\n")


def writeOutput(outpath, ctype, res, javaFlag):
    """
    Writes final results (SignatureTime, prioritizationTime, APFD, NAPFD, PFD@25%)
    to a .tsv file in outpath, e.g. "GA-line.tsv".
    """
    name, stimes, ptimes, apfds, napfds, pfds = res
    fileout = os.path.join(outpath, f"{name}-{ctype}.tsv")

    with open(fileout, "w") as fout:
        fout.write("SignatureTime\tPrioritizationTime\tAPFD\tNAPFD\tPFD@25%\n")

        for st, pt, a, na, pf in zip(stimes, ptimes, apfds, napfds, pfds):
            _emit_rows(fout, st, pt, a, na, pf, javaFlag)


###############################################################################
# Main: run all algorithms, entity='function', repeats=1
###############################################################################
def main():
    """
    - Iterates over subfolders in root_directory.
    - For each subfolder, tries to parse <prog>_<v>.
    - Runs bounding-box or white-box logic depending on the algorithm.
    - We fix entity='line' (white-box coverage).
    - We do repeats=1 so each algorithm runs exactly once.
    """

    # Root directory with subfolders (removed trailing space)
    root_directory = "/Users/Desktop/AGA-master/Input_Data/function "

    # Entity type (white-box coverage)
    entity = "function"

    # Run each algorithm once
    repeats = 1

    algorithms = [
        "GA","ART-D", "ART-F", "FAST-all", "FAST-pw",
        # "GA", "GT", "ART-D", "ART-F", "FAST-one", "FAST-pw", "FAST-log", "FAST-sqrt", "FAST-all"
        # "STR", "I-TSD"  # bbox-only
    ]

    # FAST parameters
    k, n, r, b = 5, 10, 1, 10

    def default_sel(x): return x
    selsize = default_sel

    folder_list = [d for d in os.listdir(root_directory)
                   if os.path.isdir(os.path.join(root_directory, d))]

    for folder_name in folder_list:
        if "_" in folder_name:
            prog, version = folder_name.split("_", 1)
        else:
            prog = folder_name
            version = "v0"

        print(f"\n=== Processing folder '{folder_name}' -> (prog={prog}, v={version}) ===")

        for alg in algorithms:
            print(f"  Algorithm: {alg}")

            if alg in {"STR", "I-TSD"}:
                bboxPrioritization(alg, prog, version, entity, k, n, r, b, repeats, selsize)
            else:
                wboxPrioritization(alg, prog, version, entity, n, r, b, repeats, selsize)

    print("\nAll algorithms done with entity='function' and repeats=1.")


if __name__ == "__main__":
    main()
