'''
This file is part of an ICSE'18 submission that is currently under review.
For more information visit: https://github.com/icse18-FAST/FAST.

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this source.  If not, see <http://www.gnu.org/licenses/>.
'''

# WHITE BOX PRIORITIZATIONS
# GreedyTotal, GreedyAdditional, AdditionalSpanning, Jiang, Zhou

# BLACK BOX PRIORITIZATIONS
# Ledru, I-TSD

from collections import defaultdict
from collections import OrderedDict
from functools import reduce
from pickle import dump, load
from struct import pack, unpack
import bz2
import itertools
import math
import os
import random
import time
import scipy.special  # ledru statistics
import subprocess
import sys
from pympler import muppy, summary

import lsh


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def loadTestSuite(input_file, bbox=False, k=5):
    """INPUT
    (str)input_file: path of input file
    (bool)bbox: apply k-shingles to input
    (int)k: k-shingle size

    OUTPUT
    (dict)TS: key=tc_ID, val=set(covered lines/shingles)"""
    TS = {}
    with open(input_file) as fin:
        tcID = 1
        for tc in fin:
            if bbox:
                TS[tcID] = tc[:-1]
            else:
                TS[tcID] = set(tc[:-1].split())
            tcID += 1

    # Convert dict.items() to a list before shuffling
    shuffled = list(TS.items())
    random.shuffle(shuffled)
    TS = OrderedDict(shuffled)

    if bbox:
        TS = lsh.kShingles(TS, k)
    return TS


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# GREEDY SET COVER (TOTAL)
def gt(input_file):
    def loadTestSuite(input_file):
        TS = {}
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                TS[tcID] = len(set(tc[:-1].split()))
                tcID += 1

        # Fix: shuffle by converting to a list first
        shuffled = list(TS.items())
        random.shuffle(shuffled)
        TS = OrderedDict(shuffled)
        return TS

    ptime_start = time.perf_counter()
    TCS = loadTestSuite(input_file)

    # Sort by descending coverage size
    TS = OrderedDict(sorted(TCS.items(), key=lambda t: -t[1]))
    ptime = time.perf_counter() - ptime_start

    return 0.0, ptime, list(TS.keys())

# GREEDY SET COVER (ADDITIONAL)
def ga(input_file):
    def select(TS, U, Cg):
        s, uncs_s = 0, -1
        for ui in U:
            uncs = len(TS[ui] - Cg)
            if uncs > uncs_s:
                s, uncs_s = ui, uncs
        return s

    ptime_start = time.perf_counter()
    TCS = loadTestSuite(input_file)  # calls the top-level loadTestSuite with k=5 default
    TS = OrderedDict(sorted(TCS.items(), key=lambda t: -len(t[1])))

    U = TS.copy()
    Cg = set()
    TS[0] = set()
    P = [0]

    maxC = len(reduce(lambda x, y: x | y, TS.values()))

    while len(U) > 0:
        if len(Cg) == maxC:
            Cg = set()
        s = select(TS, U, Cg)
        P.append(s)
        Cg |= U[s]
        del U[s]

    ptime = time.perf_counter() - ptime_start
    return 0.0, ptime, P[1:]




# GREEDY SET COVER (ADDITIONAL) with early return
def gaf(input_file):
    def select(TS, U, Cg):
        s, uncs_s = 0, -1
        for ui in U:
            uncs = len(TS[ui] - Cg)
            if uncs > uncs_s:
                s, uncs_s = ui, uncs
        return s

    ptime_start = time.perf_counter()
    TCS = loadTestSuite(input_file)
    TS = OrderedDict(sorted(TCS.items(), key=lambda t: -len(t[1])))

    U = TS.copy()
    Cg = set()
    TS[0] = set()
    P = [0]

    maxC = len(reduce(lambda x, y: x | y, TS.values()))

    while len(U) > 0:
        if len(Cg) == maxC:
            # early return
            ptime = time.perf_counter() - ptime_start
            return 0.0, ptime, P[1:]
            # unreachable code if we return:
            Cg = set()
        s = select(TS, U, Cg)
        P.append(s)
        Cg |= U[s]
        del U[s]

    ptime = time.perf_counter() - ptime_start
    return 0.0, ptime, P[1:]


# JIANG (ART) - dynamic candidate set
def artd(input_file):
    def generate(U):
        C, T = set(), set()
        while True:
            ui = random.choice(list(U.keys()))
            S = U[ui]
            if T | S == T:
                break
            T |= S
            C.add(ui)
        return C

    def select(TS, P, C):
        D = {}
        for cj in C:
            D[cj] = {}
            for pi in P:
                D[cj][pi] = lsh.jDistance(TS[pi], TS[cj])
        # maximum among the minimum distances
        j, jmax = 0, -1
        for cj in D.keys():
            min_di = min(D[cj].values())
            if min_di > jmax:
                j, jmax = cj, min_di
        return j

    ptime_start = time.perf_counter()
    TS = loadTestSuite(input_file)
    U = TS.copy()

    # remove empty coverage lines (added by Feng Li)
    empty = {}
    for ttt in list(U.keys()):
        if len(U[ttt]) == 0:
            empty[ttt] = U[ttt]
            del U[ttt]

    TS[0] = set()
    P = [0]
    C = generate(U)

    iteration, total = 0, float(len(U))
    while len(U) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100 * iteration / total, 2)))
            sys.stdout.flush()

        if len(C) == 0:
            C = generate(U)
        s = select(TS, P, C)
        P.append(s)
        del U[s]
        C.discard(s)

    # after finishing, add empty coverage tests to the end
    for ttt in empty.keys():
        P.append(ttt)

    ptime = time.perf_counter() - ptime_start
    return 0.0, ptime, P[1:]


# ZHOU (ART) - fixed-size candidate set + manhattan distance
def artf(input_file):
    def generate(U):
        C = set()
        if len(U) < 10:
            C = set(U.keys())
        else:
            while len(C) < 10:
                ui = random.choice(list(U.keys()))
                C.add(ui)
        return C

    def manhattanDistance(TCS, i, j):
        u, v = TCS[i], TCS[j]
        return sum(abs(float(ui) - float(vi)) for ui, vi in zip(u, v))

    def select(TS, P, C):
    # If P is empty, pick any from C
     if not P:
        return random.choice(list(C))

     D = {}
     for cj in C:
        D[cj] = {}
        for pi in P:
            D[cj][pi] = manhattanDistance(TS, pi, cj)

    # Then do the min(...) logic
     j, jmax = None, -1
     for cj in D:
        min_di = min(D[cj].values())   # now guaranteed not empty
        if min_di > jmax:
            j, jmax = cj, min_di
     return j

    ptime_start = time.perf_counter()
    TS = loadTestSuite(input_file)
    U = TS.copy()

    TCS = U  # used by manhattanDistance
    #TS[0] = set()
    #P = [0]
     # Instead, start P empty:
    P = []
    C = generate(U)
    iteration, total = 0, float(len(U))
    while len(U) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100 * iteration / total, 2)))
            sys.stdout.flush()

        if len(C) == 0:
            C = generate(U)
        s = select(TS, P, C)
        P.append(s)
        del U[s]
        C.discard(s)

    ptime = time.perf_counter() - ptime_start
    return 0.0, ptime, P[1:]


# WHITEBOX ONLY
# ADDITIONAL SPANNING (LINE, BRANCH, FUNCTION)
def ga_s(input_file):
    def storeSpanningFile(input_file, spanfile):
        TCS = loadTestSuite(input_file)
        matrixFile = "{}.mat".format(spanfile)
        with open(matrixFile, "w") as fout:
            for tcID, tc in TCS.items():
                for cov in tc:
                    fout.write("{} {}\n".format(cov, tcID))

        # subsume.pl perl script process files and creates spanningEntityFile
        with open(spanfile + ".tmp", "w") as fout:
            subprocess.call(["perl", "py/subsume.pl", matrixFile], stdout=fout)
        os.remove(matrixFile)

        # compute spanning file
        with open(spanfile + ".tmp") as fin:
            spans = {line.strip() for line in fin}
            for tcID, tc in TCS.items():
                TCS[tcID] = tc & spans
        os.remove(spanfile + ".tmp")

        # store spanning file
        with open(spanfile, "w") as fout:
            for tcID in range(1, len(TCS)):
                fout.write(" ".join(TCS[tcID]) + "\n")

    def select(TS, U, Cg):
        s, uncs_s = 0, -1
        for ui in U:
            uncs = len(TS[ui] - Cg)
            if uncs > uncs_s:
                s, uncs_s = ui, uncs
        return s

    spanfile = input_file.replace(".txt", ".span")
    spantimefile = input_file.replace(".txt", "_spantime.txt")

    if not os.path.exists(spanfile):
        span_t = time.time()
        storeSpanningFile(input_file, spanfile)
        stime = time.time() - span_t
        with open(spantimefile, "w") as fout:
            fout.write(repr(stime))
    else:
        with open(spantimefile, "r") as fin:
            stime = eval(fin.read().replace("\n", ""))

    ptime_start = time.perf_counter()
    TCS = loadTestSuite(spanfile)

    TS = OrderedDict(sorted(TCS.items(), key=lambda t: -len(t[1])))
    U = TS.copy()
    Cg = set()

    TS[0] = set()
    P = [0]

    maxC = len(reduce(lambda x, y: x | y, TS.values()))

    iteration, total = 0, float(len(TCS))
    while len(U) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100 * iteration / total, 2)))
            sys.stdout.flush()

        if len(Cg) == maxC:
            Cg = set()
        s = select(TS, U, Cg)
        P.append(s)
        Cg |= U[s]
        del U[s]

    ptime = time.perf_counter() - ptime_start
    return stime, ptime, P[1:]


# BLACK BOX
def str_(input_file):
    def loadTestSuite(input_file):
        TS = {}
        maxlen = 0
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                if len(tc) > maxlen:
                    maxlen = len(tc)
                TS[tcID] = tc[:-1]
                tcID += 1

        # Convert to list for shuffling
        shuffled = list(TS.items())
        random.shuffle(shuffled)
        return OrderedDict(shuffled)

    def manhattanDistance(TCS, i, j):
        u, v = TCS[i], TCS[j]
        return sum([abs(float(ui) - float(vi)) for ui, vi in zip(u, v)])

    def storePairwiseDistance(TCS, sigfile):
        D = defaultdict(float)
        combs = scipy.special.binom(len(TCS.keys()), 2)
        iteration, total = 0, float(combs)
        for pair in itertools.combinations(TCS.keys(), 2):
            iteration += 1
            if iteration % 100 == 0:
                sys.stdout.write("  Progress: {}%\r".format(
                    round(100*iteration/total, 2)))
                sys.stdout.flush()

            i, j = pair
            if i < j:
                D[(i, j)] = manhattanDistance(TCS, i, j)
        dump(D, open(sigfile, "wb"))

    def loadPairwiseDistance(sigfile):
        return load(open(sigfile, "rb"))

    def removeDuplicates(TCS):
        unique = set()
        P = []
        for tcID, tc in TCS.items():
            tc_tuple = tuple(tc)
            if tc_tuple in unique:
                P.append(tcID)
            else:
                unique.add(tc_tuple)
        for tcID in P:
            del TCS[tcID]
        return P

    def select(TCS, D, T):
        s, dist_s = 0, -1
        for ui in TCS:
            dist = float("Inf")
            for vi in T:
                # must handle (ui, vi) or (vi, ui) depending on which is smaller
                pair = (ui, vi) if ui < vi else (vi, ui)
                if pair in D and D[pair] < dist:
                    dist = D[pair]
            if dist > dist_s:
                s, dist_s = ui, dist
        return s

    TCS = loadTestSuite(input_file)
    P1 = []

    sigfile = input_file.replace(".txt", "___.pickle").replace("___", "_distmatrix")
    if not os.path.exists(sigfile):
        ledru_t = time.perf_counter()
        storePairwiseDistance(TCS, sigfile)
        ledru_time = time.perf_counter() - ledru_t
        with open("{}_sigtime.txt".format(input_file.split(".")[0]), "w") as fout:
            fout.write(repr(ledru_time))
    else:
        with open("{}_sigtime.txt".format(input_file.split(".")[0]), "r") as fin:
            ledru_time = eval(fin.read().replace("\n", ""))

    ptime_start = time.perf_counter()
    D = loadPairwiseDistance(sigfile)

    P2 = removeDuplicates(TCS)
    s = select(TCS, D, TCS.keys())
    P1.append(s)
    del TCS[s]

    iteration, total = 0, float(len(TCS))
    while len(TCS) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100*iteration/total, 2)))
            sys.stdout.flush()

        s = select(TCS, D, P1)
        P1.append(s)
        del TCS[s]

    ptime = time.perf_counter() - ptime_start
    return ledru_time, ptime, P1 + P2


# I-TSD
def i_tsd(input_file):
    def loadTestSuite(input_file):
        TS = {}
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                TS[tcID] = tc[:-1]
                tcID += 1

        # Convert to list for shuffle
        shuffled = list(TS.items())
        random.shuffle(shuffled)
        TS = OrderedDict(shuffled)
        return TS

    def compressExcept(TCS, toExclude):
        s = " ".join([TCS[tcID] for tcID in TCS.keys() if tcID != toExclude])
        cs = bz2.compress(s)
        return sys.getsizeof(cs)

    def select(TCS):
        maxIndex, maxCompress = 0, 0
        for tcID in TCS.keys():
            c = compressExcept(TCS, tcID)
            if c > maxCompress:
                maxIndex, maxCompress = tcID, c
        return maxIndex

    stime = 0.0
    ptime_start = time.perf_counter()
    TCS = loadTestSuite(input_file)

    P = [0]
    iteration, total = 0, float(len(TCS))
    while len(TCS) > 0:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write("  Progress: {}%\r".format(
                round(100*iteration/total, 2)))
            sys.stdout.flush()

        s = select(TCS)
        P.append(s)
        del TCS[s]

    ptime = time.perf_counter() - ptime_start
    return stime, ptime, P[1:]
