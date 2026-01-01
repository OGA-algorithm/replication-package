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

from collections import defaultdict, OrderedDict
from struct import pack, unpack
import os
import random
import sys
import time

import lsh

def loadTestSuite(input_file, bbox=False, k=5):
    """INPUT
    (str)input_file: path of input file

    OUTPUT
    (dict)TS: key=tc_ID, val=set(covered lines)
    """
    TS = defaultdict()
    with open(input_file) as fin:
        tcID = 1
        for tc in fin:
            if bbox:
                TS[tcID] = tc.rstrip('\n')
            else:
                TS[tcID] = set(tc.rstrip('\n').split())
            tcID += 1

    # Convert dict_items to list before shuffling
    shuffled = list(TS.items())
    random.shuffle(shuffled)
    TS = OrderedDict(shuffled)

    if bbox:
        TS = lsh.kShingles(TS, k)
    return TS

def storeSignatures(input_file, sigfile, hashes, bbox=False, k=5):
    with open(sigfile, "w") as sigout:
        with open(input_file) as fin:
            tcID = 1
            for tc in fin:
                if bbox:
                    # shingling
                    tc_ = tc.rstrip('\n')
                    tc_shingles = set()
                    for i in range(len(tc_) - k + 1):
                        tc_shingles.add(hash(tc_[i : i + k]))
                    sig = lsh.tcMinhashing((tcID, set(tc_shingles)), hashes)
                else:
                    tc_ = tc.rstrip('\n').split()
                    sig = lsh.tcMinhashing((tcID, set(tc_)), hashes)
                for hval in sig:
                    sigout.write(repr(unpack('>d', hval)[0]))
                    sigout.write(" ")
                sigout.write("\n")
                tcID += 1

def loadSignatures(input_file):
    """INPUT
    (str)input_file: path of signature file

    OUTPUT
    (dict) sig: { tcID: list_of_packed_doubles }, (float) time_for_loading
    """
    sig = {}
    start = time.perf_counter()  # replaced time.clock() -> perf_counter()
    with open(input_file, "r") as fin:
        tcID = 1
        for line in fin:
            floats = line.strip().split()
            sig[tcID] = [pack('>d', float(val)) for val in floats]
            tcID += 1
    load_time = time.perf_counter() - start
    return sig, load_time

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def fast_pw(input_file, r, b, bbox=False, k=5, memory=False):
    """
    LSH-based approach with a pairwise check (pw).
    """
    n = r * b  # number of hash functions
    hashes = [lsh.hashFamily(i) for i in range(n)]

    if memory:
        test_suite = loadTestSuite(input_file, bbox=bbox, k=k)
        # generate minhashes signatures
        mh_t_start = time.perf_counter()
        tcs_minhashes = {tcID: lsh.tcMinhashing((tcID, covset), hashes)
                         for tcID, covset in test_suite.items()}
        mh_time = time.perf_counter() - mh_t_start
        ptime_start = time.perf_counter()

    else:
        # loading input file and generating minhashes signatures
        sigfile = input_file.replace(".txt", ".sig")
        sigtimefile = "{}_sigtime.txt".format(input_file.split(".")[0])
        if not os.path.exists(sigfile):
            mh_t_start = time.perf_counter()
            storeSignatures(input_file, sigfile, hashes, bbox, k)
            mh_time = time.perf_counter() - mh_t_start
            with open(sigtimefile, "w") as fout:
                fout.write(repr(mh_time))
        else:
            with open(sigtimefile, "r") as fin:
                mh_time = eval(fin.read().strip())

        ptime_start = time.perf_counter()
        tcs_minhashes, load_time = loadSignatures(sigfile)

    tcs = set(tcs_minhashes.keys())
    BASE = 0.5
    SIZE = int(len(tcs) * BASE) + 1

    bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)

    prioritized_tcs = [0]

    # First TC
    selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
    first_tc = random.choice(list(tcs_minhashes.keys()))
    for i in range(n):
        if tcs_minhashes[first_tc][i] < selected_tcs_minhash[i]:
            selected_tcs_minhash[i] = tcs_minhashes[first_tc][i]
    prioritized_tcs.append(first_tc)
    tcs.remove(first_tc)
    del tcs_minhashes[first_tc]

    iteration, total = 0, float(len(tcs_minhashes))
    while tcs_minhashes:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write(f"  Progress: {round(100 * iteration / total, 2)}%\r")
            sys.stdout.flush()

        if len(tcs_minhashes) < SIZE:
            bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)
            SIZE = int(SIZE * BASE) + 1

        sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash), b, r, n)
        filtered_sim_cand = sim_cand.difference(prioritized_tcs)
        candidates = tcs - filtered_sim_cand

        if not candidates:
            # reset
            selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
            sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash), b, r, n)
            filtered_sim_cand = sim_cand.difference(prioritized_tcs)
            candidates = tcs - filtered_sim_cand
            if not candidates:
                # fallback: all
                candidates = set(tcs_minhashes.keys())

        selected_tc = random.choice(tuple(candidates))
        max_dist = -1
        for candidate in tcs_minhashes:
            if candidate in candidates:
                dist = lsh.jDistanceEstimate(selected_tcs_minhash, tcs_minhashes[candidate])
                if dist > max_dist:
                    selected_tc, max_dist = candidate, dist

        # update selected_tcs_minhash
        for i in range(n):
            if tcs_minhashes[selected_tc][i] < selected_tcs_minhash[i]:
                selected_tcs_minhash[i] = tcs_minhashes[selected_tc][i]

        prioritized_tcs.append(selected_tc)
        tcs.remove(selected_tc)
        del tcs_minhashes[selected_tc]

    ptime = time.perf_counter() - ptime_start
    return mh_time, ptime, prioritized_tcs[1:]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def fast_(input_file, selsize, r, b, bbox=False, k=5, memory=False):
    """
    LSH-based approach with flexible candidate set 'selsize'.
    """
    n = r * b  # number of hash functions
    hashes = [lsh.hashFamily(i) for i in range(n)]

    if memory:
        test_suite = loadTestSuite(input_file, bbox=bbox, k=k)
        # generate minhashes
        mh_t_start = time.perf_counter()
        tcs_minhashes = {tcID: lsh.tcMinhashing((tcID, covset), hashes)
                         for tcID, covset in test_suite.items()}
        mh_time = time.perf_counter() - mh_t_start
        ptime_start = time.perf_counter()

    else:
        # if not memory, store to .sig file
        sigfile = input_file.replace(".txt", ".sig")
        sigtimefile = "{}_sigtime.txt".format(input_file.split(".")[0])
        if not os.path.exists(sigfile):
            mh_t_start = time.perf_counter()
            storeSignatures(input_file, sigfile, hashes, bbox, k)
            mh_time = time.perf_counter() - mh_t_start
            with open(sigtimefile, "w") as fout:
                fout.write(repr(mh_time))
        else:
            with open(sigtimefile, "r") as fin:
                mh_time = eval(fin.read().strip())

        ptime_start = time.perf_counter()
        tcs_minhashes, load_time = loadSignatures(sigfile)

    tcs = set(tcs_minhashes.keys())
    BASE = 0.5
    SIZE = int(len(tcs) * BASE) + 1

    bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)

    prioritized_tcs = [0]

    # First TC
    selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
    first_tc = random.choice(list(tcs_minhashes.keys()))
    for i in range(n):
        if tcs_minhashes[first_tc][i] < selected_tcs_minhash[i]:
            selected_tcs_minhash[i] = tcs_minhashes[first_tc][i]
    prioritized_tcs.append(first_tc)
    tcs.remove(first_tc)
    del tcs_minhashes[first_tc]

    iteration, total = 0, float(len(tcs_minhashes))
    while tcs_minhashes:
        iteration += 1
        if iteration % 100 == 0:
            sys.stdout.write(f"  Progress: {round(100 * iteration / total, 2)}%\r")
            sys.stdout.flush()

        if len(tcs_minhashes) < SIZE:
            bucket = lsh.LSHBucket(tcs_minhashes.items(), b, r, n)
            SIZE = int(SIZE * BASE) + 1

        sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash), b, r, n)
        filtered_sim_cand = sim_cand.difference(prioritized_tcs)
        candidates = tcs - filtered_sim_cand

        if not candidates:
            # reset
            selected_tcs_minhash = lsh.tcMinhashing((0, set()), hashes)
            sim_cand = lsh.LSHCandidates(bucket, (0, selected_tcs_minhash), b, r, n)
            filtered_sim_cand = sim_cand.difference(prioritized_tcs)
            candidates = tcs - filtered_sim_cand
            if not candidates:
                candidates = set(tcs_minhashes.keys())

        to_sel = min(selsize(len(candidates)), len(candidates))
        selected_tc_set = random.sample(list(candidates), to_sel)

        for selected_tc in selected_tc_set:
            for i in range(n):
                if tcs_minhashes[selected_tc][i] < selected_tcs_minhash[i]:
                    selected_tcs_minhash[i] = tcs_minhashes[selected_tc][i]

            prioritized_tcs.append(selected_tc)
            tcs.remove(selected_tc)
            del tcs_minhashes[selected_tc]

    ptime = time.perf_counter() - ptime_start
    return mh_time, ptime, prioritized_tcs[1:]
