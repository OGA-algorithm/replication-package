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

from collections import defaultdict
from pickle import load
import math


def apfd(prioritization, fault_matrix, javaFlag):
    """INPUT:
    (list)prioritization: list of prioritization of test cases
    (str)fault_matrix: path of fault_matrix (pickle file)
    (bool)javaFlag: True if output for Java fault_matrix

    OUTPUT:
    (float)APFD = 1 - (sum_{i=1}^{m} t_i / n*m) + (1 / 2n)
    n = number of test cases
    m = number of faults detected
    t_i = position of first test case revealing fault i in the prioritization
    Average Percentage of Faults Detected
    """

    if javaFlag:
        # key=version, val=[faulty_tcs]
        faults_dict = getFaultDetected(fault_matrix)
        apfds = []
        for v in range(1, len(faults_dict)+1):

            faulty_tcs = set(faults_dict[v])
            numerator = 0.0  # numerator of APFD
            position = 1
            m = 0.0
            for tc_ID in prioritization:
                if tc_ID in faulty_tcs:
                    numerator, m = position, 1.0
                    break
                position += 1

            n = len(prioritization)
            apfd_val = 1.0 - (numerator / (n * m)) + (1.0 / (2 * n)) if m > 0 else 0.0
            apfds.append(apfd_val)

        return apfds

    else:
        # dict: key=tcID, val=[detected faults]
        faults_dict = getFaultDetected(fault_matrix)
        detected_faults = set()
        numerator = 0.0  # numerator of APFD
        position = 1
        for tc_ID in prioritization:
            for fault in faults_dict[tc_ID]:
                if fault not in detected_faults:
                    detected_faults.add(fault)
                    numerator += position
            position += 1

        n, m = len(prioritization), len(detected_faults)
        apfd_val = 1.0 - (numerator / (n * m)) + (1.0 / (2 * n)) if m > 0 else 0.0

        return apfd_val


# =============================================================================
# NEW: NAPFD
# =============================================================================
def napfd(prioritization, fault_matrix, javaFlag):
    """
    Normalized APFD (NAPFD), suitable when not all faults are detected.

    NAPFD = p - (sum_{i=1..m_d} t_i / (n*m)) + (p / (2n))
    where:
      n   = number of tests in the prioritization
      m   = total number of faults (or versions in javaFlag=True mode)
      m_d = number of detected faults
      p   = m_d / m
      t_i = position of first test revealing fault i (only for detected faults)
    """

    if javaFlag:
        # In javaFlag mode, each "version" is effectively one fault.
        faults_dict = getFaultDetected(fault_matrix)

        napfds = []
        n = len(prioritization)
        if n == 0:
            return [0.0 for _ in range(len(faults_dict))]

        for v in range(1, len(faults_dict) + 1):
            faulty_tcs = set(faults_dict[v])

            # Find first failing test position (if any)
            t = None
            position = 1
            for tc_ID in prioritization:
                if tc_ID in faulty_tcs:
                    t = position
                    break
                position += 1

            # m is 1 fault per version; md is 1 if detected else 0
            m = 1.0
            md = 1.0 if t is not None else 0.0
            p = md / m  # either 1 or 0

            sum_t = float(t) if t is not None else 0.0
            napfd_val = p - (sum_t / (n * m)) + (p / (2.0 * n)) if n > 0 else 0.0
            napfds.append(napfd_val)

        return napfds

    else:
        faults_dict = getFaultDetected(fault_matrix)

        n = len(prioritization)
        if n == 0:
            return 0.0

        detected_faults = set()
        sum_t = 0.0
        position = 1

        # compute sum of first-detection positions for detected faults
        for tc_ID in prioritization:
            for fault in faults_dict[tc_ID]:
                if fault not in detected_faults:
                    detected_faults.add(fault)
                    sum_t += position
            position += 1

        md = float(len(detected_faults))

        # total faults m = union of all faults present in the matrix
        all_faults = set()
        for tc_ID in faults_dict:
            for fault in faults_dict[tc_ID]:
                all_faults.add(fault)
        m = float(len(all_faults))

        if m == 0:
            return 0.0

        p = md / m
        napfd_val = p - (sum_t / (n * m)) + (p / (2.0 * n))
        return napfd_val


# =============================================================================
# NEW: PFD@k
# =============================================================================
def pfd(prioritization, fault_matrix, javaFlag, k=None, ratio=None):
    """
    Percent/Proportion of Faults Detected within first k tests.

    You can specify either:
      - ratio (e.g., 0.25 for PFD@25%), OR
      - k (absolute number of tests)

    PFD@k = (# faults detected by first k tests) / (total faults m)

    Returns:
      - javaFlag=True: list[float] (one per version)
      - javaFlag=False: float
    """

    n = len(prioritization)
    if n == 0:
        return [] if javaFlag else 0.0

    if ratio is not None:
        k_eff = int(math.ceil(ratio * n))
    elif k is not None:
        k_eff = int(k)
    else:
        # default: PFD@25%
        k_eff = int(math.ceil(0.25 * n))

    k_eff = max(1, min(k_eff, n))

    faults_dict = getFaultDetected(fault_matrix)

    if javaFlag:
        # For each version: PFD@k is either 1 (found by first k) or 0 (not found)
        pfds = []
        for v in range(1, len(faults_dict) + 1):
            faulty_tcs = set(faults_dict[v])
            found = 0.0
            for pos, tc_ID in enumerate(prioritization[:k_eff], start=1):
                if tc_ID in faulty_tcs:
                    found = 1.0
                    break
            # total faults per version is 1
            pfds.append(found)
        return pfds

    else:
        # total faults m = union of all faults in the matrix
        all_faults = set()
        for tc_ID in faults_dict:
            for fault in faults_dict[tc_ID]:
                all_faults.add(fault)
        m = float(len(all_faults))
        if m == 0:
            return 0.0

        detected_by_k = set()
        for tc_ID in prioritization[:k_eff]:
            for fault in faults_dict[tc_ID]:
                detected_by_k.add(fault)

        return float(len(detected_by_k)) / m


def getFaultDetected(fault_matrix):
    """INPUT:
    (str)fault_matrix: path of fault_matrix (pickle file)

    OUTPUT:
    (dict)faults_dict: key=tcID, val=[detected faults]
    """
    faults_dict = defaultdict(list)

    with open(fault_matrix, "rb") as picklefile:
        pickledict = load(picklefile)
    for key in pickledict.keys():
        faults_dict[int(key)] = pickledict[key]

    return faults_dict

