import os
import copy
import time
from bitarray import bitarray
import gc
from pympler import asizeof
import re
import tracemalloc
import math
import fnmatch  # <-- ADDED

###############################################################################
# CONFIG
###############################################################################
# PFD cut point: 0.25 means PFD@25% of the prioritized test sequence
PFD_CUT_RATIO = 0.25

# =========================
# ADDED: coverage mode switch
# =========================
# Choose one:
#   "statement" -> state-map.txt (statement coverage)
#   "function"  -> function/method coverage (e.g., asteriskjava-function.txt)
COVERAGE_MODE = "function"   # <-- change to "function" when needed

# Patterns we accept (case-insensitive via lower()).
COVERAGE_CANDIDATES = {
    "statement": [
        "state-map.txt",
        "*state-map*.txt",
    ],
    "function": [
        "function.txt",
        "functions.txt",
        "method.txt",
        "methods.txt",
        "*function*.txt", 
        "*method*.txt",
        "*-function.txt",
        "*-method.txt",
    ],
}

def find_coverage_file(project_dir: str, mode: str):
   
    mode = (mode or "").strip().lower()
    if mode not in COVERAGE_CANDIDATES:
        raise ValueError(f"Invalid COVERAGE_MODE='{mode}'. Use 'statement' or 'function'.")

    files = os.listdir(project_dir)
    # Map for exact case-insensitive lookup
    lower_map = {f.lower(): f for f in files}

    # 1) exact match first
    for cand in COVERAGE_CANDIDATES[mode]:
        if "*" not in cand:
            key = cand.lower()
            if key in lower_map:
                return os.path.join(project_dir, lower_map[key])

    # 2) wildcard match
    lower_files = [f.lower() for f in files]
    for pat in COVERAGE_CANDIDATES[mode]:
        if "*" in pat:
            pat_l = pat.lower()
            for lf, orig in zip(lower_files, files):
                if fnmatch.fnmatch(lf, pat_l):
                    return os.path.join(project_dir, orig)

    return None


###############################################################################
# 1) HELPERS FOR FILE READING AND AGA ALGORITHM
###############################################################################

def read_file_lines(filepath):
    """Reads a file and returns a list of lines (stripped)."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f.readlines()]

def sort_by_count(t):
    """Helper to sort by negative count (descending)."""
    return -t[1]

def greedyadditional_new(
    testlist, coveragelist,
    total_count, used, has_change,
    forward_index, inverted_index,
    resultpath
):
    
    additional_order = []
    total_count_backup = copy.deepcopy(total_count)

    pass_count = 0
    intermediate_timelist = []

    for _ in range(len(testlist)):
        maxvalue = 0
        maxindex = -1

        for j in range(len(total_count)):
            if not used[j] and total_count[j] > maxvalue:
                maxvalue = total_count[j]
                maxindex = j

        if maxindex == -1:
            pass_count += 1

            with open(os.path.join(resultpath, f'SequenceGAMethod_{pass_count}.txt'), 'w') as f:
                for item in additional_order:
                    f.write(item + '\n')

                remaining_list = []
                for j in range(len(total_count_backup)):
                    if not used[j]:
                        remaining_list.append((testlist[j], total_count_backup[j]))
                remaining_list_sorted = sorted(remaining_list, key=sort_by_count)
                for testtuple in remaining_list_sorted:
                    f.write(testtuple[0] + '\n')

            thistime = time.time()
            intermediate_timelist.append(thistime)

            total_count = copy.deepcopy(total_count_backup)
            for j in range(len(has_change)):
                has_change[j] = False

            maxvalue = 0
            maxindex = -1
            for j in range(len(total_count)):
                if not used[j] and total_count[j] > maxvalue:
                    maxvalue = total_count[j]
                    maxindex = j

        if maxindex != -1:
            additional_order.append(testlist[maxindex])
            used[maxindex] = True

            for stmt_id in forward_index[maxindex]:
                if not has_change[stmt_id]:
                    has_change[stmt_id] = True
                    for test_idx in inverted_index[stmt_id]:
                        total_count[test_idx] -= 1
        else:
            for j in range(len(total_count)):
                if not used[j]:
                    additional_order.append(testlist[j])
            break

    return [additional_order, intermediate_timelist]

###############################################################################
# 2) METRICS: APFD / NAPFD / PFD
###############################################################################

def _read_sequence(sequence_file):
    if not os.path.exists(sequence_file):
        return []
    with open(sequence_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def _compute_first_detection_positions(sequence_file, testlist, mutantkillmatrix):
    
    final_sequence = _read_sequence(sequence_file)
    if not final_sequence or not mutantkillmatrix:
        return final_sequence, []

    # Map test name -> row index in mutantkillmatrix
    testname_index = {name: i for i, name in enumerate(testlist)}

    N = len(final_sequence)
    M = len(mutantkillmatrix[0]) if mutantkillmatrix and mutantkillmatrix[0] else 0
    if N == 0 or M == 0:
        return final_sequence, []

    first_detect_pos = [None] * M

    for rank, testname in enumerate(final_sequence, start=1):
        if testname not in testname_index:
            continue
        t_idx = testname_index[testname]
        row = mutantkillmatrix[t_idx]

        # row is a string of '0'/'1'
        for j, char in enumerate(row):
            if char == '1' and first_detect_pos[j] is None:
                first_detect_pos[j] = rank

    return final_sequence, first_detect_pos

def calculate_apfd(sequence_file, testlist, mutantkillmatrix):
   
    final_sequence, first_detect_pos = _compute_first_detection_positions(sequence_file, testlist, mutantkillmatrix)
    if not final_sequence or not first_detect_pos:
        return None

    N = len(final_sequence)
    M = len(first_detect_pos)
    if N == 0 or M == 0:
        return None

    apfd_sum = 0.0
    for pos in first_detect_pos:
        if pos is not None:
            apfd_sum += pos

    apfd = 1.0 - (apfd_sum / (N * M)) + (1.0 / (2.0 * N))
    return apfd

def calculate_napfd(sequence_file, testlist, mutantkillmatrix):
   
    final_sequence, first_detect_pos = _compute_first_detection_positions(sequence_file, testlist, mutantkillmatrix)
    if not final_sequence or not first_detect_pos:
        return None

    N = len(final_sequence)
    M = len(first_detect_pos)
    if N == 0 or M == 0:
        return None

    detected_positions = [pos for pos in first_detect_pos if pos is not None]
    md = len(detected_positions)
    p = md / M

    sum_tf = float(sum(detected_positions))  # only detected faults
    napfd = p - (sum_tf / (N * M)) + (p / (2.0 * N))
    return napfd

def calculate_pfd(sequence_file, testlist, mutantkillmatrix, cut_ratio=PFD_CUT_RATIO):
   
    final_sequence, first_detect_pos = _compute_first_detection_positions(sequence_file, testlist, mutantkillmatrix)
    if not final_sequence or not first_detect_pos:
        return None

    N = len(final_sequence)
    M = len(first_detect_pos)
    if N == 0 or M == 0:
        return None

    k = int(math.ceil(cut_ratio * N))
    k = max(1, min(k, N))

    detected_by_k = sum(1 for pos in first_detect_pos if (pos is not None and pos <= k))
    pfd = detected_by_k / M
    return pfd

###############################################################################
# 3) MAIN SCRIPT TO SCAN DIRECTORIES, RUN AGA, CALCULATE METRICS, BUILD HTML
###############################################################################

def extract_total_time_from_file(file_path):
   
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    time_values = []
    for line in lines:
        parts = line.strip().split(',')
        if not parts:
            continue
        # Skip the pre-time line
        if parts[0] == 'pre':
            continue
        if len(parts) >= 2:
            try:
                time_values.append(float(parts[1]))
            except ValueError:
                continue

    if len(time_values) >= 10:
        return time_values[9]  # 10th time (index 9)
    elif time_values:
        return time_values[-1]
    else:
        return None

def main():
    
    root_directory    = '/Users/Desktop/AGA-master/Input_Data/function2'
    results_directory = '/Users/Desktop/AGA METHOD OUTPUT'

    os.makedirs(results_directory, exist_ok=True)

    
    results_overview = []

    summary_html = []
    summary_html.append(f"""
    <html>
    <head><title>AGA Summary</title></head>
    <body>
    <h1>AGA (Greedy Additional) Summary</h1>
    <table border="1" cellpadding="5" cellspacing="0">
    <tr>
      <th>Project</th>
      <th>Initial Memory (MB)</th>
      <th>Final Memory (MB)</th>
      <th>AGA Execution Time (s)</th>
      <th>AGA APFD</th>
      <th>AGA NAPFD</th>
      <th>AGA PFD@{int(PFD_CUT_RATIO*100)}%</th>
    </tr>
    """)

    def extract_number(folder_name):
        match = re.search(r'(\d+)', folder_name)
        return int(match.group(1)) if match else 999999

    for folder in os.listdir(root_directory):
        project_path = os.path.join(root_directory, folder)
        if not os.path.isdir(project_path):
            continue

        test_list_file = os.path.join(project_path, "testList")

        
        coverage_file = find_coverage_file(project_path, COVERAGE_MODE)

        mutant_file    = os.path.join(project_path, "mutantKillMatrix")

        if not (os.path.exists(test_list_file) and
                coverage_file is not None and os.path.exists(coverage_file) and
                os.path.exists(mutant_file)):
            print(f"Skipping '{folder}' - required files not found.")
            continue

        print(f"\nProcessing project: {folder}")
        print(f"  Using coverage: {os.path.basename(coverage_file)} (mode={COVERAGE_MODE})")

        testlist         = read_file_lines(test_list_file)
        coveragelist     = read_file_lines(coverage_file)
        mutantkillmatrix = read_file_lines(mutant_file)

        project_results_path = os.path.join(results_directory, folder)
        os.makedirs(project_results_path, exist_ok=True)

        initial_memory = (
            asizeof.asizeof(testlist)
            + asizeof.asizeof(coveragelist)
        )

        st0 = time.time()
        used = [False] * len(coveragelist)

        sloc = 0
        forward_index = []
        for line in coveragelist:
            splits = line.strip().split()
            if not splits:
                forward_index.append([])
                continue
            mx_line = max(int(x) for x in splits)
            sloc = max(sloc, mx_line)
            forward_index.append([int(x) for x in splits])
        sloc += 1

        inverted_index = [[] for _ in range(sloc)]
        has_change = [False] * sloc
        for t_idx, line in enumerate(coveragelist):
            splits = line.strip().split()
            for stmt_str in splits:
                stmt_id = int(stmt_str)
                inverted_index[stmt_id].append(t_idx)

        total_count = [len(lst) for lst in forward_index]

        st = time.time()
        pre_time = st - st0

        ga_result = greedyadditional_new(
            testlist, coveragelist,
            total_count, used, has_change,
            forward_index, inverted_index,
            project_results_path
        )
        prioritize_time = time.time() - st

        final_seq_path = os.path.join(project_results_path, 'SequenceGAMethod_final.txt')
        with open(final_seq_path, 'w') as f:
            for testname in ga_result[0]:
                f.write(testname + "\n")

        time_adj_path = os.path.join(project_results_path, "TimeGAMethod_adjacencylist")
        with open(time_adj_path, 'w') as f:
            f.write(f"pre,{pre_time}\n")
            for index, intermediate_time in enumerate(ga_result[1]):
                f.write(f"{index+1},{intermediate_time - st}\n")
            f.write(f"{len(ga_result[1]) + 1},{prioritize_time}\n")

        extracted_time = extract_total_time_from_file(time_adj_path)
        if extracted_time is not None:
            total_time = extracted_time + pre_time
        else:
            total_time = prioritize_time + pre_time

        gc.collect()

        final_memory = (
            asizeof.asizeof(testlist)
            + asizeof.asizeof(coveragelist)
            + asizeof.asizeof(ga_result[0])
            + asizeof.asizeof(forward_index)
            + asizeof.asizeof(inverted_index)
            + asizeof.asizeof(used)
            + asizeof.asizeof(has_change)
            + asizeof.asizeof(total_count)
        )

        if len(ga_result[1]) >= 10:
            seq10_path = os.path.join(project_results_path, 'SequenceGAMethod_10.txt')
            if os.path.exists(seq10_path):
                seq_file_for_metrics = seq10_path
                print("SequenceGAMethod_10.txt is used for metrics")
            else:
                print("SequenceGAMethod_10.txt not found; using final sequence file instead.")
                seq_file_for_metrics = final_seq_path
        else:
            seq_file_for_metrics = final_seq_path

        apfd  = calculate_apfd(seq_file_for_metrics, testlist, mutantkillmatrix)
        napfd = calculate_napfd(seq_file_for_metrics, testlist, mutantkillmatrix)
        pfd25 = calculate_pfd(seq_file_for_metrics, testlist, mutantkillmatrix, cut_ratio=PFD_CUT_RATIO)

        initial_memory_mb = initial_memory / (1024**2)
        final_memory_mb   = final_memory / (1024**2)

        results_overview.append((
            folder,
            extract_number(folder),
            initial_memory_mb,
            final_memory_mb,
            total_time,
            apfd,
            napfd,
            pfd25
        ))

        print(f"  Initial Memory:    {initial_memory_mb:.2f} MB")
        print(f"  Final Memory:      {final_memory_mb:.2f} MB")
        print(f"  AGA Execution Time: {total_time:.4f} s")
        print(f"  AGA APFD:           {apfd if apfd is not None else 'N/A'}")
        print(f"  AGA NAPFD:          {napfd if napfd is not None else 'N/A'}")
        print(f"  PFD@{int(PFD_CUT_RATIO*100)}%:         {pfd25 if pfd25 is not None else 'N/A'}")

    results_overview.sort(key=lambda x: x[1])

    for (proj, _, initMB, finalMB, etime, apfd, napfd, pfd25) in results_overview:
        apfd_display  = f"{apfd:.4f}" if apfd is not None else "N/A"
        napfd_display = f"{napfd:.4f}" if napfd is not None else "N/A"
        pfd_display   = f"{pfd25:.4f}" if pfd25 is not None else "N/A"

        summary_html.append(f"""
        <tr>
          <td>{proj}</td>
          <td>{initMB:.2f}</td>
          <td>{finalMB:.2f}</td>
          <td>{etime:.4f}</td>
          <td>{apfd_display}</td>
          <td>{napfd_display}</td>
          <td>{pfd_display}</td>
        </tr>
        """)

    summary_html.append("</table></body></html>")

    out_html_path = os.path.join(results_directory, "AGA_summary.html")
    with open(out_html_path, 'w') as f:
        f.write("\n".join(summary_html))

    print(f"\nSummary HTML written to: {out_html_path}")

if __name__ == "__main__":
    main()
