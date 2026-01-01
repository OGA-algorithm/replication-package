import os
import time
import heapq
import math
import gc
import re

import numpy as np
from pympler import asizeof
from pyroaring import BitMap  # pip install pyroaring


# =========================
# CONFIG
# =========================
PFD_CUT_RATIO = 0.25  # PFD@25%

# Choose coverage mode:
#   "statement" => reads state-map.txt
#   "function"  => reads function.txt (or common variants)
COVERAGE_MODE = "function"   # <-- change to "statement" when needed

COVERAGE_CANDIDATES = {
    "statement": ["state-map.txt"],
    "function": [
        "function.txt",
        "functions.txt",
        "function-map.txt",
        "functions-map.txt",
        "method.txt",
        "methods.txt",
        "method-map.txt",
        "methods-map.txt",
    ],
}


# =========================
# 1) COVERAGE PREPROCESSING + GROUPING
# =========================
def preprocess_coverage_info(coverage_info):
    return [BitMap(cov) for cov in coverage_info]


def group_test_cases_on_the_fly(test_cases, coverage_bitmaps, threshold=0.96, max_group_size=8):
   
    coverage_counts = [len(rb) for rb in coverage_bitmaps]

    groups = []
    group_anchors = []
    n = len(test_cases)
    chosen = [False] * n
    extra_test_cases = []

    cb = coverage_bitmaps
    cc = coverage_counts
    thr = threshold

    anchor_counts_np = np.empty(n, dtype=np.float32)
    num_anchors = 0

    for i in range(n):
        if chosen[i]:
            continue

        placed = False
        count_i = cc[i]
        curr_bitmap = cb[i]

        if num_anchors > 0:
            anchor_counts_arr = anchor_counts_np[:num_anchors]
            m = np.minimum(count_i, anchor_counts_arr)
            M = np.maximum(count_i, anchor_counts_arr)

            valid = m >= (thr * M)
            valid_indices = np.flatnonzero(valid)

            for idx_v in valid_indices:
                anchor_idx = group_anchors[idx_v]
                inter = len(curr_bitmap & cb[anchor_idx])
                union = count_i + cc[anchor_idx] - inter
                sim = inter / union if union > 0 else 0.0

                if sim >= thr:
                    if len(groups[idx_v]) < max_group_size:
                        groups[idx_v].append(i)
                        chosen[i] = True
                        placed = True
                        break
                    else:
                        extra_test_cases.append(i)
                        chosen[i] = True
                        placed = True
                        break

        if not placed:
            groups.append([i])
            group_anchors.append(i)
            anchor_counts_np[num_anchors] = count_i
            num_anchors += 1
            chosen[i] = True

    return groups, extra_test_cases


# =========================
# 2) PRIORITIZATION (LAZY GREEDY)
# =========================
def prioritize_test_cases(groups, selected_coverage_info):
    
    flat_candidates = []
    candidate_group = {}
    for group_id, group in enumerate(groups):
        for test_index in group:
            flat_candidates.append(test_index)
            candidate_group[test_index] = group_id

    n = len(selected_coverage_info)
    heap = []
    for i in range(n):
        heapq.heappush(heap, (-len(selected_coverage_info[i]), i))

    universe = BitMap()
    for rb in selected_coverage_info:
        universe |= rb

    covered = BitMap()
    prioritized_list = []

    while heap:
        if covered == universe:
            covered = BitMap()
            new_heap = []
            for (_, pos) in heap:
                heapq.heappush(new_heap, (-len(selected_coverage_info[pos]), pos))
            heap = new_heap

        neg_stored_score, pos = heapq.heappop(heap)
        rb = selected_coverage_info[pos]
        new_elems = rb.difference(covered)
        current_score = len(new_elems)

        if current_score != -neg_stored_score:
            heapq.heappush(heap, (-current_score, pos))
            continue

        original_test_index = flat_candidates[pos]
        group_id = candidate_group[original_test_index]
        prioritized_list.append((original_test_index, current_score, group_id))
        covered |= rb

    return prioritized_list


# =========================
# 3) METRICS: APFD / NAPFD / PFD
# =========================
def _compute_first_detection_positions(result_path, testname_testno_dict, testno_testkill_dict, mutantkillmatrix):
   
    sequence_file = os.path.join(result_path, "SequenceGAMethod_final.txt")
    if not os.path.exists(sequence_file):
        return [], []

    with open(sequence_file, "r") as f:
        test_sequence = [line.strip() for line in f.readlines() if line.strip()]

    if not mutantkillmatrix or len(mutantkillmatrix) == 0:
        return test_sequence, []

    m_faults = len(mutantkillmatrix[0])
    first_detect_pos = [None] * m_faults
    is_killed = [False] * m_faults

    for i, test_case in enumerate(test_sequence):
        if test_case not in testname_testno_dict:
            continue
        thistestno = testname_testno_dict[test_case]
        killlist = testno_testkill_dict.get(thistestno, [])
        for j in killlist:
            if 0 <= j < m_faults and not is_killed[j]:
                is_killed[j] = True
                first_detect_pos[j] = i + 1  # 1-based position

    return test_sequence, first_detect_pos


def calculate_apfd(result_path, testname_testno_dict, testno_testkill_dict, mutantkillmatrix):
   
    test_sequence, first_detect_pos = _compute_first_detection_positions(
        result_path, testname_testno_dict, testno_testkill_dict, mutantkillmatrix
    )
    n = len(test_sequence)
    if n == 0 or not first_detect_pos:
        return None

    m = len(first_detect_pos)
    detected_positions = [pos for pos in first_detect_pos if pos is not None]
    sum_tf = float(sum(detected_positions))

    return 1.0 - (sum_tf / float(n * m)) + (1.0 / float(2 * n))


def calculate_napfd(result_path, testname_testno_dict, testno_testkill_dict, mutantkillmatrix):
   
    test_sequence, first_detect_pos = _compute_first_detection_positions(
        result_path, testname_testno_dict, testno_testkill_dict, mutantkillmatrix
    )
    n = len(test_sequence)
    if n == 0 or not first_detect_pos:
        return None

    m = len(first_detect_pos)
    detected_positions = [pos for pos in first_detect_pos if pos is not None]
    md = len(detected_positions)
    p = md / m if m > 0 else 0.0

    sum_tf = float(sum(detected_positions))
    return p - (sum_tf / float(n * m)) + (p / float(2 * n))


def calculate_pfd(result_path, testname_testno_dict, testno_testkill_dict, mutantkillmatrix, cut_ratio=PFD_CUT_RATIO):
   
    test_sequence, first_detect_pos = _compute_first_detection_positions(
        result_path, testname_testno_dict, testno_testkill_dict, mutantkillmatrix
    )
    n = len(test_sequence)
    if n == 0 or not first_detect_pos:
        return None

    m = len(first_detect_pos)
    k = int(math.ceil(cut_ratio * n))
    k = max(1, min(k, n))

    detected_by_k = sum(1 for pos in first_detect_pos if (pos is not None and pos <= k))
    return detected_by_k / m if m > 0 else 0.0


# =========================
# 4) HTML OUTPUTS
# =========================
def generate_html_output(groups, prioritized_list, test_cases, total_time, test_case_names,
                         resultpath, apfd=None, napfd=None, pfd=None, pfd_cut_ratio=PFD_CUT_RATIO):
    """Per-project detailed HTML (kept)."""
    total_groups = len(groups)
    total_test_cases = len(test_cases)

    metrics_html = "<h2>Metrics</h2><ul>"
    if apfd is not None:
        metrics_html += f"<li><strong>APFD:</strong> {apfd:.6f}</li>"
    if napfd is not None:
        metrics_html += f"<li><strong>NAPFD:</strong> {napfd:.6f}</li>"
    if pfd is not None:
        metrics_html += f"<li><strong>PFD@{int(pfd_cut_ratio*100)}%:</strong> {pfd:.6f}</li>"
    metrics_html += "</ul>"

    html_content = f"""
    <html>
    <head>
        <title>Test Case Prioritization</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .group-table {{ margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <h1>Test Case Prioritization</h1>
        <p><strong>Total Number of Groups Generated:</strong> {total_groups}</p>
        <p><strong>Total Number of Test Cases:</strong> {total_test_cases}</p>
        <p><strong>Total Time Taken:</strong> {total_time:.6f} seconds</p>
        {metrics_html}

        <h2>Groups</h2>
    """

    for group_id, group in enumerate(groups):
        html_content += f"<h3>Group {group_id + 1}</h3>"
        html_content += "<table class='group-table'><tr><th>Test Case</th></tr>"
        for test_case_index in group:
            html_content += f"<tr><td>Test Case {test_case_index + 1}: {test_case_names[test_case_index]}</td></tr>"
        html_content += "</table>"

    html_content += "<h2>Prioritized Test Cases</h2>"
    html_content += "<table><tr><th>Test Case</th><th>New Elements Covered</th><th>Group</th></tr>"
    for test_case_index, new_count, group_id in prioritized_list:
        group_name = f"Group {group_id + 1}" if group_id != -1 else "No Group"
        html_content += (
            f"<tr><td>Test Case {test_case_index + 1}: {test_case_names[test_case_index]}</td>"
            f"<td>{new_count}</td><td>{group_name}</td></tr>"
        )
    html_content += "</table></body></html>"

    os.makedirs(resultpath, exist_ok=True)
    with open(os.path.join(resultpath, "test_case_prioritization.html"), "w") as f:
        f.write(html_content)


def generate_summary_html_report_without_iterations(results, result_directory, pfd_cut_ratio=PFD_CUT_RATIO):
    """
    Minimal summary HTML with ONLY these columns:
      Initial Memory Usage (MB)
      Final Memory Usage (MB)
      pre Time (seconds)
      prioritize_time (seconds)
      Execution Time (seconds)
      APFD Score
      NAPFD Score
      PFD@25%

    Sorted strictly by leading S<number>: S1 .. S54
    """
    def extract_s_number(name: str) -> int:
        m = re.match(r'^\s*S(\d+)\b', name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m2 = re.search(r'(\d+)', name)
        return int(m2.group(1)) if m2 else 999999

    results_sorted = sorted(results, key=lambda r: extract_s_number(r["folder"]))

    html = f"""
    <html>
    <head>
        <title>OGA Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>OGA Summary Report </h1>
        <table>
            <tr>
                <th>Project Folder</th>
                <th>Initial Memory Usage (MB)</th>
                <th>Final Memory Usage (MB)</th>
                <th>pre Time (seconds)</th>
                <th>prioritize_time (seconds)</th>
                <th>OGA Execution Time (seconds)</th>
                <th>OGA APFD Score</th>
                <th>OGA NAPFD Score</th>
                <th>OGA PFD@{int(pfd_cut_ratio*100)}%</th>
            </tr>
    """

    def fmt(x, nd):
        return "N/A" if x is None else f"{x:.{nd}f}"

    for r in results_sorted:
        html += f"""
            <tr>
                <td>{r['folder']}</td>
                <td>{fmt(r.get('initial_memory'), 2)}</td>
                <td>{fmt(r.get('final_memory'), 2)}</td>
                <td>{fmt(r.get('pre_time'), 4)}</td>
                <td>{fmt(r.get('prioritize_time'), 4)}</td>
                <td>{fmt(r.get('execution_time'), 4)}</td>
                <td>{fmt(r.get('apfd_score'), 4)}</td>
                <td>{fmt(r.get('napfd_score'), 4)}</td>
                <td>{fmt(r.get('pfd_score'), 4)}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    os.makedirs(result_directory, exist_ok=True)
    out_path = os.path.join(result_directory, "summary_report_without_iterations.html")
    with open(out_path, "w") as f:
        f.write(html)


# =========================
# 5) FILE READERS
# =========================
def get_file_paths(directory, coverage_mode="function"):
    """
    coverage_mode:
      - "statement" => state-map.txt
      - "function"  => function.txt (or common variants)
    """
    files = os.listdir(directory)
    file_paths = {}

   
    wanted = COVERAGE_CANDIDATES.get(coverage_mode, [])
    coverage_file = None

   
    for cand in wanted:
        for fn in files:
            if fn.lower() == cand.lower():
                coverage_file = os.path.join(directory, fn)
                break
        if coverage_file:
            break

    if coverage_file is None:
        for cand in wanted:
            for fn in files:
                if cand.lower() in fn.lower():
                    coverage_file = os.path.join(directory, fn)
                    break
            if coverage_file:
                break

    if coverage_file:
        file_paths["coverage_map"] = coverage_file

    for fn in files:
        if 'testList' in fn:
            file_paths['test_case_names'] = os.path.join(directory, fn)
        elif 'mutantKillMatrix' in fn:
            file_paths['mutant_kill_matrix'] = os.path.join(directory, fn)

    return file_paths


def read_test_cases_and_coverage_info(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    test_cases = [f"Test Case {i + 1}" for i in range(len(lines))]
    coverage_info = [list(map(int, line.split())) for line in lines]
    return test_cases, coverage_info


def read_test_case_names(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def readFile(filepath):
    with open(filepath) as f:
        return f.read().splitlines()


# =========================
# 6) MAIN
# =========================
def main():
    root_directory = '/Users/Desktop/AGA-master/Input_Data/function2'
    result_directory = '/Users/Desktop/OGA METHOD OUTPUT'

    all_results = []

    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder_name}")

        file_paths = get_file_paths(folder_path, coverage_mode=COVERAGE_MODE)
        if ('coverage_map' not in file_paths
            or 'test_case_names' not in file_paths
            or 'mutant_kill_matrix' not in file_paths):
            print(f"Required files not found in {folder_name}, skipping...")
            continue

        coverage_file = file_paths['coverage_map']
        print(f"Processing folder: {coverage_file}")
        test_case_names_file = file_paths['test_case_names']
        mutant_kill_matrix_file = file_paths['mutant_kill_matrix']

        resultpath = os.path.join(result_directory, folder_name)
        os.makedirs(resultpath, exist_ok=True)

        
        test_cases, coverage_info = read_test_cases_and_coverage_info(coverage_file)
        test_case_names = read_test_case_names(test_case_names_file)
        mutantkillmatrix = readFile(mutant_kill_matrix_file)

       
        testname_testno_dict = {test_case_names[i]: i for i in range(len(test_case_names))}
        testno_testkill_dict = {}
        for i in range(len(mutantkillmatrix)):
            line_str = mutantkillmatrix[i]
            killlist = [j for j in range(len(line_str)) if line_str[j] == '1']
            testno_testkill_dict[i] = killlist

       
        initial_memory_bytes = (
            asizeof.asizeof(test_cases)
            + asizeof.asizeof(coverage_info)
            + asizeof.asizeof(test_case_names)
        )

        st0 = time.time()

        
        coverage_bitmaps = preprocess_coverage_info(coverage_info)

  
        groups, extra_test_cases = group_test_cases_on_the_fly(test_cases, coverage_bitmaps)

        selected_coverage_info = [coverage_bitmaps[i] for group in groups for i in group]

        start_time = time.time()
        pre_time = start_time - st0

       
        prioritized_list = prioritize_test_cases(groups, selected_coverage_info)

        for test_case in sorted(extra_test_cases, key=lambda idx: len(coverage_bitmaps[idx]), reverse=True):
            prioritized_list.append((test_case, len(coverage_bitmaps[test_case]), -1))

        prioritize_time = time.time() - start_time
        total_time = prioritize_time + pre_time

        
        final_memory_bytes = (
            asizeof.asizeof(groups)
            + asizeof.asizeof(prioritized_list)
            + asizeof.asizeof(selected_coverage_info)
        )

        
        sequence_file = os.path.join(resultpath, "SequenceGAMethod_final.txt")
        with open(sequence_file, "w") as f:
            for test_case_index, _, _ in prioritized_list:
                f.write(f"{test_case_names[test_case_index]}\n")

        
        apfd = calculate_apfd(resultpath, testname_testno_dict, testno_testkill_dict, mutantkillmatrix)
        napfd = calculate_napfd(resultpath, testname_testno_dict, testno_testkill_dict, mutantkillmatrix)
        pfd = calculate_pfd(resultpath, testname_testno_dict, testno_testkill_dict, mutantkillmatrix, cut_ratio=PFD_CUT_RATIO)

       
        generate_html_output(
            groups=groups,
            prioritized_list=prioritized_list,
            test_cases=test_cases,
            total_time=total_time,
            test_case_names=test_case_names,
            resultpath=resultpath,
            apfd=apfd,
            napfd=napfd,
            pfd=pfd,
            pfd_cut_ratio=PFD_CUT_RATIO
        )

       
        all_results.append({
            "folder": folder_name,
            "initial_memory": initial_memory_bytes / (1024 ** 2),
            "final_memory": final_memory_bytes / (1024 ** 2),
            "pre_time": pre_time,
            "prioritize_time": prioritize_time,
            "execution_time": total_time,
            "apfd_score": apfd,
            "napfd_score": napfd,
            "pfd_score": pfd,
        })

        gc.collect()

    generate_summary_html_report_without_iterations(all_results, result_directory, pfd_cut_ratio=PFD_CUT_RATIO)
    print(f"\nDone. Summary saved to: {os.path.join(result_directory, 'summary_report_without_iterations.html')}")


if __name__ == "__main__":
    main()
