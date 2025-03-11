#!/usr/bin/env python3
import subprocess
import re
import os
import matplotlib.pyplot as plt

# Regex to capture a line with elapsed_real_time.
# Expected format: "mergesort serial: {elapsed_real_time: 0.000805 s}"
time_pattern = re.compile(r"(serial|parallel).*elapsed_real_time:\s*([\d.]+)\s*s", re.IGNORECASE)

def run_benchmark_multi(exec_name: str, problem_size_range, thread_counts):
    """
    Run the executable for each problem size:
      - First, run sequentially (looking for "serial:" timing).
      - Then run parallel for each thread count (looking for "parallel:" timing).
      
    Returns:
      sequential_times: dict mapping problem size to sequential execution time.
      parallel_times: dict mapping thread_count -> { problem size -> parallel execution time }.
    """
    sequential_times = {}
    parallel_times = {tc: {} for tc in thread_counts}

    for N in problem_size_range:
        print(f"\n=== Problem Size: {N} for {exec_name} ===")
        
        # Run sequentially.
        print(f"\nRunning {exec_name} sequentially with problem size {N}:")
        result = subprocess.run(
            [exec_name, str(N)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        output_seq = result.stdout
        print(output_seq)
        t_serial = None
        for line in output_seq.splitlines():
            if "serial" in line.lower():
                match = time_pattern.search(line)
                if match and match.group(1).lower() == "serial":
                    t_serial = float(match.group(2))
                    break
        if t_serial is None:
            print(f"Warning: sequential time not found for N={N} in {exec_name}")
        else:
            sequential_times[N] = t_serial

        # Run parallel for each thread count.
        for tc in thread_counts:
            print(f"\nRunning {exec_name} with problem size {N} using {tc} threads:")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(tc)
            result = subprocess.run(
                [exec_name, str(N)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env
            )
            output_par = result.stdout
            print(output_par)
            t_parallel = None
            for line in output_par.splitlines():
                if "parallel" in line.lower():
                    match = time_pattern.search(line)
                    if match and match.group(1).lower() == "parallel":
                        t_parallel = float(match.group(2))
                        break
            if t_parallel is None:
                print(f"Warning: parallel time not found for N={N} using {tc} threads in {exec_name}")
            else:
                parallel_times[tc][N] = t_parallel

    return sequential_times, parallel_times

def plot_execution_times(sequential_times, parallel_times, algo_name, time_range, save_folder=None):
    """
    Plot execution time vs. problem size.
    Only includes problem sizes within time_range (tuple: (min, max)).
    The sequential curve is always plotted in black.
    The parallel curves are colored in a gradient.
    """
    plt.figure()
    # Filter problem sizes.
    problem_sizes = sorted([p for p in sequential_times.keys() if time_range[0] <= p <= time_range[1]])
    seq_vals = [sequential_times[N] for N in problem_sizes]
    plt.plot(problem_sizes, seq_vals, label="Sequential", marker="o", color="black")
    
    # Use a gradient colormap for parallel thread curves.
    parallel_keys = sorted(parallel_times.keys())
    cmap = plt.get_cmap("viridis")
    n_threads = len(parallel_keys)
    for idx, tc in enumerate(parallel_keys):
        color = cmap(idx / (n_threads - 1)) if n_threads > 1 else cmap(0)
        ps = sorted([p for p in parallel_times[tc].keys() if time_range[0] <= p <= time_range[1]])
        vals = [parallel_times[tc][p] for p in ps]
        plt.plot(ps, vals, label=f"Parallel ({tc} threads)", marker="o", color=color)
    
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Execution Time (s)")
    plt.title(f"Execution Time vs. Problem Size\n{algo_name}")
    plt.legend()
    plt.grid(True)
    
    if save_folder:
        filename = os.path.join(save_folder, f"{algo_name}_execution_time.png")
        plt.savefig(filename)
        print(f"Saved execution time graph to {filename}")

def plot_speedups(sequential_times, parallel_times, algo_name, speedup_range, save_folder=None):
    """
    Plot speedup (sequential_time / parallel_time) vs. problem size.
    Only includes problem sizes within speedup_range (tuple: (min, max)).
    The parallel curves are colored in a gradient.
    """
    plt.figure()
    problem_sizes = sorted([p for p in sequential_times.keys() if speedup_range[0] <= p <= speedup_range[1]])
    # Horizontal line at speedup=1 for reference.
    plt.axhline(1, color="black", linestyle="--", label="Serial Baseline")
    
    parallel_keys = sorted(parallel_times.keys())
    cmap = plt.get_cmap("viridis")
    n_threads = len(parallel_keys)
    for idx, tc in enumerate(parallel_keys):
        speedups = []
        ps = []
        for N in problem_sizes:
            if N in parallel_times[tc] and parallel_times[tc][N] != 0:
                speedups.append(sequential_times[N] / parallel_times[tc][N])
                ps.append(N)
        color = cmap(idx / (n_threads - 1)) if n_threads > 1 else cmap(0)
        plt.plot(ps, speedups, marker="o", label=f"{tc} threads", color=color)
    
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Speedup (Sequential / Parallel)")
    plt.title(f"Speedup vs. Problem Size\n{algo_name}")
    plt.legend()
    plt.grid(True)
    
    if save_folder:
        filename = os.path.join(save_folder, f"{algo_name}_speedup.png")
        plt.savefig(filename)
        print(f"Saved speedup graph to {filename}")

def plot_speedup_vs_threads(bench_speedup_data, save_folder=None):
    """
    Plot speedup vs. number of threads.
    For each benchmark, the representative speedup is computed at the maximum problem size.
    A horizontal dashed black line at speedup=1 is added as the serial baseline.
    bench_speedup_data should be a list of tuples:
      (algo_name, thread_counts (list), speedups (list))
    """
    plt.figure()
    # Serial baseline at speedup = 1.
    plt.axhline(1, color="black", linestyle="--", label="Serial Baseline")
    
    cmap = plt.get_cmap("plasma")
    n_bench = len(bench_speedup_data)
    for idx, (name, thread_list, speedup_list) in enumerate(bench_speedup_data):
        color = cmap(idx / (n_bench - 1)) if n_bench > 1 else cmap(0)
        plt.plot(thread_list, speedup_list, marker="o", label=name, color=color)
    
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (Sequential / Parallel)")
    plt.title("Speedup vs. Number of Threads")
    plt.legend()
    plt.grid(True)
    
    if save_folder:
        filename = os.path.join(save_folder, "speedup_vs_threads.png")
        plt.savefig(filename)
        print(f"Saved speedup vs. threads graph to {filename}")

def main():
    # Define thread counts to test.
    thread_counts = list(range(2, 13))
    
    # Define benchmarks with controllable ranges.
    benchmarks = [
        {
            "name": "mergesort",
            "executable": "./mergesort.run",
            "problem_range": range(2, 20),
            "speedup_range": (4, 19),
            "time_range": (12, 19)
        },
        {
            "name": "bubble",
            "executable": "./bubble.run",
            "problem_range": range(2, 12),
            "speedup_range": (2, 11),
            "time_range": (2, 11)
        },
        {
            "name": "odd-even",
            "executable": "./odd-even.run",
            "problem_range": range(2, 14),
            "speedup_range": (2, 13),
            "time_range": (2, 13)
        }
    ]
    
    # Create output folder (one directory up, in "plotting")
    save_folder = os.path.join("..", "plotting")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created folder: {save_folder}")
    
    # This list will hold data for the speedup vs. threads graph.
    bench_speedup_data = []
    
    # Loop over each benchmark.
    for bm in benchmarks:
        print(f"\n\n===== Running benchmark for {bm['name']} =====")
        sequential_times, parallel_times = run_benchmark_multi(bm["executable"], bm["problem_range"], thread_counts)
        plot_execution_times(sequential_times, parallel_times, bm["name"], bm["time_range"], save_folder)
        plot_speedups(sequential_times, parallel_times, bm["name"], bm["speedup_range"], save_folder)
        
        # For speedup vs. threads, choose representative problem size as the maximum in the range.
        rep_size = max(bm["problem_range"])
        speedup_vs_threads = []
        for tc in thread_counts:
            if rep_size in parallel_times[tc] and parallel_times[tc][rep_size] != 0:
                speedup_vs_threads.append(sequential_times[rep_size] / parallel_times[tc][rep_size])
            else:
                speedup_vs_threads.append(0)
        bench_speedup_data.append((bm["name"], thread_counts, speedup_vs_threads))
    
    # Plot speedup vs. number of threads (all benchmarks in one figure).
    plot_speedup_vs_threads(bench_speedup_data, save_folder)
    
    # Finally, show all plots.
    plt.show()

if __name__ == "__main__":
    main()
