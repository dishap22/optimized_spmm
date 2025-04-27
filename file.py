import os
import ssl
import tarfile
import tempfile
import subprocess
import time

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread
import requests
from tqdm import tqdm
from tabulate import tabulate

# ----------------------------
# 1) SETUP & CONFIGURATION
# ----------------------------
ssl._create_default_https_context = ssl._create_unverified_context

# SuiteSparse Matrix Market archives
BASE_URL = 'https://sparse.tamu.edu/MM/'
# List of Group/Name tarballs to download
DATASETS = [
    'HB/1138_bus.tar.gz',
    'HB/494_bus.tar.gz',
    'HB/662_bus.tar.gz',
    'HB/jagmesh7.tar.gz',
    'HB/blckhole.tar.gz',
    'CPM/cz5108.tar.gz',
    'VDOL/hangGlider_3.tar.gz',
    'Gaertner/big.tar.gz',

]

# Directory for SuiteSparse downloads
SS_DIR = 'SuiteSparse'
os.makedirs(SS_DIR, exist_ok=True)

# ----------------------------
# 2) DOWNLOAD SUITESPARSE DATASETS
# ----------------------------

def download_tamu():
    print("\n[DOWNLOAD] SuiteSparse matrices")
    for relpath in DATASETS:
        fname = os.path.basename(relpath)
        dst = os.path.join(SS_DIR, fname)
        if os.path.exists(dst):
            print(f"  EXISTS {fname}")
            continue
        url = BASE_URL + relpath
        print(f"  FETCH {fname} ...")
        try:
            with requests.get(url, stream=True, verify=False) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(dst, 'wb') as f, tqdm(
                    total=total, unit='B', unit_scale=True, desc=fname, ncols=70
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            print(" DONE")
        except Exception as e:
            print(" FAIL:", e)

# ----------------------------
# 3) LOAD & SAVE MATRICES
# ----------------------------

def load_tamu_adj(tar_path: str) -> sp.csr_matrix:
    """
    Unpack a .tar.gz from SuiteSparse, read the .mtx inside,
    and return a CSR adjacency (float32).
    """
    with tarfile.open(tar_path, 'r:gz') as tar:
        # find .mtx member
        mtx_members = [m for m in tar.getmembers() if m.name.endswith('.mtx')]
        if not mtx_members:
            raise RuntimeError("No .mtx file found in " + tar_path)
        member = mtx_members[0]
        f = tar.extractfile(member)
        M = mmread(f).astype(np.float32)
        A = sp.csr_matrix(M)
    print(f"[INFO] Loaded {os.path.basename(tar_path)} -> shape {A.shape}, nnz={A.nnz}")
    return A


def save_bin(mat: np.ndarray, path: str):
    mat.astype(np.float32).tofile(path)

# ----------------------------
# 4) RUN BENCHMARKS
# ----------------------------

def run_exec(exec_path, W_path, WT_path, n, k, repeat=1):
    # This function seems designed for the old tester/benchmark which took dense files.
    # It's not directly used by the modified SPMM logic which calls benchmark with --mtx.
    # Keep it for now in case it's needed elsewhere, or remove if confirmed unused.
    times = []
    for _ in range(repeat):
        start = time.time()
        try:
            p = subprocess.run(
                [exec_path, W_path, WT_path, str(n), str(k), str(n)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=100  # 100 seconds timeout per run
            )
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {exec_path} exceeded 100 seconds and was terminated.")
            return None
        if p.returncode == 0:
            times.append(time.time() - start)
        else:
            print(f"  ERROR: {exec_path} exited with code {p.returncode}.")
            print(f"    STDERR: {p.stderr.decode().strip()}")
            return None
    return sum(times) / len(times) if times else None

def benchmark_all(spmm_exec='./spmm_exec'):
    results = []
    for relpath in DATASETS:
        fname = os.path.basename(relpath)
        tar = os.path.join(SS_DIR, fname)
        print(f"\n[BENCH] {fname}")
        try:
            A = load_tamu_adj(tar)
        except Exception as e:
            print("  LOAD FAIL:", e)
            results.append({'File': fname, 'Dimensions': 'N/A', 'Nodes': 'N/A', 'Edges': 'N/A', 'Sparsity (%)': 'N/A', 'SPMM (s)': 'LOAD_ERR'})
            continue

        n = A.shape[0]
        k = A.shape[1] # Use k from A's shape
        nonzeros = A.nnz
        total = n * k
        sparsity = 100 * (1 - nonzeros / total) if total > 0 else 0

        # sparse phase (SPMM)
        t_sp = None # Initialize t_sp
        with tempfile.TemporaryDirectory() as td:
            # Prepare inputs for benchmark executable (assuming it takes CSR A and dense B)
            # Path for CSR matrix A
            csr_A_path = os.path.join(td, 'matrix_A.csr.dat')
            # Path for dense matrix B
            dense_B_path = os.path.join(td, 'matrix_B.dat')

            # Save A in CSR format (using the C++ benchmark's expected format)
            # This requires calling the benchmark executable in a mode to convert/save, or reimplementing save_mtx_to_csr in Python
            # For now, let's assume the benchmark executable handles the MTX file directly
            # We need the original MTX path from the tarball
            mtx_path_in_tar = None
            with tarfile.open(tar, 'r:gz') as tar_f:
                mtx_members = [m for m in tar_f.getmembers() if m.name.endswith('.mtx')]
                if mtx_members:
                    member = mtx_members[0]
                    # Extract the MTX file to the temporary directory
                    extracted_mtx_path = os.path.join(td, os.path.basename(member.name))
                    with tar_f.extractfile(member) as src, open(extracted_mtx_path, 'wb') as dst:
                        dst.write(src.read())
                    mtx_path_in_tar = extracted_mtx_path

            if mtx_path_in_tar:
                # Generate a dense matrix B (k x m, let's use m=n for simplicity for now)
                m_dim = n # Set m dimension, e.g., equal to n
                B = np.random.rand(k, m_dim).astype(np.float32)
                save_bin(B, dense_B_path)

                # Run the benchmark executable using the --mtx mode
                # Command: ./benchmark --mtx <mtx_file> <k_B> <m_B>
                # Note: The benchmark code internally converts MTX to CSR
                # We pass k and m for matrix B
                cmd = [spmm_exec, '--mtx', mtx_path_in_tar, str(k), str(m_dim)]
                print(f"  Running command: {' '.join(cmd)}")
                start_time = time.time()
                try:
                    p = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300 # Increased timeout
                    )
                    run_time = time.time() - start_time
                    if p.returncode == 0:
                        t_sp = run_time
                        # Try to parse time from output if available (optional)
                        output = p.stdout.decode()
                        for line in output.splitlines():
                            if "Execution time:" in line:
                                try:
                                    t_sp = float(line.split(':')[1].strip().split()[0]) / 1000.0 # Convert ms to s
                                    break
                                except Exception:
                                    pass # Keep the measured time if parsing fails
                    else:
                        print(f"  SPMM ERROR: {spmm_exec} exited with code {p.returncode}.")
                        print(f"    STDOUT: {p.stdout.decode().strip()}")
                        print(f"    STDERR: {p.stderr.decode().strip()}")
                except subprocess.TimeoutExpired:
                    print(f"  SPMM TIMEOUT: {spmm_exec} exceeded 300 seconds.")
                except Exception as run_e:
                    print(f"  SPMM RUN FAILED: {run_e}")
            else:
                print("  SPMM SKIP: Could not find or extract .mtx file from tarball.")

            print(f"  SPMM: {t_sp * 1000:.2f} ms" if t_sp is not None else "SPMM: ERR")

        # Updated results dictionary - removed GEMM fields
        results.append({
            'File': fname,
            'Dimensions': f"{n}x{k}", # Use n x k for A
            'Nodes': n, # Assuming n represents nodes
            'Edges': nonzeros, # Use raw nonzeros count
            'Sparsity (%)': f"{sparsity:.2f}",
            'SPMM (ms)': f"{t_sp * 1000:.2f}" if t_sp is not None else 'ERR',
        })

    print("\n=== RESULTS ===")
    # Updated headers for tabulate
    print(tabulate(results, headers={'File': 'File', 'Dimensions': 'Dimensions', 'Nodes': 'Nodes', 'Edges': 'Edges', 'Sparsity (%)': 'Sparsity (%)', 'SPMM (ms)': 'SPMM (ms)'}, tablefmt="github"))

# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == '__main__':
    download_tamu()
    print("\n=== DATASET DETAILS ===")
    for rel in DATASETS:
        path = os.path.join(SS_DIR, os.path.basename(rel))
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        print(f"  {os.path.basename(rel)}: {size_mb:.2f} MB")

    print("\nCompiling C++ code...")
    # use same paths and compilation steps as before
    # codes_dir = os.path.join(os.path.dirname(__file__), "Codes") # Incorrect path
    project_root = os.path.dirname(__file__) # Get the directory where the script is located
    build_dir = os.path.join(project_root, "build") # Use the existing build directory
    os.makedirs(build_dir, exist_ok=True)
    cmake_cmd = ["cmake", ".."]
    make_cmd = ["make", "-j"] # Use parallel make for faster compilation
    try:
        print(f"  Running cmake in {build_dir}...")
        subprocess.run(cmake_cmd, cwd=build_dir, check=True)
        print(f"  Running make in {build_dir}...")
        subprocess.run(make_cmd, cwd=build_dir, check=True)
        print("  Compilation successful.")
    except subprocess.CalledProcessError as e:
        print("Compilation failed:", e)
        exit(1)

    # Call benchmark_all without gemm_exec
    benchmark_all(
        spmm_exec=os.path.join(build_dir, "bin", "benchmark")
    )