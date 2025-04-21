#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

struct CSRMatrix {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    int rows, cols;

    CSRMatrix(int r, int c) : rows(r), cols(c) {
        row_ptrs.resize(r + 1, 0);
    }

    void from_dense(const float* dense, int rows, int cols, float threshold = 1e-10f) {
        row_ptrs[0] = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float val = dense[i * cols + j];
                if (std::abs(val) > threshold) {
                    values.push_back(val);
                    col_indices.push_back(j);
                }
            }
            row_ptrs[i + 1] = values.size();
        }
    }
};

struct CSCMatrix {
    std::vector<float> values;
    std::vector<int> row_indices;
    std::vector<int> col_ptrs;
    int rows, cols;

    CSCMatrix(int r, int c) : rows(r), cols(c) {
        col_ptrs.resize(c + 1, 0);
    }

    void from_dense(const float* dense, int rows, int cols, float threshold = 1e-10f) {
        std::vector<int> col_counts(cols, 0);
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                if (std::abs(dense[i * cols + j]) > threshold) {
                    col_counts[j]++;
                }
            }
        }

        col_ptrs[0] = 0;
        for (int j = 0; j < cols; j++) {
            col_ptrs[j + 1] = col_ptrs[j] + col_counts[j];
        }

        values.resize(col_ptrs[cols]);
        row_indices.resize(col_ptrs[cols]);

        std::vector<int> current_pos(cols, 0);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float val = dense[i * cols + j];
                if (std::abs(val) > threshold) {
                    int pos = col_ptrs[j] + current_pos[j];
                    values[pos] = val;
                    row_indices[pos] = i;
                    current_pos[j]++;
                }
            }
        }
    }
};

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        auto m1_dense = std::make_unique<float[]>(n * k);
        auto m2_dense = std::make_unique<float[]>(k * m);

        m1_fs.read(reinterpret_cast<char*>(m1_dense.get()), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2_dense.get()), sizeof(float) * k * m);
        m1_fs.close();
        m2_fs.close();

        CSRMatrix m1_sparse(n, k);
        m1_sparse.from_dense(m1_dense.get(), n, k);
        m1_dense.reset();

        CSCMatrix m2_sparse(k, m);
        m2_sparse.from_dense(m2_dense.get(), k, m);
        m2_dense.reset();

        auto result = std::make_unique<float[]>(n * m);
        float* __restrict res_ptr = result.get();
        std::fill(res_ptr, res_ptr + n * m, 0.0f);

        int num_threads = 64;
        omp_set_num_threads(num_threads);

        #pragma omp parallel for schedule(dynamic, 16)
        for (int idx = 0; idx < n * m; idx++) {
            int i = idx / m;
            int j = idx % m;

            float sum = 0.0f;

            int ptrA = m1_sparse.row_ptrs[i];
            int ptrB = m2_sparse.col_ptrs[j];

            const int endA = m1_sparse.row_ptrs[i + 1];
            const int endB = m2_sparse.col_ptrs[j + 1];

            while (ptrA < endA && ptrB < endB) {
                int colA = m1_sparse.col_indices[ptrA];
                int rowB = m2_sparse.row_indices[ptrB];

                if (ptrA + 4 < endA) {
                    _mm_prefetch(reinterpret_cast<const char*>(&m1_sparse.col_indices[ptrA + 4]), _MM_HINT_T0);
                }
                if (ptrB + 4 < endB) {
                    _mm_prefetch(reinterpret_cast<const char*>(&m2_sparse.row_indices[ptrB + 4]), _MM_HINT_T0);
                }

                if (colA < rowB) {
                    ptrA++;
                } else if (colA > rowB) {
                    ptrB++;
                } else {
                    sum += m1_sparse.values[ptrA] * m2_sparse.values[ptrB];
                    ptrA++;
                    ptrB++;
                }
            }

            res_ptr[idx] = sum;
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};