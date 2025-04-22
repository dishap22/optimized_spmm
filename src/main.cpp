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
#include <cmath>

struct CSRMatrix {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    int rows, cols;

    CSRMatrix(int r, int c) : rows(r), cols(c) {}

    void from_dense_parallel(const float* dense, int threshold_rows, int threshold_cols, float threshold = 1e-10f) {
        rows = threshold_rows;
        cols = threshold_cols;
        row_ptrs.resize(rows + 1);
        std::vector<std::vector<float>> temp_values(rows);
        std::vector<std::vector<int>> temp_indices(rows);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float val = dense[i * cols + j];
                if (std::abs(val) > threshold) {
                    temp_values[i].emplace_back(val);
                    temp_indices[i].emplace_back(j);
                }
            }
        }

        row_ptrs[0] = 0;
        for (int i = 0; i < rows; ++i) {
            row_ptrs[i + 1] = row_ptrs[i] + temp_values[i].size();
            values.insert(values.end(), temp_values[i].begin(), temp_values[i].end());
            col_indices.insert(col_indices.end(), temp_indices[i].begin(), temp_indices[i].end());
        }
    }
};

struct CSCMatrix {
    std::vector<float> values;
    std::vector<int> row_indices;
    std::vector<int> col_ptrs;
    int rows, cols;

    CSCMatrix(int r, int c) : rows(r), cols(c) {}

    void from_dense_parallel(const float* dense, int threshold_rows, int threshold_cols, float threshold = 1e-10f) {
        rows = threshold_rows;
        cols = threshold_cols;
        col_ptrs.resize(cols + 1);
        std::vector<std::vector<float>> temp_values(cols);
        std::vector<std::vector<int>> temp_indices(cols);

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                float val = dense[i * cols + j];
                if (std::abs(val) > threshold) {
                    temp_values[j].emplace_back(val);
                    temp_indices[j].emplace_back(i);
                }
            }
        }

        col_ptrs[0] = 0;
        for (int j = 0; j < cols; ++j) {
            col_ptrs[j + 1] = col_ptrs[j] + temp_values[j].size();
            values.insert(values.end(), temp_values[j].begin(), temp_values[j].end());
            row_indices.insert(row_indices.end(), temp_indices[j].begin(), temp_indices[j].end());
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
        m1_sparse.from_dense_parallel(m1_dense.get(), n, k);
        m1_dense.reset();

        CSCMatrix m2_sparse(k, m);
        m2_sparse.from_dense_parallel(m2_dense.get(), k, m);
        m2_dense.reset();

        auto result = std::make_unique<float[]>(n * m);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n * m; i++) {
            result[i] = 0.0f;
        }

        float* __restrict res_ptr = result.get();

        int num_threads = 64;
        omp_set_num_threads(num_threads);

        #pragma omp parallel for schedule(guided)
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

                if (ptrA + 4 < endA)
                    _mm_prefetch(reinterpret_cast<const char*>(&m1_sparse.col_indices[ptrA + 4]), _MM_HINT_T0);
                if (ptrB + 4 < endB)
                    _mm_prefetch(reinterpret_cast<const char*>(&m2_sparse.row_indices[ptrB + 4]), _MM_HINT_T0);

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

        sol_fs.write(reinterpret_cast<const char*>(res_ptr), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};
