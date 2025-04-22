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
#include <unordered_map>
#include <cmath>

struct CSRMatrix {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    int rows, cols;

    CSRMatrix(int r, int c) : rows(r), cols(c) {}

    void from_dense_parallel(const float* dense, float threshold = 1e-10f) {
        row_ptrs.resize(rows + 1);
        std::vector<std::vector<float>> temp_vals(rows);
        std::vector<std::vector<int>> temp_idx(rows);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float val = dense[i * cols + j];
                if (std::abs(val) > threshold) {
                    temp_vals[i].push_back(val);
                    temp_idx[i].push_back(j);
                }
            }
        }

        row_ptrs[0] = 0;
        for (int i = 0; i < rows; ++i) {
            row_ptrs[i + 1] = row_ptrs[i] + temp_vals[i].size();
            values.insert(values.end(), temp_vals[i].begin(), temp_vals[i].end());
            col_indices.insert(col_indices.end(), temp_idx[i].begin(), temp_idx[i].end());
        }
    }

    void from_transpose_dense_parallel(const float* dense, float threshold = 1e-10f) {
        row_ptrs.resize(cols + 1);
        std::vector<std::vector<float>> temp_vals(cols);
        std::vector<std::vector<int>> temp_idx(cols);

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                float val = dense[i * cols + j];
                if (std::abs(val) > threshold) {
                    temp_vals[j].push_back(val);
                    temp_idx[j].push_back(i);
                }
            }
        }

        row_ptrs[0] = 0;
        for (int i = 0; i < cols; ++i) {
            row_ptrs[i + 1] = row_ptrs[i] + temp_vals[i].size();
            values.insert(values.end(), temp_vals[i].begin(), temp_vals[i].end());
            col_indices.insert(col_indices.end(), temp_idx[i].begin(), temp_idx[i].end());
        }
        std::swap(rows, cols);
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

        CSRMatrix m1_csr(n, k);
        m1_csr.from_dense_parallel(m1_dense.get());

        CSRMatrix m2t_csr(m, k);
        m2t_csr.from_transpose_dense_parallel(m2_dense.get());

        auto result = std::make_unique<float[]>(n * m);
        float* __restrict res = result.get();

        #pragma omp parallel num_threads(16)
        {
            std::vector<float> tmp(m);
            float* tmp_ptr = tmp.data();
            const int m_aligned = m - (m % 8);

            #pragma omp for schedule(static, 4)
            for (int i = 0; i < n; ++i) {
                int j = 0;
                for (; j + 7 < m; j += 8) {
                    _mm256_storeu_ps(tmp_ptr + j, _mm256_setzero_ps());
                }
                for (; j < m; ++j) {
                    tmp_ptr[j] = 0.0f;
                }

                const int row_start = m1_csr.row_ptrs[i];
                const int row_end = m1_csr.row_ptrs[i + 1];

                for (int ptrA = row_start; ptrA < row_end; ++ptrA) {
                    const int k = m1_csr.col_indices[ptrA];
                    const float valA = m1_csr.values[ptrA];

                    const int bt_row_start = m2t_csr.row_ptrs[k];
                    const int bt_row_end = m2t_csr.row_ptrs[k + 1];

                    for (int ptrB = bt_row_start; ptrB < bt_row_end; ++ptrB) {
                        const int j = m2t_csr.col_indices[ptrB];
                        tmp_ptr[j] += valA * m2t_csr.values[ptrB];
                    }
                }

                float* out_row = res + i * m;
                j = 0;
                for (; j + 7 < m; j += 8) {
                    _mm256_storeu_ps(out_row + j, _mm256_loadu_ps(tmp_ptr + j));
                }
                for (; j < m; ++j) {
                    out_row[j] = tmp_ptr[j];
                }
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(res), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};