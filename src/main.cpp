#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,avx512dq,avx512vl,avx512bw,bmi,bmi2,lzcnt,popcnt")

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
            std::vector<std::pair<int, float>> row;
            for (size_t j = 0; j < temp_vals[i].size(); ++j)
                row.emplace_back(temp_idx[i][j], temp_vals[i][j]);
            std::sort(row.begin(), row.end());

            for (auto& p : row) {
                col_indices.push_back(p.first);
                values.push_back(p.second);
            }

            row_ptrs[i + 1] = row_ptrs[i] + static_cast<int>(row.size());
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
            std::vector<std::pair<int, float>> row;
            for (size_t j = 0; j < temp_vals[i].size(); ++j)
                row.emplace_back(temp_idx[i][j], temp_vals[i][j]);
            std::sort(row.begin(), row.end());

            for (auto& p : row) {
                col_indices.push_back(p.first);
                values.push_back(p.second);
            }

            row_ptrs[i + 1] = row_ptrs[i] + static_cast<int>(row.size());
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

        #pragma omp parallel for
        for (int i = 0; i < n * k; ++i) m1_dense[i] = 0;
        #pragma omp parallel for
        for (int i = 0; i < k * m; ++i) m2_dense[i] = 0;

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

        #pragma omp parallel for collapse(2) schedule(guided, 1) num_threads(64)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                int row_start_A = m1_csr.row_ptrs[i];
                int row_end_A = m1_csr.row_ptrs[i + 1];
                int row_start_B = m2t_csr.row_ptrs[j];
                int row_end_B = m2t_csr.row_ptrs[j + 1];

                __m512 sum = _mm512_setzero_ps();

                int ptrA = row_start_A, ptrB = row_start_B;

                while (ptrA < row_end_A && ptrB < row_end_B) {
                    int colA = m1_csr.col_indices[ptrA];
                    int colB = m2t_csr.col_indices[ptrB];

                    if (colA < colB) {
                        ++ptrA;
                    } else if (colA > colB) {
                        ++ptrB;
                    } else {
                        __m512 valA = _mm512_set1_ps(m1_csr.values[ptrA]);
                        __m512 valB = _mm512_set1_ps(m2t_csr.values[ptrB]);

                        sum = _mm512_fmadd_ps(valA, valB, sum);

                        ++ptrA;
                        ++ptrB;
                    }
                }


                float row_sum[16];
                _mm512_storeu_ps(row_sum, sum);
                res[i * m + j] = 0.0f;
                for (int idx = 0; idx < 16; ++idx) {
                    res[i * m + j] += row_sum[idx];
                }
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};
