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

    constexpr int TILE_SIZE = 64;

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

        #pragma omp parallel for collapse(2) schedule(dynamic, 4) num_threads(16)
        for (int bi = 0; bi < n; bi += TILE_SIZE) {
            for (int bj = 0; bj < m; bj += TILE_SIZE) {

                for (int i = bi; i < std::min(bi + TILE_SIZE, n); ++i) {
                    float* out_row = res + i * m;
                    std::fill(out_row + bj, out_row + std::min(bj + TILE_SIZE, m), 0.0f);

                    int row_start_A = m1_csr.row_ptrs[i];
                    int row_end_A   = m1_csr.row_ptrs[i + 1];

                    for (int p = row_start_A; p < row_end_A; ++p) {
                        int colA = m1_csr.col_indices[p];
                        float valA = m1_csr.values[p];

                        int row_start_B = m2t_csr.row_ptrs[colA];
                        int row_end_B = m2t_csr.row_ptrs[colA + 1];

                        for (int q = row_start_B; q < row_end_B; ++q) {
                            int colB = m2t_csr.col_indices[q];
                            if (colB >= bj && colB < bj + TILE_SIZE) {
                                out_row[colB] += valA * m2t_csr.values[q];
                            }
                        }
                    }
                }
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};
