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
#include <cstring>

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

        CSRMatrix A(n, k);
        A.from_dense_parallel(m1_dense.get());

        CSRMatrix BT(m, k);
        BT.from_transpose_dense_parallel(m2_dense.get());

        auto result = std::make_unique<float[]>(n * m);
        float* __restrict res = result.get();

        #pragma omp parallel
        {
            std::vector<float> local_accum(m, 0.0f);
            std::vector<char> mask(m, 0);

            #pragma omp for schedule(static)
            for (int i = 0; i < A.rows; ++i) {
                float* out_row = res + i * m;
                std::fill(local_accum.begin(), local_accum.end(), 0.0f);
                std::fill(mask.begin(), mask.end(), 0);

                int startA = A.row_ptrs[i];
                int endA = A.row_ptrs[i + 1];

                for (int idxA = startA; idxA < endA; ++idxA) {
                    int colA = A.col_indices[idxA];
                    float valA = A.values[idxA];

                    int startB = BT.row_ptrs[colA];
                    int endB = BT.row_ptrs[colA + 1];

                    for (int idxB = startB; idxB < endB; ++idxB) {
                        int j = BT.col_indices[idxB];
                        float valB = BT.values[idxB];

                        if (!mask[j]) {
                            mask[j] = 1;
                            local_accum[j] = valA * valB;
                        } else {
                            local_accum[j] += valA * valB;
                        }
                    }
                }

                for (int j = 0; j < m; ++j) {
                    out_row[j] = local_accum[j];
                }
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
}
