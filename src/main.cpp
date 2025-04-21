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

    void from_dense(const float* dense, int rows, int cols, float threshold = 1e-6f) {
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

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        const auto m1_dense = std::make_unique<float[]>(n * k);
        const auto m2_dense = std::make_unique<float[]>(k * m);

        m1_fs.read(reinterpret_cast<char*>(m1_dense.get()), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2_dense.get()), sizeof(float) * k * m);
        m1_fs.close();
        m2_fs.close();

        CSRMatrix m1_sparse(n, k);
        CSRMatrix m2_sparse(k, m);

        m1_sparse.from_dense(m1_dense.get(), n, k);
        m2_sparse.from_dense(m2_dense.get(), k, m);

        auto result = std::make_unique<float[]>(n * m);
        std::fill(result.get(), result.get() + n * m, 0.0f);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int ptr = m1_sparse.row_ptrs[i]; ptr < m1_sparse.row_ptrs[i + 1]; ptr++) {
                float val = m1_sparse.values[ptr];
                int l = m1_sparse.col_indices[ptr];

                for (int jptr = m2_sparse.row_ptrs[l]; jptr < m2_sparse.row_ptrs[l + 1]; jptr++) {
                    int b_col = m2_sparse.col_indices[jptr];
                    float b_val = m2_sparse.values[jptr];

                    result[i * m + b_col] += val * b_val;
                }
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};
