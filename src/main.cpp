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
        m1_sparse.from_dense(m1_dense.get(), n, k);

        auto result = std::make_unique<float[]>(n * m);
        std::fill(result.get(), result.get() + n * m, 0.0f);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            for (int ptr = m1_sparse.row_ptrs[i]; ptr < m1_sparse.row_ptrs[i + 1]; ptr++) {
                float val = m1_sparse.values[ptr];
                int l = m1_sparse.col_indices[ptr];

                for (int j = 0; j < m; j += 8) {
                    if (j + 8 <= m) {
                        __m256 val_vec = _mm256_set1_ps(val);
                        __m256 m2_vec = _mm256_loadu_ps(&m2_dense[l * m + j]);
                        __m256 res_vec = _mm256_loadu_ps(&result[i * m + j]);

                        res_vec = _mm256_fmadd_ps(val_vec, m2_vec, res_vec);
                        _mm256_storeu_ps(&result[i * m + j], res_vec);
                    }
                    else {
                        for (int jj = j; jj < m; jj++) {
                            result[i * m + jj] += val * m2_dense[l * m + jj];
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