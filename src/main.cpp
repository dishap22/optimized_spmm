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
#include <cmath>
#include <omp.h>
#include <unordered_map>

struct CSRMatrix {
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    int rows, cols;

    CSRMatrix(int r, int c) : rows(r), cols(c) {
        row_ptrs.resize(r + 1, 0);
    }

    void from_dense(const float* dense, int threshold = 1e-10f) {
        #pragma omp parallel
        {
            std::vector<float> local_vals;
            std::vector<int> local_cols;
            std::vector<int> local_row_ptrs(rows + 1, 0);

            #pragma omp for nowait
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float val = dense[i * cols + j];
                    if (std::fabs(val) > threshold) {
                        local_vals.push_back(val);
                        local_cols.push_back(j);
                        local_row_ptrs[i + 1]++;
                    }
                }
            }

            #pragma omp critical
            {
                int base = values.size();
                values.insert(values.end(), local_vals.begin(), local_vals.end());
                col_indices.insert(col_indices.end(), local_cols.begin(), local_cols.end());
                for (int i = 1; i <= rows; ++i)
                    row_ptrs[i] += local_row_ptrs[i];
            }
        }
        for (int i = 1; i <= rows; ++i)
            row_ptrs[i] += row_ptrs[i - 1];
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

    void from_dense(const float* dense, int threshold = 1e-10f) {
        std::vector<int> col_counts(cols, 0);
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                if (std::fabs(dense[i * cols + j]) > threshold) {
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
                if (std::fabs(val) > threshold) {
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
        m1_fs.close(); m2_fs.close();

        CSRMatrix A(n, k);
        CSCMatrix B(k, m);
        A.from_dense(m1_dense.get());
        B.from_dense(m2_dense.get());

        auto result = std::make_unique<float[]>(n * m);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n * m; ++i) result[i] = 0.0f;

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                float sum = 0.0f;
                int a_ptr = A.row_ptrs[i], a_end = A.row_ptrs[i + 1];
                int b_ptr = B.col_ptrs[j], b_end = B.col_ptrs[j + 1];
                while (a_ptr < a_end && b_ptr < b_end) {
                    int a_col = A.col_indices[a_ptr];
                    int b_row = B.row_indices[b_ptr];
                    if (a_col == b_row) {
                        sum += A.values[a_ptr] * B.values[b_ptr];
                        ++a_ptr; ++b_ptr;
                    } else if (a_col < b_row) {
                        ++a_ptr;
                    } else {
                        ++b_ptr;
                    }
                }
                result[i * m + j] = sum;
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};
