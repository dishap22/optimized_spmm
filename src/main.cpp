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
#include <unordered_map>
#include <omp.h>

namespace solution {

    struct SparseRow {
        std::vector<int> cols;
        std::vector<float> vals;
    };

    std::vector<SparseRow> denseToCSR(const float* mat, int rows, int cols, float epsilon = 1e-6f) {
        std::vector<SparseRow> csr(rows);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < rows; ++i) {
            SparseRow row;
            for (int j = 0; j < cols; ++j) {
                float val = mat[i * cols + j];
                if (std::abs(val) > epsilon) {
                    row.cols.push_back(j);
                    row.vals.push_back(val);
                }
            }
            csr[i] = std::move(row);
        }
        return csr;
    }

    std::vector<std::unordered_map<int, float>> transposeAndCSR(const float* mat, int rows, int cols, float epsilon = 1e-6f) {
        std::vector<std::unordered_map<int, float>> transposed(cols);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float val = mat[i * cols + j];
                if (std::abs(val) > epsilon) {
                    #pragma omp critical
                    transposed[j][i] = val;
                }
            }
        }
        return transposed;
    }

    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        std::unique_ptr<float[]> m1 = std::make_unique<float[]>(n * k);
        std::unique_ptr<float[]> m2 = std::make_unique<float[]>(k * m);
        m1_fs.read(reinterpret_cast<char*>(m1.get()), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2.get()), sizeof(float) * k * m);
        m1_fs.close(); m2_fs.close();

        auto m1_csr = denseToCSR(m1.get(), n, k);
        auto m2_csc = transposeAndCSR(m2.get(), k, m);

        std::unique_ptr<float[]> result = std::make_unique<float[]>(n * m);
        std::fill(result.get(), result.get() + n * m, 0.0f);

        #pragma omp parallel for schedule(dynamic) num_threads(64)
        for (int i = 0; i < n; ++i) {
            const auto& row = m1_csr[i];
            for (int col = 0; col < m; ++col) {
                float sum = 0.0f;
                const auto& m2_row = m2_csc[col];
                for (size_t idx = 0; idx < row.cols.size(); ++idx) {
                    int k_index = row.cols[idx];
                    auto it = m2_row.find(k_index);
                    if (it != m2_row.end()) {
                        sum += row.vals[idx] * it->second;
                    }
                }
                result[i * m + col] = sum;
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
}
