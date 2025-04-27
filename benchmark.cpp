#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include "src/main.cpp"

void generate_matrix(const std::string& filename, int rows, int cols, float sparsity = 0.999f) {
    std::ofstream out(filename, std::ios::binary);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist_value(0.1f, 10.0f);
    std::uniform_real_distribution<float> dist_sparse(0.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        float val = (dist_sparse(rng) < sparsity) ? 0.0f : dist_value(rng);
        out.write(reinterpret_cast<char*>(&val), sizeof(float));
    }
    out.close();
}

int main() {
    int n = 2048, k = 2048, m = 2048;
    std::string m1 = "matrixA.dat";
    std::string m2 = "matrixB.dat";

    std::cout << "Generating matrices..." << std::endl;
    generate_matrix(m1, n, k); // A: n x k
    generate_matrix(m2, k, m); // B: k x m
    // m1 = "../matrixA.dat";
    // m2 = "../matrixB.dat";
    std::cout << "Running computation..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::string output = solution::compute(m1, m2, n, k, m);
    auto end = std::chrono::high_resolution_clock::now();

    double time_sec = std::chrono::duration<double>(end - start).count();
    std::cout << "Time taken: " << time_sec << " seconds\n";
    std::cout << "Output file: " << output << std::endl;

    return 0;
}
