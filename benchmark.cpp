#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include "src/main.cpp"

bool convert_mm_to_dat(const std::string& mm_filename, const std::string& dat_filename, int& rows, int& cols) {
    std::ifstream mm_file(mm_filename);
    if (!mm_file.is_open()) {
        std::cerr << "Error: Could not open Matrix Market file: " << mm_filename << std::endl;
        return false;
    }

    // Skip comments
    std::string line;
    do {
        if (!std::getline(mm_file, line)) {
            std::cerr << "Error: Invalid Matrix Market file format" << std::endl;
            return false;
        }
    } while (line[0] == '%');

    // Parse header
    std::istringstream iss(line);
    std::string format, object, field, symmetry;
    int num_entries;
    if (!(iss >> rows >> cols >> num_entries)) {
        std::cerr << "Error: Failed to parse Matrix Market header dimensions" << std::endl;
        return false;
    }

    // Create a dense matrix initialized with zeros
    std::vector<float> matrix(rows * cols, 0.0f);

    // Read entries
    int row, col;
    float value;
    while (std::getline(mm_file, line)) {
        // Skip empty lines
        if (line.empty() || std::all_of(line.begin(), line.end(), [](char c) { return std::isspace(c); }))
            continue;

        std::istringstream entry_iss(line);
        if (!(entry_iss >> row >> col >> value)) {
            std::cerr << "Error: Failed to parse Matrix Market entry" << std::endl;
            return false;
        }

        // Adjust for 1-based indexing in Matrix Market format
        row--;
        col--;

        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cerr << "Error: Invalid indices in Matrix Market file: " << (row + 1) << ", " << (col + 1) << std::endl;
            return false;
        }

        // Store value in dense matrix
        matrix[row * cols + col] = value;
    }

    // Write to binary .dat file
    std::ofstream dat_file(dat_filename, std::ios::binary);
    if (!dat_file.is_open()) {
        std::cerr << "Error: Could not open output .dat file: " << dat_filename << std::endl;
        return false;
    }

    dat_file.write(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(float));
    dat_file.close();

    std::cout << "Successfully converted " << mm_filename << " to " << dat_filename << std::endl;
    std::cout << "Matrix dimensions: " << rows << " x " << cols << std::endl;
    return true;
}

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
    std::string sparsesuit_path = "bcsstk13.mtx";
    std::string m3 = "matrixC.dat";

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

    int rowsA, colsA;
    if (!convert_mm_to_dat(sparsesuit_path, m3, rowsA, colsA)) {
        std::cerr << "Error converting matrixA.mtx" << std::endl;
        return 1;
    }

    std::cout << "Multiplying sparse suit with itself..." << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();
    std::string output2 = solution::compute(m1, m1, rowsA, colsA, rowsA);
    auto end2 = std::chrono::high_resolution_clock::now();
    double time_sec2 = std::chrono::duration<double>(end2 - start2).count();
    std::cout << "Time taken: " << time_sec2 << " seconds\n";
    std::cout << "Output file: " << output2 << std::endl;


    return 0;
}
