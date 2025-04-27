#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <limits> // Required for numeric_limits
#include <algorithm> // Required for std::fill, std::max
#include <set> // Required for finding max node ID efficiently
#include <tuple> // Required for std::tie in the new function

#include <studentlib.h> // Include the header for solution::compute

// --- Type Definitions and Constants (matching src/main.cpp) ---
using RowPtrType = int64_t; // Type for row pointers and NNZ count in CSR
constexpr float ZERO_EPSILON = 1e-9f; // Threshold for considering a float as zero

// --- Helper Functions ---

// Function to print usage instructions
void print_usage() {
    std::cerr << "Usage:\\n";
    std::cerr << "  ./benchmark --mtx <mtx_file_path> <k> <m> [seed]\\n";
    std::cerr << "  ./benchmark --graph <graph_file_path> <m> [seed]\\n"; // k determined by graph
    std::cerr << "  ./benchmark --generate <n> <k> <m> <sparsity> [seed]\\n";
    std::cerr << "Arguments:\\n";
    std::cerr << "  --mtx <mtx_file_path>: Path to the Matrix Market file for sparse matrix A.\\n";
    std::cerr << "  --graph <graph_file_path>: Path to the graph file (SNAP format) for sparse matrix A.\\n";
    std::cerr << "                         Matrix A will be NxN (adjacency), value 1.0 for edges.\\n";
    std::cerr << "  <n>, <k>, <m>: Dimensions of the matrices (A: n x k, B: k x m).\\n";
    std::cerr << "                 For --graph, n=k=number_of_nodes.\\n";
    std::cerr << "  <sparsity>: Target sparsity for randomly generated matrix A (e.g., 0.01 for 1%).\\n";
    std::cerr << "  [seed]: Optional random seed (integer).\\n";
}

// Function to generate a random float
float generate_random_float(std::mt19937& rng) {
    static std::uniform_real_distribution<float> distribution(1.0, 10.0); // Use a smaller range for potentially better numerical stability
    return distribution(rng);
}

// Function to save a dense matrix to a binary file
void save_dense_matrix(const float* matrix, size_t rows, size_t cols, const std::string& file_path) {
    std::ofstream out_fs(file_path, std::ios::binary | std::ios::trunc);
    if (!out_fs) {
        throw std::runtime_error("Failed to open file for writing: " + file_path);
    }
    out_fs.write(reinterpret_cast<const char*>(matrix), sizeof(float) * rows * cols);
    if (!out_fs) {
        throw std::runtime_error("Failed to write matrix data to file: " + file_path);
    }
}

// Function to convert dense matrix to CSR format and save to binary file
void save_dense_to_csr(const float* dense_matrix, size_t n, size_t k, const std::string& file_path) {
    std::vector<RowPtrType> row_ptr;
    std::vector<int32_t> col_idx;
    std::vector<float> values;
    row_ptr.reserve(n + 1);
    // Estimate sparsity to reserve space, crude but better than nothing
    size_t estimated_nnz = static_cast<size_t>(static_cast<double>(n) * k * 0.1); // Assume up to 10% sparsity
    col_idx.reserve(estimated_nnz);
    values.reserve(estimated_nnz);

    row_ptr.push_back(0); // First element is always 0

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float val = dense_matrix[i * k + j];
            if (std::abs(val) > ZERO_EPSILON) { // Check if non-zero
                col_idx.push_back(static_cast<int32_t>(j));
                values.push_back(val);
            }
        }
        row_ptr.push_back(static_cast<RowPtrType>(col_idx.size())); // Mark end of row i
    }

    RowPtrType nnz = static_cast<RowPtrType>(values.size());

    std::ofstream out_fs(file_path, std::ios::binary | std::ios::trunc);
    if (!out_fs) {
        throw std::runtime_error("Failed to open file for writing CSR: " + file_path);
    }

    // Write metadata: n, k, nnz
    out_fs.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
    out_fs.write(reinterpret_cast<const char*>(&k), sizeof(size_t));
    out_fs.write(reinterpret_cast<const char*>(&nnz), sizeof(RowPtrType));

    // Write CSR data arrays
    if (n > 0) { // Only write row_ptr if n > 0
        out_fs.write(reinterpret_cast<const char*>(row_ptr.data()), (n + 1) * sizeof(RowPtrType));
    }
    if (nnz > 0) { // Only write col_idx and values if nnz > 0
        out_fs.write(reinterpret_cast<const char*>(col_idx.data()), nnz * sizeof(int32_t));
        out_fs.write(reinterpret_cast<const char*>(values.data()), nnz * sizeof(float));
    }

    if (!out_fs) {
        throw std::runtime_error("Failed to write CSR data to file: " + file_path);
    }
}


// NEW FUNCTION: Reads MTX file directly and saves in CSR binary format
// Returns the dimensions (n, k) read from the MTX header
std::pair<size_t, size_t> save_mtx_to_csr(const std::string& mtx_file_path, const std::string& csr_file_path) {
    std::ifstream mtx_file(mtx_file_path);
    if (!mtx_file.is_open()) {
        throw std::runtime_error("Failed to open Matrix Market file: " + mtx_file_path);
    }

    std::string line;
    // Skip comments
    while (std::getline(mtx_file, line) && line[0] == '%') {}

    // Read header line (dimensions and nnz)
    size_t file_n = 0, file_k = 0;
    RowPtrType file_nnz = 0; // Use RowPtrType for nnz consistency
    std::stringstream ss_header(line);
    if (!(ss_header >> file_n >> file_k >> file_nnz)) {
         mtx_file.close();
         throw std::runtime_error("Failed to parse Matrix Market header line: " + line);
    }

    if (file_n == 0 || file_k == 0) {
         mtx_file.close();
         throw std::runtime_error("Matrix dimensions in header cannot be zero.");
    }
     if (file_nnz < 0) { // Basic check
         mtx_file.close();
         throw std::runtime_error("Number of non-zeros in header cannot be negative.");
     }

    // --- Prepare CSR data structures ---
    // Use vectors for easier handling during read, then write directly
    std::vector<RowPtrType> row_ptr(file_n + 1, 0);
    std::vector<int32_t> col_idx;
    std::vector<float> values;
    if (file_nnz > 0) { // Reserve space if nnz > 0
        try {
            col_idx.reserve(static_cast<size_t>(file_nnz));
            values.reserve(static_cast<size_t>(file_nnz));
        } catch (const std::bad_alloc& e) {
            mtx_file.close();
            throw std::runtime_error("Memory allocation failed for temporary CSR vectors (reserve): " + std::string(e.what()));
        }
    }


    // --- Read MTX entries and populate CSR vectors (coordinate to CSR conversion) ---
    // Need to store entries temporarily to sort them by row/col for CSR
    std::vector<std::tuple<size_t, size_t, float>> coords;
     if (file_nnz > 0) {
         try {
             coords.reserve(static_cast<size_t>(file_nnz));
         } catch (const std::bad_alloc& e) {
             mtx_file.close();
             throw std::runtime_error("Memory allocation failed for coordinate vector (reserve): " + std::string(e.what()));
         }
     }

    size_t r_idx, c_idx; // 1-based from file
    float value;
    RowPtrType nnz_read = 0;
    while (nnz_read < file_nnz && mtx_file >> r_idx >> c_idx >> value) {
        // Adjust from 1-based indexing to 0-based
        if (r_idx > 0 && r_idx <= file_n && c_idx > 0 && c_idx <= file_k) {
             if (std::abs(value) > ZERO_EPSILON) { // Only store non-zeros
                 try {
                    coords.emplace_back(r_idx - 1, c_idx - 1, value);
                 } catch (const std::bad_alloc& e) {
                     mtx_file.close();
                     throw std::runtime_error("Memory allocation failed adding coordinate: " + std::string(e.what()));
                 }
             }
        } else {
             std::cerr << "Warning: Skipping out-of-bounds entry (" << r_idx << ", " << c_idx << ") in MTX file " << mtx_file_path << std::endl;
        }
        nnz_read++;
        // Skip rest of line if necessary
        mtx_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    mtx_file.close(); // Close input file

    RowPtrType actual_nnz = static_cast<RowPtrType>(coords.size()); // Actual non-zeros stored

    if (nnz_read != file_nnz) {
         std::cerr << "Warning: Expected " << file_nnz << " non-zeros in header, but read " << nnz_read << " lines from MTX file " << mtx_file_path << ". Stored " << actual_nnz << " non-zero values." << std::endl;
         // Use actual_nnz from here on
         file_nnz = actual_nnz;
    } else if (actual_nnz != file_nnz) {
         std::cerr << "Warning: Read " << file_nnz << " lines, but stored " << actual_nnz << " non-zero values (due to zero-value entries)." << std::endl;
         // Use actual_nnz from here on
         file_nnz = actual_nnz;
    }


    // Sort coordinates by row, then column
    std::sort(coords.begin(), coords.end());

    // Build CSR from sorted coordinates
    RowPtrType current_nnz = 0;
    for (const auto& coord : coords) {
        size_t r, c;
        float v;
        std::tie(r, c, v) = coord;

        col_idx.push_back(static_cast<int32_t>(c));
        values.push_back(v);
        current_nnz++;

        // Update row pointers for rows between the last entry and this one
        // Note: This assumes coords are sorted by row.
        // We need to fill row_ptr counts.
        // row_ptr[r+1] will store the count of non-zeros *up to the end of row r*.
        row_ptr[r + 1]++; // Increment count for the current row r
    }

    // Cumulative sum for row pointers
    for (size_t i = 0; i < file_n; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // --- Write CSR data to binary file ---
    std::ofstream csr_file(csr_file_path, std::ios::binary | std::ios::trunc);
    if (!csr_file) {
        throw std::runtime_error("Failed to open file for writing CSR: " + csr_file_path);
    }

    // Write metadata: n, k, nnz (use the actual nnz count)
    csr_file.write(reinterpret_cast<const char*>(&file_n), sizeof(size_t));
    csr_file.write(reinterpret_cast<const char*>(&file_k), sizeof(size_t));
    csr_file.write(reinterpret_cast<const char*>(&file_nnz), sizeof(RowPtrType));

    // Write CSR data arrays
    if (file_n > 0) {
        csr_file.write(reinterpret_cast<const char*>(row_ptr.data()), (file_n + 1) * sizeof(RowPtrType));
    }
    if (file_nnz > 0) {
        csr_file.write(reinterpret_cast<const char*>(col_idx.data()), static_cast<size_t>(file_nnz) * sizeof(int32_t));
        csr_file.write(reinterpret_cast<const char*>(values.data()), static_cast<size_t>(file_nnz) * sizeof(float));
    }

    if (!csr_file) {
        throw std::runtime_error("Failed to write CSR data to file: " + csr_file_path);
    }
    csr_file.close();

    return {file_n, file_k}; // Return dimensions read
}


// Function to read a Matrix Market file (.mtx) and return a dense representation
// THIS FUNCTION IS NO LONGER USED FOR --mtx mode, but kept for reference or other modes if needed.
std::unique_ptr<float[]> read_mtx_to_dense(const std::string& file_path, size_t& n, size_t& k) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open Matrix Market file: " + file_path);
    }

    std::string line;
    // Skip comments
    while (std::getline(file, line) && line[0] == '%') {}

    // Read header line (dimensions and nnz)
    size_t rows = 0, cols = 0, nnz = 0;
    std::stringstream ss(line);
    if (!(ss >> rows >> cols >> nnz)) {
         throw std::runtime_error("Failed to parse Matrix Market header line: " + line);
    }

    if (rows == 0 || cols == 0) {
         throw std::runtime_error("Matrix dimensions in header cannot be zero.");
    }

    n = rows;
    k = cols;

    // Allocate dense matrix (initialize to zero)
    size_t num_elements = n * k;
     if (num_elements == 0) {
         // Handle case of empty matrix gracefully
         return std::unique_ptr<float[]>(nullptr);
     }
    auto dense_matrix = std::make_unique<float[]>(num_elements);
    std::fill(dense_matrix.get(), dense_matrix.get() + num_elements, 0.0f);

    // Read non-zero entries
    size_t r_idx, c_idx;
    float value;
    size_t nnz_read = 0;
    while (nnz_read < nnz && file >> r_idx >> c_idx >> value) {
        // Adjust from 1-based indexing to 0-based
        if (r_idx > 0 && r_idx <= n && c_idx > 0 && c_idx <= k) {
            dense_matrix[(r_idx - 1) * k + (c_idx - 1)] = value;
        } else {
             std::cerr << "Warning: Skipping out-of-bounds entry (" << r_idx << ", " << c_idx << ") in MTX file " << file_path << std::endl;
        }
        nnz_read++;
        // Skip rest of line if necessary (e.g., if complex numbers were present but ignored)
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

     if (nnz_read != nnz) {
         std::cerr << "Warning: Expected " << nnz << " non-zeros, but read " << nnz_read << " entries from MTX file " << file_path << std::endl;
     }


    return dense_matrix;
}

// Function to read a graph file (SNAP format) and return a dense adjacency matrix
std::unique_ptr<float[]> read_graph_to_dense(const std::string& file_path, size_t& n, size_t& k) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open graph file: " + file_path);
    }

    std::string line;
    size_t expected_nodes = 0;
    size_t expected_edges = 0;
    size_t max_node_id = 0;
    std::vector<std::pair<size_t, size_t>> edges;
    bool nodes_comment_found = false;

    // First pass: Read comments for dimensions and find max node ID from edges
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            std::stringstream ss(line);
            std::string token;
            ss >> token; // Skip '#'
            if (ss >> token && token == "Nodes:") {
                if (ss >> expected_nodes) {
                    nodes_comment_found = true;
                } else {
                     std::cerr << "Warning: Could not parse Nodes count from comment: " << line << std::endl;
                }
                // Optionally parse edges count too
                if (ss >> token && token == "Edges:") {
                     if (!(ss >> expected_edges)) {
                         std::cerr << "Warning: Could not parse Edges count from comment: " << line << std::endl;
                     }
                }
            }
            continue; // Skip rest of comment processing
        }

        // Parse edge data
        std::stringstream ss(line);
        size_t u, v;
        if (ss >> u >> v) {
            edges.push_back({u, v});
            max_node_id = std::max({max_node_id, u, v});
        } else {
            std::cerr << "Warning: Skipping malformed edge line: " << line << std::endl;
        }
    }

    // Determine dimensions n and k (for adjacency matrix, n=k)
    if (nodes_comment_found) {
        n = expected_nodes;
        if (max_node_id >= n) {
            std::cerr << "Warning: Max node ID (" << max_node_id << ") in edges is >= declared Nodes (" << n << "). Adjusting matrix dimension." << std::endl;
            n = max_node_id + 1; // Adjust dimension based on actual data
        }
    } else {
        std::cerr << "Warning: '# Nodes:' comment not found. Determining dimension from max node ID." << std::endl;
        n = max_node_id + 1; // Dimension based on 0-based indexing up to max_node_id
    }
    k = n; // Adjacency matrix is square

    if (n == 0) {
         std::cerr << "Warning: Graph appears empty or could not determine dimensions. Resulting matrix will be empty." << std::endl;
         return std::unique_ptr<float[]>(nullptr);
    }

    std::cout << "  Graph Info: Determined dimensions n=" << n << ", k=" << k << " (max_node_id=" << max_node_id << ")" << std::endl;
    if (edges.size() != expected_edges && nodes_comment_found) { // Only warn if edge count was declared
         std::cerr << "Warning: Read " << edges.size() << " edges, but expected " << expected_edges << " from comments." << std::endl;
    }


    // Allocate dense matrix (initialize to zero)
    size_t num_elements = n * k;
    auto dense_matrix = std::make_unique<float[]>(num_elements);
    std::fill(dense_matrix.get(), dense_matrix.get() + num_elements, 0.0f);

    // Populate dense matrix from edges (set value to 1.0f)
    size_t edges_placed = 0;
    for (const auto& edge : edges) {
        size_t u = edge.first;
        size_t v = edge.second;
        // Check bounds before writing
        if (u < n && v < k) {
            dense_matrix[u * k + v] = 1.0f;
            edges_placed++;
        } else {
             std::cerr << "Warning: Skipping out-of-bounds edge (" << u << ", " << v << ") for determined dimensions " << n << "x" << k << "." << std::endl;
        }
    }
     std::cout << "  Placed " << edges_placed << " edges as non-zeros in the dense matrix." << std::endl;


    return dense_matrix;
}

// Function to generate a random dense matrix
std::unique_ptr<float[]> generate_dense_matrix(size_t rows, size_t cols, std::mt19937& rng) {
    size_t num_elements = rows * cols;
     if (num_elements == 0) return std::unique_ptr<float[]>(nullptr);
    auto matrix = std::make_unique<float[]>(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        matrix[i] = generate_random_float(rng);
    }
    return matrix;
}

// Function to generate a random sparse matrix and return its dense representation
std::unique_ptr<float[]> generate_random_sparse_to_dense(size_t n, size_t k, float sparsity, std::mt19937& rng) {
    size_t num_elements = n * k;
     if (num_elements == 0) return std::unique_ptr<float[]>(nullptr);
    auto dense_matrix = std::make_unique<float[]>(num_elements);
    std::fill(dense_matrix.get(), dense_matrix.get() + num_elements, 0.0f);

    size_t num_nonzero = static_cast<size_t>(num_elements * sparsity);
    if (num_nonzero > num_elements) num_nonzero = num_elements; // Cap at total elements

    std::uniform_int_distribution<size_t> dist(0, num_elements - 1);
    size_t count = 0;
    // Use a simple approach: try random indices until enough non-zeros are placed.
    // For very high sparsity or very large matrices, a set could be more efficient
    // to avoid collisions, but this is simpler for moderate cases.
    while (count < num_nonzero) {
        size_t idx = dist(rng);
        // Only add if it's currently zero to ensure exact sparsity count
        if (dense_matrix[idx] == 0.0f) {
            dense_matrix[idx] = generate_random_float(rng);
            count++;
        }
        // Add a safeguard against infinite loops if sparsity is 1.0 and float generation hits 0.0f, though unlikely with current range.
        // A more robust method would track indices tried.
    }

    return dense_matrix;
}


// --- Main Function ---
int main(int argc, char *argv[]) {
    if (argc < 4) { // Need at least mode, path/n, k/m, m/sparsity
        print_usage();
        return EXIT_FAILURE;
    }

    std::string mode = argv[1];
    std::string input_file_path; // For MTX or Graph
    size_t n = 0, k = 0, m = 0;
    float sparsity = 0.0f;
    unsigned int seed = std::random_device{}();
    enum class InputType { GENERATE, MTX, GRAPH };
    InputType input_type;


    try {
        if (mode == "--generate") {
            if (argc < 6) { print_usage(); return EXIT_FAILURE; }
            input_type = InputType::GENERATE;
            n = std::stoull(argv[2]);
            k = std::stoull(argv[3]);
            m = std::stoull(argv[4]);
            sparsity = std::stof(argv[5]);
            if (sparsity < 0.0f || sparsity > 1.0f) {
                 throw std::invalid_argument("Sparsity must be between 0.0 and 1.0");
            }
            if (argc > 6) { seed = std::stoul(argv[6]); }
            if (n == 0) throw std::invalid_argument("Dimension n must be positive for generation mode.");

        } else if (mode == "--mtx") {
             if (argc < 5) { print_usage(); return EXIT_FAILURE; }
             input_type = InputType::MTX;
             input_file_path = argv[2];
             // k and m are specified for B, n and actual k for A come from the file
             // We still parse k and m here, but k might be overridden
             try {
                 k = std::stoull(argv[3]); // Initial k for B
                 m = std::stoull(argv[4]);
             } catch (const std::invalid_argument& e) {
                 throw std::invalid_argument("Invalid number format for k or m: " + std::string(e.what()));
             } catch (const std::out_of_range& e) {
                 throw std::out_of_range("k or m value out of range: " + std::string(e.what()));
             }
             if (argc > 5) {
                 try {
                     seed = std::stoul(argv[5]);
                 } catch (const std::invalid_argument& e) {
                     throw std::invalid_argument("Invalid number format for seed: " + std::string(e.what()));
                 } catch (const std::out_of_range& e) {
                     throw std::out_of_range("Seed value out of range: " + std::string(e.what()));
                 }
             }

        } else if (mode == "--graph") {
             if (argc < 4) { print_usage(); return EXIT_FAILURE; }
             input_type = InputType::GRAPH;
             input_file_path = argv[2];
             // n and k will be determined by reading the graph file (n=k)
             m = std::stoull(argv[3]); // Only m is specified
             if (argc > 4) { seed = std::stoul(argv[4]); }

        } else {
            // Handle the old positional argument style for backward compatibility or error
            // Check if the first argument looks like a file path (contains '.' or '/')
            // This is a basic check and might not cover all cases.
            if (mode.find('.') != std::string::npos || mode.find('/') != std::string::npos || mode.find('\\') != std::string::npos) {
                 if (argc < 4) { print_usage(); return EXIT_FAILURE; }
                 std::cerr << "Warning: Using deprecated positional argument style for MTX file. Use '--mtx " << mode << "' instead." << std::endl;
                 input_type = InputType::MTX;
                 input_file_path = mode;
                 k = std::stoull(argv[2]);
                 m = std::stoull(argv[3]);
                 if (argc > 4) { seed = std::stoul(argv[4]); }
            } else {
                 std::cerr << "Error: Invalid mode specified: " << mode << std::endl;
                 print_usage();
                 return EXIT_FAILURE;
            }
        }

        // Validate m (k is validated after processing input file)
        if (m == 0) {
             throw std::invalid_argument("Dimension m must be positive.");
        }

    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        print_usage();
        return EXIT_FAILURE;
    }

    std::mt19937 rng(seed);
    std::unique_ptr<float[]> matrix_B_dense;

    // Create temporary file paths outside the try-catch for cleanup
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::string path_A = (temp_dir / ("matrix_A_" + timestamp + ".csr.dat")).string(); // Indicate CSR format
    std::string path_B = (temp_dir / ("matrix_B_" + timestamp + ".dat")).string();
    std::string student_sol_path; // Also declare here for cleanup

    try {
        std::cout << "Benchmark Setup:\n";
        size_t k_A_check = 0; // k dimension read from matrix A source

        if (input_type == InputType::GENERATE) {
            std::cout << "  Mode: Generate Random Sparse A\n";
            std::cout << "  Dimensions: n=" << n << ", k=" << k << ", m=" << m << "\n";
            std::cout << "  Sparsity: " << sparsity * 100.0 << "%\n";
            // Need matrix_A_dense temporarily for generation
            std::unique_ptr<float[]> matrix_A_dense = generate_random_sparse_to_dense(n, k, sparsity, rng);
            k_A_check = k; // k is known for generation

            // Save generated A to CSR
            std::cout << "\nSaving temporary matrices (A in CSR, B in Dense)...\n";
            std::cout << "  A (CSR) -> " << path_A << std::endl;
            if (matrix_A_dense) {
                 save_dense_to_csr(matrix_A_dense.get(), n, k, path_A);
            } else if (n > 0 && k > 0) {
                 std::cout << "  Warning: Generated Matrix A is empty, creating an empty CSR file." << std::endl;
                 save_dense_to_csr(nullptr, n, k, path_A);
            } else {
                 throw std::runtime_error("Cannot save generated Matrix A: dimensions are zero or matrix pointer is null.");
            }
            matrix_A_dense.reset(); // Free memory immediately

        } else if (input_type == InputType::MTX) {
            std::cout << "  Mode: Load Sparse A from MTX file\n";
            std::cout << "  MTX File: " << input_file_path << "\n";

            // Directly save MTX to CSR format
            std::cout << "\nConverting MTX to CSR and saving temporary matrices (A in CSR, B in Dense)...\n";
            std::cout << "  A (MTX -> CSR) -> " << path_A << std::endl;
            std::tie(n, k_A_check) = save_mtx_to_csr(input_file_path, path_A); // Read MTX, save CSR, get dims

             if (n == 0 || k_A_check == 0) {
                 throw std::runtime_error("Matrix A loaded from MTX file has zero dimensions (n=" + std::to_string(n) + ", k=" + std::to_string(k_A_check) + ").");
             }
             // Use k from the matrix file, overriding command line if necessary
             if (k != k_A_check) {
                 std::cout << "  Info: Using k=" << k_A_check << " from MTX file columns for matrix B (overriding command line k=" << k << ")." << std::endl;
                 k = k_A_check; // Use the dimension from the actual matrix A
             }
             std::cout << "  Matrix A Dimensions (from file): n=" << n << ", k=" << k << "\n";

        } else { // InputType::GRAPH
             std::cout << "  Mode: Load Sparse A from Graph file\n";
             std::cout << "  Graph File: " << input_file_path << "\n";
             // Need matrix_A_dense temporarily for graph reading
             std::unique_ptr<float[]> matrix_A_dense = read_graph_to_dense(input_file_path, n, k);
             k_A_check = k; // n=k for graph adjacency

             if (n == 0 || k == 0) {
                  throw std::runtime_error("Matrix A loaded from graph file has zero dimensions.");
             }
             std::cout << "  Matrix A Dimensions (from file): n=" << n << ", k=" << k << "\n";

             // Save graph A to CSR
             std::cout << "\nSaving temporary matrices (A in CSR, B in Dense)...\n";
             std::cout << "  A (CSR) -> " << path_A << std::endl;
             if (matrix_A_dense) {
                  save_dense_to_csr(matrix_A_dense.get(), n, k, path_A);
             } else { // Should not happen if n,k > 0, but handle defensively
                  std::cout << "  Warning: Graph Matrix A is empty, creating an empty CSR file." << std::endl;
                  save_dense_to_csr(nullptr, n, k, path_A);
             }
             matrix_A_dense.reset(); // Free memory immediately
        }

         std::cout << "  Matrix B Dimensions (generated): k=" << k << ", m=" << m << "\n";
         std::cout << "  Seed: " << seed << std::endl;

        // Validate k dimension consistency after processing A
        if (k == 0) {
             throw std::runtime_error("Dimension k for matrix A/B could not be determined or is zero.");
        }

        // Generate dense matrix B (common to all modes)
        matrix_B_dense = generate_dense_matrix(k, m, rng);

        // Save B (common to all modes)
        std::cout << "  B (Dense) -> " << path_B << std::endl;
        if (matrix_B_dense) {
             save_dense_matrix(matrix_B_dense.get(), k, m, path_B);
        } else {
             // This might happen if k or m is 0, though we check m earlier and k now.
             throw std::runtime_error("Cannot save Matrix B: matrix pointer is null (k=" + std::to_string(k) + ", m=" + std::to_string(m) + ").");
        }

        // Ensure matrix B is freed before calling student solution
        matrix_B_dense.reset();

        // --- Run and Time Student Solution ---
        std::cout << "\nRunning solution::compute...\n";
        // student_sol_path declared outside try block
        auto start = std::chrono::high_resolution_clock::now();
        try {
             // Convert size_t potentially large dimensions safely to int for the API
             if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                 k > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                 m > static_cast<size_t>(std::numeric_limits<int>::max())) {
                 throw std::overflow_error("Matrix dimensions exceed integer limits for solution::compute API (n=" + std::to_string(n) + ", k=" + std::to_string(k) + ", m=" + std::to_string(m) + ").");
             }
            student_sol_path = solution::compute(path_A, path_B, static_cast<int>(n), static_cast<int>(k), static_cast<int>(m));
        } catch (const std::exception& sol_ex) {
             std::cerr << "!!! Error during solution::compute execution: " << sol_ex.what() << std::endl;
             // Clean up temporary files even if solution fails
             try { if (!path_A.empty() && std::filesystem::exists(path_A)) std::filesystem::remove(path_A); } catch(...) {}
             try { if (!path_B.empty() && std::filesystem::exists(path_B)) std::filesystem::remove(path_B); } catch(...) {}
             return EXIT_FAILURE;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "solution::compute finished.\n";
        std::cout << "  Output file: " << student_sol_path << std::endl;
        std::cout << "  Execution time: " << duration.count() << " ms" << std::endl;

        // --- Cleanup ---
        std::cout << "\nCleaning up temporary files...\n";
        try { if (!path_A.empty() && std::filesystem::exists(path_A)) { std::filesystem::remove(path_A); std::cout << "  Removed " << path_A << std::endl; } } catch(const std::exception& e) { std::cerr << "Warning: Failed to remove temp file " << path_A << ": " << e.what() << std::endl; }
        try { if (!path_B.empty() && std::filesystem::exists(path_B)) { std::filesystem::remove(path_B); std::cout << "  Removed " << path_B << std::endl; } } catch(const std::exception& e) { std::cerr << "Warning: Failed to remove temp file " << path_B << ": " << e.what() << std::endl; }
        // Also remove the student's output file
        if (!student_sol_path.empty()) {
             try {
                 if (std::filesystem::exists(student_sol_path)) {
                     std::filesystem::remove(student_sol_path);
                     std::cout << "  Removed " << student_sol_path << std::endl;
                 } else {
                     std::cout << "  Student solution path not found, skipping removal: " << student_sol_path << std::endl;
                 }
             } catch(const std::exception& e) {
                 std::cerr << "Warning: Failed to remove student solution file " << student_sol_path << ": " << e.what() << std::endl;
             }
        } else {
             std::cout << "  Student solution path was empty, skipping removal." << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << "\n!!! An error occurred during setup or execution: " << e.what() << std::endl;
        // Attempt cleanup even on error
        try { if (!path_A.empty() && std::filesystem::exists(path_A)) std::filesystem::remove(path_A); } catch(...) {}
        try { if (!path_B.empty() && std::filesystem::exists(path_B)) std::filesystem::remove(path_B); } catch(...) {}
        try { if (!student_sol_path.empty() && std::filesystem::exists(student_sol_path)) std::filesystem::remove(student_sol_path); } catch(...) {}
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}