#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "common.h"
#include "kernels/gemm.h"

struct Args {
    std::string kernel = "naive";
    int M = 512;
    int N = 512;
    int K = 512;
    int seed = 42;
};

inline void print_usage(const char* program_name) {
    std::cout
        << "Usage: " << program_name << " [options]\n\n"
        << "Options:\n"
        << "  --kernel <name>   Kernel name (naive, tiled)\n"
        << "  --M <value>       Rows of A / C\n"
        << "  --N <value>       Columns of B / C\n"
        << "  --K <value>       Columns of A / Rows of B\n"
        << "  --seed <value>    RNG seed\n"
        << "  --help            Show this help\n";
}

inline Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--kernel") {
            if (++i >= argc) {
                throw std::runtime_error("--kernel requires a value");
            }
            args.kernel = argv[i];
        } else if (arg == "--M") {
            if (++i >= argc) {
                throw std::runtime_error("--M requires a value");
            }
            args.M = std::stoi(argv[i]);
        } else if (arg == "--N") {
            if (++i >= argc) {
                throw std::runtime_error("--N requires a value");
            }
            args.N = std::stoi(argv[i]);
        } else if (arg == "--K") {
            if (++i >= argc) {
                throw std::runtime_error("--K requires a value");
            }
            args.K = std::stoi(argv[i]);
        } else if (arg == "--seed") {
            if (++i >= argc) {
                throw std::runtime_error("--seed requires a value");
            }
            args.seed = std::stoi(argv[i]);
        } else if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return args;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        std::cout << "Kernel : " << args.kernel << '\n';
        std::cout << "M      : " << args.M << '\n';
        std::cout << "N      : " << args.N << '\n';
        std::cout << "K      : " << args.K << '\n';
        std::cout << "Seed   : " << args.seed << '\n';

        std::vector<float> a = random_matrix(args.M * args.K, 0.5, args.seed);
        std::vector<float> b = random_matrix(args.K * args.N, 0.4, args.seed);
        std::vector<float> c(args.M * args.N);

        if (args.kernel == "naive") {
            std::cout << "Running naive GEMM" << std::endl;
            naive_gemm(a, b, c, args.M, args.N, args.K);
        } else if (args.kernel == "tiled") {
            std::cout << "Running tiled GEMM" << std::endl;
            tiled_gemm(a, b, c, args.M, args.N, args.K);
        } else if (args.kernel == "tiled_v2") {
            std::cout << "Running tiledV2 GEMM" << std::endl;
            tiled_gemm_v2(a, b, c, args.M, args.N, args.K);
        } else if (args.kernel == "tiled_vec_v2") {
            std::cout << "Running tiledV2 with Vectorized load" << std::endl;
            tiled_gemm_vectorized_v2(a, b, c, args.M, args.N, args.K);
        } else if (args.kernel == "tiled_async_v2") {
            std::cout << "Running tiledV2 with Async Memcpy load" << std::endl;
            tiled_gemm_async_v2(a, b, c, args.M, args.N, args.K);
        } else if (args.kernel == "cublas") {
            std::cout << "Running cuBLASLt GEMM" << std::endl;
            cuBLASLt_gemm(a, b, c, args.M, args.N, args.K);
        } else {
            throw std::runtime_error("Unknown kernel: " + args.kernel);
        }
        print_matrix(c, args.M, args.N, 10, 10);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
