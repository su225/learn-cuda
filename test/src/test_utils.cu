#include "test_utils.h"
#include "common.h"
#include "gemm_test_case.h"

#include <iostream>
#include <stdexcept>

void compare(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    int rows, int cols,
    float atol) {
    if (actual.size() != expected.size())
        throw std::runtime_error("size mismatch");

    for (size_t i = 0; i < actual.size(); i++) {
        if (std::fabs(actual[i] - expected[i]) > atol) {
            int mismatchRow = i / cols;
            int mismatchCol = i % cols;
            std::cout << "Mismatch at index " << i
                << '(' << "row=" << mismatchRow << ","
                       << "col=" << mismatchCol << ')' << std::endl;
            std::cout << "Expected: " << std::endl;
            print_matrix(actual, rows, cols, 10, 10); std::cout << std::endl;
            std::cout << "Actual: " << std::endl;
            print_matrix(expected, rows, cols, 10, 10); std::cout << std::endl;
            throw std::runtime_error("Mismatch at index " + std::to_string(i));
        }
    }
}

std::ostream& operator<<(std::ostream& os, const GemmCase& c) {
    return os << "M=" << c.M << "_N=" << c.N << "_K=" << c.K;
}
