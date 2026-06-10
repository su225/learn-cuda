#include "test_utils.h"
#include <random>
#include <stdexcept>

std::vector<float> random_matrix(int size, float scale, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);

    std::vector<float> v(size);
    for (auto& x : v)
        x = dist(gen);
    return v;
}

void compare(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float atol) {
    if (a.size() != b.size())
        throw std::runtime_error("size mismatch");

    for (size_t i = 0; i < a.size(); i++) {
        if (std::fabs(a[i] - b[i]) > atol) {
            throw std::runtime_error("Mismatch at index " + std::to_string(i));
        }
    }
}
