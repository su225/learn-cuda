#pragma once

#include <vector>
#include <random>

std::vector<float> random_matrix(int size, float scale = 1.0f, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);

    std::vector<float> v(size);
    for (auto& x : v)
        x = dist(gen);
    return v;
}
