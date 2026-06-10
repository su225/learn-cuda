#pragma once

#include <vector>

std::vector<float> random_matrix(int size, float scale = 1.0f, int seed = 42);

void compare(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float atol = 1e-4f);
