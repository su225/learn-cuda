#pragma once

#include <vector>
#include <stdexcept>

std::vector<float> random_matrix(int size, float scale = 1.0f);

void compare(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float atol = 1e-4f);
