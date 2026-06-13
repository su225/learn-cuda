#pragma once

#include <vector>

void compare(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int rows, int cols,
    float atol = 1e-4f);
