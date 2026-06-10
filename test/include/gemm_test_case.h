#pragma once

#include <gtest/gtest.h>

struct GemmCase {
    int M, N, K;
};

class GemmTest : public ::testing::TestWithParam<GemmCase> {};

std::ostream& operator<<(std::ostream& os, const GemmCase& c) {
    return os << "M=" << c.M << "_N=" << c.N << "_K=" << c.K;
}
