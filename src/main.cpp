#include <iostream>
#include <format>
#include <fmt/core.h>

int main() {
#if __cplusplus >= 202002L
    std::cout << std::format("Standard: {}\n", __cplusplus);
#else
    std::cout << std::format("Default standard: {}\n", __cplusplus);
#endif

    fmt::print("Hello World from fmt version {}.{}.{}!\n",
               FMT_VERSION / 10000, (FMT_VERSION % 10000) / 100, FMT_VERSION % 100);
}
