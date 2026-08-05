#include <iostream>
#include <format>

int main() {
#if __cplusplus >= 202002L
    std::cout << std::format("Standard: {}\n", __cplusplus);
#else
    std::cout << std::format("Default standard: {}\n", __cplusplus);
#endif

    std::cout << "Hello world\n";
}