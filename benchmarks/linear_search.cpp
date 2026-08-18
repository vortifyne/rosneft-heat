#include <algorithm>
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

std::vector<int> create_sorted_vector(size_t size) {
    constexpr int sid = 42;
    std::vector<int> data(size);
    std::mt19937 gen(sid);
    std::uniform_int_distribution<size_t> dis(1, size * 2);
    std::ranges::generate(data, [&]() { return dis(gen); });
    std::ranges::sort(data);

    return data;
}

static void bm_linear_search(benchmark::State& state) {
    const size_t size = state.range(0);
    auto data = create_sorted_vector(size);
    const int target = data[size / 2];

    for (auto _ : state) {
        auto it = std::ranges::find(data, target);
        benchmark::DoNotOptimize(it);
    }

    state.SetComplexityN(static_cast<benchmark::IterationCount>(size));
}

BENCHMARK(bm_linear_search)->Range(1 << 3, 1 << 12)->Complexity();

BENCHMARK_MAIN();
