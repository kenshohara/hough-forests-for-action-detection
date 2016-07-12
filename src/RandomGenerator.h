#ifndef RANDOM_GENERATOR
#define RANDOM_GENERATOR

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>

namespace nuisken {
class RandomGenerator {
   public:
    std::mt19937 generator_;

   private:
    static std::unique_ptr<RandomGenerator> instance_;
    static std::once_flag onceFlag_;

    // -1ならランダムシード
    static int manualSeed_;

   public:
    RandomGenerator(const RandomGenerator& randomGenerator) = delete;
    RandomGenerator& operator=(const RandomGenerator& randomGenerator) = delete;

    static RandomGenerator& getInstance();

    static void setManualSeed(int seed) { manualSeed_ = seed; }

   private:
    RandomGenerator() {
        if (manualSeed_ == -1) {
            std::random_device randomDevice;
            std::vector<std::uint_least32_t> vec(10);
            std::generate(std::begin(vec), std::end(vec), std::ref(randomDevice));
            generator_.seed(std::seed_seq(std::begin(vec), std::end(vec)));
        } else {
            generator_.seed(manualSeed_);
        }
    };
};
}

#endif