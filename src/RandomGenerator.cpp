#include "RandomGenerator.h"

namespace nuisken {
std::unique_ptr<RandomGenerator> RandomGenerator::instance_;
std::once_flag RandomGenerator::onceFlag_;
int RandomGenerator::manualSeed_;

RandomGenerator& RandomGenerator::getInstance() {
    std::call_once(onceFlag_, [] { instance_.reset(new RandomGenerator); });
    return *instance_.get();
}
}