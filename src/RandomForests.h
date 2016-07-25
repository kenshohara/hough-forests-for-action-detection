#ifndef RANDOM_FORESTS
#define RANDOM_FORESTS

#include "DecisionTree.hpp"
#include "TreeParameters.h"

#include <opencv2/core/core.hpp>

#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nuisken {
namespace randomforests {

/**
 * ランダムフォレストのクラス
 */
template <class Type>
class RandomForests {
   private:
    using FeaturePtr = std::shared_ptr<typename Type::FeatureType>;
    using FeatureRawPtr = typename Type::FeatureType*;
    using LeafPtr = std::shared_ptr<typename Type::LeafType>;

   private:
    Type type;

    /**
     * 木のパラメータ
     */
    TreeParameters parameters;

    /**
     * 決定木
     */
    std::vector<DecisionTree<Type>> forests;

   public:
    RandomForests(){};

    RandomForests(const Type& type, const TreeParameters& parameters)
            : type(type), parameters(parameters) {
        //決定木の初期化
        initForests();
    }

    void initForests() {
        forests.clear();
        forests.reserve(parameters.getNumberOfTrees());
        for (int i = 0; i < parameters.getNumberOfTrees(); ++i) {
            forests.emplace_back(type, parameters);
        }
    }

    int getNumberOfTrees() const { return forests.size(); }

    void RandomForests::setParameters(const TreeParameters& parameters) {
        this->parameters = parameters;

        for (auto& forest : forests) {
            forest.setParameters(parameters);
        }
    }

    void RandomForests::setType(const Type& type) { this->type = type; }

    void train(const std::vector<FeaturePtr>& features, int maxNumberOfThreads = 1);
    void match(const FeaturePtr& feature, std::vector<LeafPtr>& leavesData) const;
    void save(const std::string& directoryPath) const;
    void load(const std::string& directoryPath);

   private:
    void selectBootstrapData(const std::vector<FeaturePtr>& features,
                             std::vector<FeatureRawPtr>& bootstrapData);
    void selectBootstrapDataAllRatio(const std::vector<FeaturePtr>& features,
                                     std::vector<FeatureRawPtr>& bootstrapData);
    void selectBootstrapDataMaxWithoutNegative(const std::vector<FeaturePtr>& features,
                                               std::vector<FeatureRawPtr>& bootstrapData);
    void trainOneTree(const std::vector<FeaturePtr>& features, int index);
    void trainOneTree(const std::vector<FeaturePtr>& features,
                      std::vector<FeatureRawPtr>& bootstrapData, int index);
};
}
}

#endif