#ifndef RANDOM_FORESTS
#define RANDOM_FORESTS

#include "TreeParameters.h"
#include "DecisionTree.hpp"

#include <opencv2/core/core.hpp>

#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <filesystem>

namespace nuisken {
namespace randomforests {

/**
 * ランダムフォレストのクラス
 */
template <class Type>
class RandomForests {
   private:
    typedef std::shared_ptr<typename Type::FeatureType> FeaturePtr;
    typedef typename Type::FeatureType* FeatureRawPtr;
    typedef std::shared_ptr<typename Type::LeafType> LeafPtr;

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
            forests.push_back(std::move(DecisionTree<Type>(type, parameters)));
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
    std::vector<LeafPtr> match(const FeaturePtr& feature) const;
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