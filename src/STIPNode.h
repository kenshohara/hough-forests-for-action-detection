#ifndef SITP_NODE
#define SITP_NODE

#include "STIPFeature.h"
#include "STIPSplitParameters.h"
#include "STIPLeaf.h"
#include "RandomGenerator.h"

#include <random>
#include <memory>

namespace nuisken {
namespace randomforests {

class STIPNode {
   private:
    typedef storage::STIPFeature* FeatureRawPtr;
    typedef std::shared_ptr<STIPLeaf> LeafPtr;

    enum MeasureType { CLASS, VECTOR };

   public:
    typedef storage::STIPFeature FeatureType;
    typedef STIPSplitParameters SplitParametersType;
    typedef STIPLeaf LeafType;

   private:
    MeasureType type;
    STIPLeaf stipLeaf;
    int numberOfClasses;
    int numberOfFeatureChannels;
    std::vector<int> numberOfFeatureDimensions;

   public:
    STIPNode() { type = decideType(); }

    STIPNode(int numberOfClasses, int numberOfFeatureChannels,
             const std::vector<int>& numberOfFeatureDimensions)
            : numberOfClasses(numberOfClasses),
              numberOfFeatureChannels(numberOfFeatureChannels),
              numberOfFeatureDimensions(numberOfFeatureDimensions) {
        type = decideType();
    }

    STIPNode(const STIPNode& stipNode) {
        this->stipLeaf = stipNode.stipLeaf;
        this->numberOfClasses = stipNode.numberOfClasses;
        this->numberOfFeatureChannels = stipNode.numberOfFeatureChannels;
        this->numberOfFeatureDimensions = stipNode.numberOfFeatureDimensions;

        type = decideType();
    }

    STIPNode& operator=(const STIPNode& stipNode) {
        this->stipLeaf = stipNode.stipLeaf;
        this->numberOfClasses = stipNode.numberOfClasses;
        this->numberOfFeatureChannels = stipNode.numberOfFeatureChannels;
        this->numberOfFeatureDimensions = stipNode.numberOfFeatureDimensions;

        type = decideType();

        return *this;
    }

    double calculateSplitValue(const FeatureRawPtr& feature,
                               const STIPSplitParameters& parameter) const {
        return feature->getFeatureValue(parameter.getIndex1(), parameter.getFeatureChannel()) -
               feature->getFeatureValue(parameter.getIndex2(), parameter.getFeatureChannel());
    }

    double generateTau(double minValue, double maxValue) {
        std::uniform_real_distribution<> distribution(minValue, maxValue);
        return distribution(RandomGenerator::getInstance().generator_);
    }

    STIPSplitParameters generateRandomParameter();

    double evaluateSplit(const std::vector<FeatureRawPtr>& leftFeatures,
                         const std::vector<FeatureRawPtr>& rightFeatures) const;

    /**
        * マッチした時に返すデータを計算（葉ノードのみ）
        */
    LeafPtr calculateLeafData(const std::vector<FeatureRawPtr>& features) const;

    /**
        * どちらの葉ノードに判別されるか
        * trueなら左，falseなら右
        */
    bool decision(const FeatureRawPtr& feature, const SplitParametersType& splitParameter,
                  double tau) const;

    int getNumberOfClasses() const { return numberOfClasses; }

    void setNumberOfClasses(int classes) { numberOfClasses = classes; }

    LeafPtr loadLeafData(std::queue<std::string>& nodeElements) const;

   private:
    MeasureType decideType();
    double calculateClassUncertainty(const std::vector<FeatureRawPtr>& features) const;
    double calculateVectorUncertainty(const std::vector<FeatureRawPtr>& features) const;
};
}
}

#endif