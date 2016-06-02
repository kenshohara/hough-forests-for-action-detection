#include "STIPNode.h"

#include <Eigen/Core>

#include <boost/timer.hpp>

#include <iostream>
#include <numeric>

namespace nuisken {
namespace randomforests {

STIPNode::MeasureType STIPNode::decideType() {
    std::uniform_int_distribution<> distribution(0, 1);
    int typeNumber = distribution(RandomGenerator::getInstance().generator_);
    switch (typeNumber) {
        case 0:
            return CLASS;
        case 1:
            return VECTOR;
        default:
            return CLASS;
    }
}

STIPSplitParameters STIPNode::generateRandomParameter() {
    std::uniform_int_distribution<> channelDistribution(0, numberOfFeatureChannels - 1);
    int featureChannel = channelDistribution(RandomGenerator::getInstance().generator_);

    std::uniform_int_distribution<> indexDistribution(
            0, numberOfFeatureDimensions.at(featureChannel) - 1);
    int index1 = indexDistribution(RandomGenerator::getInstance().generator_);
    // int index2;
    // do {
    int index2 = indexDistribution(RandomGenerator::getInstance().generator_);
    //} while (index1 == index2);

    return STIPSplitParameters(index1, index2, featureChannel);
}

double STIPNode::evaluateSplit(const std::vector<FeatureRawPtr>& leftFeatures,
                               const std::vector<FeatureRawPtr>& rightFeatures) const {
    auto leftValue = 0.0;
    auto rightValue = 0.0;

    switch (type) {
        case CLASS:
            leftValue = calculateClassUncertainty(leftFeatures);
            rightValue = calculateClassUncertainty(rightFeatures);
            break;
        case VECTOR:
            leftValue = calculateVectorUncertainty(leftFeatures);
            rightValue = calculateVectorUncertainty(rightFeatures);
            break;
    }

    return (leftValue + rightValue) / (leftFeatures.size() + rightFeatures.size());
}

double STIPNode::calculateClassUncertainty(const std::vector<FeatureRawPtr>& features) const {
    //各クラスの割合を計算
    Eigen::VectorXd classProbabilities = Eigen::VectorXd::Zero(numberOfClasses);
    auto oneProbability = 1.0 / features.size();
    for (const auto& feature : features) {
        classProbabilities(feature->getClassLabel()) += oneProbability;
    }

    //曖昧さ（エントロピー）を計算
    double uncertainty = 0.0;
    for (auto i = 0; i < classProbabilities.size(); ++i) {
        if (0.0 != classProbabilities(i)) {
            uncertainty += classProbabilities(i) * std::log(classProbabilities(i));
        }
    }
    uncertainty *= static_cast<double>(features.size());

    return uncertainty;
}

double STIPNode::calculateVectorUncertainty(const std::vector<FeatureRawPtr>& features) const {
    // displacementVectorの平均を計算
    std::vector<cv::Vec3f> meanDisplacementVectors(numberOfClasses);
    Eigen::VectorXi sizes = Eigen::VectorXi::Zero(numberOfClasses);

    auto end = std::end(features);
    for (auto itr = std::begin(features); itr != end; ++itr) {
        auto displacementVector = (*itr)->getDisplacementVector();
        auto classLabel = (*itr)->getClassLabel();

        meanDisplacementVectors.at(classLabel) += displacementVector;

        ++sizes(classLabel);
    }
    for (auto i = 0; i < numberOfClasses; ++i) {
        meanDisplacementVectors.at(i) /= static_cast<double>(sizes(i));
    }

    //曖昧さを計算
    auto uncertainty = 0.0;
    for (auto itr = std::begin(features); itr != end; ++itr) {
        // cv::Vec3iをcv::Vec3fに変換
        cv::Vec3f displacementVector((*itr)->getDisplacementVector());
        auto difference = displacementVector - meanDisplacementVectors.at((*itr)->getClassLabel());

        uncertainty += cv::norm(difference);
    }

    return -uncertainty;
}

bool STIPNode::decision(const FeatureRawPtr& feature, const STIPSplitParameters& splitParameter,
                        double tau) const {
    double value1 = feature->getFeatureValue(splitParameter.getIndex1(),
                                             splitParameter.getFeatureChannel());
    double value2 = feature->getFeatureValue(splitParameter.getIndex2(),
                                             splitParameter.getFeatureChannel());

    if (value1 < (value2 + tau)) {
        return true;
    } else {
        return false;
    }
}

std::shared_ptr<STIPLeaf> STIPNode::calculateLeafData(
        const std::vector<FeatureRawPtr>& features) const {
    std::vector<STIPLeaf::FeatureInfo> featureInfo;

    auto end = std::end(features);
    for (auto itr = std::begin(features); itr != end; ++itr) {
        featureInfo.push_back(STIPLeaf::FeatureInfo(
                (*itr)->getIndex(), (*itr)->getClassLabel(), (*itr)->getSpatialScale(),
                (*itr)->getTemporalScale(), (*itr)->getDisplacementVector()));
    }

    return std::make_shared<STIPLeaf>(featureInfo);
}

std::shared_ptr<STIPLeaf> STIPNode::loadLeafData(std::queue<std::string>& nodeElements) const {
    auto leaf = std::make_shared<STIPLeaf>();
    leaf->load(nodeElements);

    return leaf;
}
}
}