#define NOMINMAX

#include "HoughForests.h"
#include "Utils.h"
#include "ThreadProcess.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <functional>
#include <algorithm>
#include <chrono>

namespace nuisken {
namespace houghforests {

// void HoughForests::train(const std::vector<FeaturePtr>& features) {
//    randomForests_.train(features, maxNumberOfThreads_);
//}

void HoughForests::detect(const std::vector<std::string>& featureFilePaths,
                          std::vector<std::vector<DetectionResult>>& detectionResults) {
    std::vector<std::unordered_map<int, std::vector<FeaturePtr>>> scaleFeatures;
    scaleFeatures.reserve(featureFilePaths.size());
    for (const auto& featureFilePath : featureFilePaths) {
        std::vector<std::vector<Eigen::MatrixXf>> features;
        std::vector<cv::Vec3i> points;
        io::readSTIPFeatures(featureFilePath, features, points);
        std::unordered_map<int, std::vector<FeaturePtr>> featuresMap(features.size());
        for (int i = 0; i < features.size(); ++i) {
            auto feature = std::make_shared<randomforests::STIPNode::FeatureType>(
                    features.at(i), points.at(i), cv::Vec3i(), std::make_pair(0.0, 0.0), 0);
            if (featuresMap.count(points.at(i)(T)) == 0) {
                featuresMap.insert(
                        std::make_pair(points.at(i)(T), std::vector<FeaturePtr>{feature}));
            } else {
                featuresMap.at(points.at(i)(T)).push_back(feature);
            }
        }
        scaleFeatures.push_back(featuresMap);
    }

    std::vector<LocalMaxima> localMaxima;

    std::cout << "initialize" << std::endl;
    initializeMeanShifts();

    std::cout << "vote" << std::endl;
    VotesInfoMap votesInfoMap;
    for (int scaleIndex = 0; scaleIndex < scaleFeatures.size(); ++scaleIndex) {
        calculateVotes(scaleFeatures.at(scaleIndex), scaleIndex, votesInfoMap);
    }
    std::cout << "input mean shift" << std::endl;
    inputToMeanShift(votesInfoMap);

    //“Š•[Œã‚Ìˆ—
    //•½ŠŠ‰»C‹É’lŒŸo
    std::cout << "find local maxima" << std::endl;
    for (auto& meanShift : meanShifts_) {
        if (meanShift->getDataPointsSize() != 0) {
            meanShift->buildTree();
        }
    }
    localMaxima = findLocalMaxima();
    localMaxima = verifyLocalMaxima(votesInfoMap, localMaxima);
    localMaxima = thresholdLocalMaxima(std::move(localMaxima));

    std::cout << "post process" << std::endl;
    detectionResults = postProcess(localMaxima, votesInfoMap);
}

void HoughForests::initializeMeanShifts() {
    meanShifts_.clear();
    for (int i = 0; i < parameters_.getNumberOfPositiveClasses(); ++i) {
        meanShifts_.push_back(std::make_unique<MSType>(
                parameters_.getSizes(), parameters_.getScales(), parameters_.getSigma(),
                parameters_.getTau(), parameters_.getScaleBandwidth()));
    }
}

void HoughForests::calculateVotes(const std::vector<FeaturePtr>& features, int scaleIndex,
                                  VotesInfoMap& votesInfoMap) const {
    for (const auto& feature : features) {
        auto votingData = randomForests_.match(feature);
        calculateVotesBasedOnOnePatch(feature, scaleIndex, votingData, votesInfoMap);
    }
}

void HoughForests::calculateVotesBasedOnOnePatch(const FeaturePtr& feature, int scaleIndex,
                                                 const std::vector<LeafPtr>& votingData,
                                                 VotesInfoMap& votesInfoMap) const {
    std::vector<std::vector<VoteInfo>> treeVotesInfo;
    treeVotesInfo.reserve(votingData.size());

    for (const auto& aVotingData : votingData) {
        auto featureInfo = aVotingData->getFeatureInfo();
        auto weight = 1.0 / (featureInfo.size() * votingData.size());

        // std::cout << featureInfo.size() << std::endl;

        std::vector<VoteInfo> votesInfo;
        for (const auto& aFeatureInfo : featureInfo) {
            int classLabel = aFeatureInfo.getClassLabel();
            if (classLabel != parameters_.getNegativeLabel()) {
                cv::Vec3f votingPoint = calculateVotingPoint(
                        feature, parameters_.getScale(scaleIndex), aFeatureInfo);
                VoteInfo voteInfo(votingPoint, weight, classLabel, scaleIndex);
                votesInfo.push_back(voteInfo);
            }
        }

        treeVotesInfo.push_back(std::move(votesInfo));
    }

    votesInfoMap.setFeatureVoteInfo(feature->getCenterPoint()[T],
                                    FeatureVoteInfo(feature->getCenterPoint(), treeVotesInfo));
}

void HoughForests::inputToMeanShift(VotesInfoMap& votesInfoMap) {
    typedef std::pair<int, std::vector<FeatureVoteInfo>> FrameAndFeaturesVotesInfo;
    for (const FrameAndFeaturesVotesInfo& featuresVotesInfo : votesInfoMap) {
        for (const FeatureVoteInfo& featureVotesInfo : featuresVotesInfo.second) {
            for (const auto& votesInfo : featureVotesInfo.getTreeVotesInfo()) {
                for (const auto& voteInfo : votesInfo) {
                    meanShifts_.at(voteInfo.getClassLabel())
                            ->addInput(voteInfo.getVotingPoint(), voteInfo.getIndex(),
                                       voteInfo.getWeight());
                }
            }
        }
    }
}

cv::Vec3f HoughForests::calculateVotingPoint(
        const FeaturePtr& feature, double scale,
        const randomforests::STIPLeaf::FeatureInfo& featureInfo) const {
    cv::Vec3i displacementVector = featureInfo.getDisplacementVector();
    cv::Vec3f votingPoint = feature->getCenterPoint() + displacementVector;
    votingPoint(Y) /= scale;
    votingPoint(X) /= scale;
    return votingPoint;
}

std::vector<LocalMaxima> HoughForests::findLocalMaxima() {
    std::vector<LocalMaxima> localMaxima(parameters_.getNumberOfPositiveClasses());

    typedef std::function<void()> FindingTask;
    std::queue<FindingTask> tasks;
    for (int i = 0; i < localMaxima.size(); ++i) {
        if (meanShifts_.at(i)->isBuild()) {
            tasks.push([&, this, i]() {
                localMaxima.at(i) = findOneClassLocalMaxima(meanShifts_.at(i));
            });
        }
    }

    thread::threadProcess(tasks, maxNumberOfThreads_);

    return localMaxima;
}

LocalMaxima HoughForests::findOneClassLocalMaxima(std::unique_ptr<MSType>& meanShift) {
    auto start = std::chrono::system_clock::now();
    std::vector<cv::Vec4f> gridPoints = prepareGridPoints();

    LocalMaxima localMaxima = meanShift->findMode(gridPoints);

    LocalMaxima refinedLocalMaxima;
    refinedLocalMaxima.reserve(localMaxima.size());
    for (const auto& localMaximum : localMaxima) {
        refinedLocalMaxima.push_back(meanShift->findMode(localMaximum.getPoint()));
    }

    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << std::endl;
    return refinedLocalMaxima;
}

std::vector<cv::Vec4f> HoughForests::prepareGridPoints() const {
    int maxTime = 0;
    int minTime = std::numeric_limits<int>::max();
    for (const auto& meanShift : meanShifts_) {
        int tmpMaxTime = meanShift->getMaxIndexPoint()(T);
        int tmpMinTime = meanShift->getMinIndexPoint()(T);

        if (tmpMaxTime > maxTime) {
            maxTime = tmpMaxTime;
        }
        if (tmpMinTime < minTime) {
            minTime = tmpMinTime;
        }
    }

    std::vector<cv::Vec4f> gridPoints;
    for (int t = minTime; t <= maxTime; t += parameters_.getTemporalStep()) {
        for (int y = 0; y < parameters_.getSize(Y); y += parameters_.getSpatialStep()) {
            for (int x = 0; x < parameters_.getSize(X); x += parameters_.getSpatialStep()) {
                for (double s : parameters_.getScales()) {
                    gridPoints.push_back(cv::Vec4f(t, y, x, s));
                }
            }
        }
    }

    return gridPoints;
}

std::vector<LocalMaxima> HoughForests::verifyLocalMaxima(
        const VotesInfoMap& votesInfoMap, const std::vector<LocalMaxima>& localMaxima) const {
    return localMaxima;
}

std::vector<LocalMaxima> HoughForests::thresholdLocalMaxima(
        std::vector<LocalMaxima> localMaxima) const {
    std::vector<LocalMaxima> thresholdedLocalMaxima(localMaxima.size());
    for (int classLabel = 0; classLabel < localMaxima.size(); ++classLabel) {
        std::sort(std::begin(localMaxima.at(classLabel)), std::end(localMaxima.at(classLabel)),
                  [](const LocalMaximum& maximum, const LocalMaximum& other) {
                      return maximum.getValue() > other.getValue();
                  });

        for (int i = 0; i < std::min(static_cast<int>(localMaxima.at(classLabel).size()),
                                     parameters_.getLocalMaximaSize());
             ++i) {
            thresholdedLocalMaxima.at(classLabel).push_back(localMaxima.at(classLabel).at(i));
        }
    }

    return thresholdedLocalMaxima;
}

void HoughForests::save(const std::string& directoryPath) const {
    randomForests_.save(directoryPath);
}

void HoughForests::load(const std::string& directoryPath) {
    stipNode_.setNumberOfClasses(parameters_.getNumberOfClasses());
    randomForests_.initForests();
    randomForests_.setType(stipNode_);
    randomForests_.load(directoryPath);
}
}
}