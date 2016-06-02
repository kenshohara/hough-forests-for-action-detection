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
    int minT = std::numeric_limits<int>::max();
    int maxT = 0;
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

            if (points.at(i)(T) < minT) {
                minT = points.at(i)(T);
            }
            if (points.at(i)(T) > maxT) {
                maxT = points.at(i)(T);
            }
        }
        scaleFeatures.push_back(featuresMap);
    }

    std::cout << "initialize" << std::endl;
    initialize();

    for (int t = minT; t < maxT; ++t) {
        std::vector<std::vector<VoteInfo>> votesInfo;
        for (int scaleIndex = 0; scaleIndex < scaleFeatures.size(); ++scaleIndex) {
            if (scaleFeatures.at(scaleIndex).count(t) == 0) {
                continue;
            }

            calculateVotes(scaleFeatures.at(scaleIndex).at(t), scaleIndex, votesInfo);
        }
        inputInVotingSpace(votesInfo);

        std::vector<std::pair<int, int>> minMaxRanges;
        getMinMaxVotingT(votesInfo, minMaxRanges);

        std::vector<LocalMaxima> localMaxima = findLocalMaxima(minMaxRanges);


    }
}

void HoughForests::initialize() {
    votingSpaces_.clear();
    std::vector<double> scales = parameters_.getScales();
    for (int i = 0; i < parameters_.getNumberOfPositiveClasses(); ++i) {
        votingSpaces_.push_back(VotingSpace(parameters_.getWidth(), parameters_.getHeight(),
                                            scales.size(), scales,
                                            parameters_.getVotesDeleteStep(), parameters_.getVotesBufferLength()));
    }

    std::vector<int> steps = {parameters_.getTemporalStep(), parameters_.getSpatialStep(),
                              parameters_.getSpatialStep()};
    finder_ = LocalMaximaFinder(steps, scales, parameters_.getSigma(), parameters_.getTau(),
                                parameters_.getScaleBandwidth());
}

void HoughForests::calculateVotes(const std::vector<FeaturePtr>& features, int scaleIndex, std::vector<std::vector<VoteInfo>>& votesInfo) const {
    for (const auto& feature : features) {
        std::vector<LeafPtr> leavesData = randomForests_.match(feature);
        votesInfo.push_back(calculateVotes(feature, scaleIndex, leavesData));
    }
}

std::vector<HoughForests::VoteInfo> HoughForests::calculateVotes(const FeaturePtr& feature, int scaleIndex,
                                                                 const std::vector<LeafPtr>& leavesData) const {
    std::vector<VoteInfo> votesInfo;
    for (const auto& leafData : leavesData) {
        auto featuresInfo = leafData->getFeatureInfo();
        double weight = 1.0 / (featuresInfo.size() * leavesData.size());

        for (const auto& featureInfo : featuresInfo) {
            int classLabel = featureInfo.getClassLabel();
            if (classLabel != parameters_.getNegativeLabel()) {
                cv::Vec3i votingPoint = calculateVotingPoint(
                    feature, parameters_.getScale(scaleIndex), featureInfo);
                VoteInfo voteInfo(votingPoint, weight, classLabel, scaleIndex);
                votesInfo.push_back(voteInfo);
            }
        }
    }

    return votesInfo;
}

cv::Vec3i HoughForests::calculateVotingPoint(
        const FeaturePtr& feature, double scale,
        const randomforests::STIPLeaf::FeatureInfo& featureInfo) const {
    cv::Vec3i displacementVector = featureInfo.getDisplacementVector();
    cv::Vec3i votingPoint = feature->getCenterPoint() + displacementVector;
    votingPoint(Y) /= scale;
    votingPoint(X) /= scale;
    return votingPoint;
}

void HoughForests::inputInVotingSpace(const std::vector<std::vector<VoteInfo>>& votesInfo) {
    for (const auto& oneFeatureVotesInfo : votesInfo) {
        for (const auto& voteInfo : oneFeatureVotesInfo) {
            votingSpaces_.at(voteInfo.getClassLabel()).inputVote(voteInfo.getVotingPoint(), voteInfo.getIndex(), voteInfo.getWeight());
        }
    }
}

void HoughForests::getMinMaxVotingT(const std::vector<std::vector<VoteInfo>>& votesInfo, std::vector<std::pair<int, int>>& minMaxRanges) const {
    minMaxRanges.resize(parameters_.getNumberOfPositiveClasses());
    for (auto& oneClassRange : minMaxRanges) {
        int minT = std::numeric_limits<int>::max();
        int maxT = 0;
        oneClassRange = std::make_pair(minT, maxT);
    }
    for (const auto& oneFeatureVotesInfo : votesInfo) {
        for (const auto& voteInfo : oneFeatureVotesInfo) {
            int t = voteInfo.getVotingPoint()(T);
            int classLabel = voteInfo.getClassLabel();
            if (t < minMaxRanges.at(classLabel).first) {
                minMaxRanges.at(classLabel).first = t;
            }
            if (t > minMaxRanges.at(classLabel).second) {
                minMaxRanges.at(classLabel).second = t;
            }
        }
    }
}

std::vector<LocalMaxima> HoughForests::findLocalMaxima(const std::vector<std::pair<int, int>>& minMaxRanges) {
    std::vector<LocalMaxima> localMaxima(parameters_.getNumberOfPositiveClasses());

    typedef std::function<void()> FindingTask;
    std::queue<FindingTask> tasks;
    for (int classLabel = 0; classLabel < localMaxima.size(); ++classLabel) {
        tasks.push([&minMaxRanges, &localMaxima, this, classLabel]() {
            localMaxima.at(classLabel) = findLocalMaxima(votingSpaces_.at(classLabel), minMaxRanges.at(classLabel).first, minMaxRanges.at(classLabel).second);
        });
    }

    thread::threadProcess(tasks, maxNumberOfThreads_);

    return localMaxima;
}

LocalMaxima HoughForests::findLocalMaxima(VotingSpace& votingSpace, int voteStartT, int voteEndT) {
    LocalMaxima localMaxima = finder_.findLocalMaxima(votingSpace, voteStartT, voteEndT);
    LocalMaxima trueLocalMaxima;
    for (const auto& localMaximum : localMaxima) {
        int t = localMaximum.getPoint()(T);
        if (t >= voteStartT && t < voteEndT) {
            trueLocalMaxima.push_back(localMaximum);
        }
    }

    return trueLocalMaxima;
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