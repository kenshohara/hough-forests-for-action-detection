#define NOMINMAX

#include "HoughForests.h"
#include "ThreadProcess.h"
#include "Utils.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace nuisken {
namespace houghforests {

void HoughForests::train(const std::vector<FeaturePtr>& features) {
    randomForests_.train(features, nThreads_);
}

void HoughForests::detect(LocalFeatureExtractor& extractor) {
    std::cout << "initialize" << std::endl;
    initialize();

    while (true) {
        std::cout << "read" << std::endl;
        std::vector<std::vector<cv::Vec3i>> scalePoints;
        std::vector<std::vector<std::vector<float>>> scaleDescriptors;
        extractor.extractLocalFeatures(scalePoints, scaleDescriptors);
        if (extractor.isEnd()) {
            break;
        }
        std::vector<std::unordered_map<int, std::vector<FeaturePtr>>> scaleFeatures;
        int minT = std::numeric_limits<int>::max();
        int maxT = 0;
        for (int scaleIndex = 0; scaleIndex < scalePoints.size(); ++scaleIndex) {
            std::unordered_map<int, std::vector<FeaturePtr>> featuresMap(
                    scalePoints[scaleIndex].size());
            for (int i = 0; i < scalePoints[scaleIndex].size(); ++i) {
                cv::Vec3i point = scalePoints.at(scaleIndex).at(i);
                std::vector<Eigen::MatrixXf> features(extractor.N_CHANNELS_);
                int nChannelFeatures =
                        scaleDescriptors.at(scaleIndex).at(i).size() / extractor.N_CHANNELS_;
                for (int channelIndex = 0; channelIndex < features.size(); ++channelIndex) {
                    Eigen::MatrixXf feature(1, nChannelFeatures);
                    for (int featureIndex = 0; featureIndex < nChannelFeatures; ++featureIndex) {
                        int index = channelIndex * nChannelFeatures + featureIndex;
                        feature.coeffRef(0, featureIndex) = scaleDescriptors[scaleIndex][i][index];
                    }
                    features.at(channelIndex) = feature;
                }
                auto feature = std::make_shared<randomforests::STIPNode::FeatureType>(
                        features, point, cv::Vec3i(), std::make_pair(0.0, 0.0), 0);
                if (featuresMap.count(point(T)) == 0) {
                    featuresMap.insert(std::make_pair(point(T), std::vector<FeaturePtr>{feature}));
                } else {
                    featuresMap.at(point(T)).push_back(feature);
                }

                if (point(T) < minT) {
                    minT = point(T);
                }
                if (point(T) > maxT) {
                    maxT = point(T);
                }
            }
            scaleFeatures.push_back(featuresMap);
        }

        std::cout << "votes" << std::endl;
        std::vector<LocalMaxima> totalLocalMaxima(parameters_.getNumberOfPositiveClasses());
        for (int t = minT; t < maxT; ++t) {
            std::cout << "t: " << t << std::endl;
            auto start = std::chrono::system_clock::now();

            std::vector<std::vector<VoteInfo>> votesInfo;
            for (int scaleIndex = 0; scaleIndex < scaleFeatures.size(); ++scaleIndex) {
                if (scaleFeatures.at(scaleIndex).count(t) == 0) {
                    continue;
                }

                calculateVotes(scaleFeatures.at(scaleIndex).at(t), scaleIndex, votesInfo);
            }
            auto inputStart = std::chrono::system_clock::now();
            inputInVotingSpace(votesInfo);

            auto calcMMStart = std::chrono::system_clock::now();
            std::vector<std::pair<int, int>> minMaxRanges;
            getMinMaxVotingT(votesInfo, minMaxRanges);

            auto findStart = std::chrono::system_clock::now();
            std::vector<LocalMaxima> localMaxima = findLocalMaxima(minMaxRanges);
            auto threshStart = std::chrono::system_clock::now();
            localMaxima = thresholdLocalMaxima(localMaxima);

            auto combineStart = std::chrono::system_clock::now();
            for (int classLabel = 0; classLabel < totalLocalMaxima.size(); ++classLabel) {
                LocalMaxima oneClassLocalMaxima(totalLocalMaxima.at(classLabel));
                std::copy(std::begin(localMaxima.at(classLabel)),
                          std::end(localMaxima.at(classLabel)),
                          std::back_inserter(oneClassLocalMaxima));
                totalLocalMaxima.at(classLabel) =
                        finder_.combineNeighborLocalMaxima(oneClassLocalMaxima);
            }

            for (int classLabel = 0; classLabel < votingSpaces_.size(); ++classLabel) {
                deleteOldVotes(classLabel, minMaxRanges.at(classLabel).second);
            }

            auto end = std::chrono::system_clock::now();
            std::cout << "one cycle: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                      << std::endl;
        }
    }
}

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

    std::cout << "votes" << std::endl;
    std::vector<LocalMaxima> totalLocalMaxima(parameters_.getNumberOfPositiveClasses());
    for (int t = minT; t < maxT; ++t) {
        std::cout << "t: " << t << std::endl;
        auto start = std::chrono::system_clock::now();

        std::vector<std::vector<VoteInfo>> votesInfo;
        for (int scaleIndex = 0; scaleIndex < scaleFeatures.size(); ++scaleIndex) {
            if (scaleFeatures.at(scaleIndex).count(t) == 0) {
                continue;
            }

            calculateVotes(scaleFeatures.at(scaleIndex).at(t), scaleIndex, votesInfo);
        }
        auto inputStart = std::chrono::system_clock::now();
        inputInVotingSpace(votesInfo);

        auto calcMMStart = std::chrono::system_clock::now();
        std::vector<std::pair<int, int>> minMaxRanges;
        getMinMaxVotingT(votesInfo, minMaxRanges);

        auto findStart = std::chrono::system_clock::now();
        std::vector<LocalMaxima> localMaxima = findLocalMaxima(minMaxRanges);
        auto threshStart = std::chrono::system_clock::now();
        localMaxima = thresholdLocalMaxima(localMaxima);

        auto combineStart = std::chrono::system_clock::now();
        for (int classLabel = 0; classLabel < totalLocalMaxima.size(); ++classLabel) {
            LocalMaxima oneClassLocalMaxima(totalLocalMaxima.at(classLabel));
            std::copy(std::begin(localMaxima.at(classLabel)), std::end(localMaxima.at(classLabel)),
                      std::back_inserter(oneClassLocalMaxima));
            totalLocalMaxima.at(classLabel) =
                    finder_.combineNeighborLocalMaxima(oneClassLocalMaxima);
        }

        for (int classLabel = 0; classLabel < votingSpaces_.size(); ++classLabel) {
            deleteOldVotes(classLabel, minMaxRanges.at(classLabel).second);
        }

        auto end = std::chrono::system_clock::now();
        std::cout << "one cycle: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << std::endl;
    }

    std::cout << "output process" << std::endl;
    detectionResults.resize(totalLocalMaxima.size());
    for (int classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
        for (const auto& localMaximum : totalLocalMaxima.at(classLabel)) {
            detectionResults.at(classLabel).emplace_back(localMaximum);
        }
    }
}

void HoughForests::initialize() {
    votingSpaces_.clear();
    std::vector<double> scales = parameters_.getScales();
    for (int i = 0; i < parameters_.getNumberOfPositiveClasses(); ++i) {
        votingSpaces_.emplace_back(parameters_.getWidth(), parameters_.getHeight(), scales.size(),
                                   scales, parameters_.getVotesDeleteStep(),
                                   parameters_.getVotesBufferLength());
    }

    std::vector<int> steps = {parameters_.getTemporalStep(), parameters_.getSpatialStep(),
                              parameters_.getSpatialStep()};
    finder_ = LocalMaximaFinder(steps, scales, parameters_.getSigma(), parameters_.getTau(),
                                parameters_.getScaleBandwidth());
}

void HoughForests::calculateVotes(const std::vector<FeaturePtr>& features, int scaleIndex,
                                  std::vector<std::vector<VoteInfo>>& votesInfo) const {
    for (const auto& feature : features) {
        std::vector<LeafPtr> leavesData = randomForests_.match(feature);
        votesInfo.push_back(calculateVotes(feature, scaleIndex, leavesData));
    }
}

std::vector<HoughForests::VoteInfo> HoughForests::calculateVotes(
        const FeaturePtr& feature, int scaleIndex, const std::vector<LeafPtr>& leavesData) const {
    std::vector<VoteInfo> votesInfo;
    for (const auto& leafData : leavesData) {
        auto featuresInfo = leafData->getFeatureInfo();
        double weight = 1.0 / (featuresInfo.size() * leavesData.size());
        for (const auto& featureInfo : featuresInfo) {
            int classLabel = featureInfo.getClassLabel();
            if (classLabel != parameters_.getNegativeLabel()) {
                cv::Vec3i votingPoint = calculateVotingPoint(
                        feature, parameters_.getScale(scaleIndex), featureInfo);
                votesInfo.emplace_back(votingPoint, weight, classLabel, scaleIndex);
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
            votingSpaces_.at(voteInfo.getClassLabel())
                    .inputVote(voteInfo.getVotingPoint(), voteInfo.getIndex(),
                               voteInfo.getWeight());
        }
    }
}

void HoughForests::getMinMaxVotingT(const std::vector<std::vector<VoteInfo>>& votesInfo,
                                    std::vector<std::pair<int, int>>& minMaxRanges) const {
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

std::vector<LocalMaxima> HoughForests::findLocalMaxima(
        const std::vector<std::pair<int, int>>& minMaxRanges) {
    std::vector<LocalMaxima> localMaxima(parameters_.getNumberOfPositiveClasses());

    typedef std::function<void()> FindingTask;
    std::queue<FindingTask> tasks;
    for (int classLabel = 0; classLabel < localMaxima.size(); ++classLabel) {
        if (minMaxRanges.at(classLabel).first > minMaxRanges.at(classLabel).second) {
            continue;
        }

        tasks.push([&minMaxRanges, &localMaxima, this, classLabel]() {
            localMaxima.at(classLabel) = findLocalMaxima(
                    votingSpaces_.at(classLabel), parameters_.getScoreThreshold(classLabel),
                    minMaxRanges.at(classLabel).first, minMaxRanges.at(classLabel).second);
        });
    }

    thread::threadProcess(tasks, nThreads_);

    return localMaxima;
}

LocalMaxima HoughForests::findLocalMaxima(VotingSpace& votingSpace, double scoreThreshold,
                                          int voteStartT, int voteEndT) {
    LocalMaxima localMaxima =
            finder_.findLocalMaxima(votingSpace, scoreThreshold, voteStartT, voteEndT);
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

        for (int i = 0; i < localMaxima.at(classLabel).size(); ++i) {
            if (localMaxima.at(classLabel).at(i).getValue() <
                parameters_.getScoreThreshold(classLabel)) {
                break;
            }

            thresholdedLocalMaxima.at(classLabel).push_back(localMaxima.at(classLabel).at(i));
        }
    }

    return thresholdedLocalMaxima;
}

void HoughForests::deleteOldVotes(int classLabel, int voteMaxT) {
    if (voteMaxT < votingSpaces_.at(classLabel).getMaxT()) {
        return;
    }

    std::cout << "delete votes: class " << classLabel << std::endl;
    std::cout << "before: " << votingSpaces_.at(classLabel).getVotesCount() << std::endl;

    votingSpaces_.at(classLabel).deleteOldVotes();

    std::cout << "after: " << votingSpaces_.at(classLabel).getVotesCount() << std::endl;
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