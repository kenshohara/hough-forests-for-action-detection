#define NOMINMAX

#include "HoughForests.h"
#include "ThreadProcess.h"
#include "Utils.h"

#include <omp.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>
#include <vector>

namespace nuisken {
namespace houghforests {

void HoughForests::train(const std::vector<FeaturePtr>& features) {
    randomForests_.train(features, nThreads_);
}

// void HoughForests::classify(LocalFeatureExtractor& extractor) {
// std::cout << "initialize" << std::endl;
// initialize();

// std::vector<LocalMaxima> totalLocalMaxima(parameters_.getNumberOfPositiveClasses());
// while (true) {
//    auto begin = std::chrono::system_clock::now();
//    std::cout << "read" << std::endl;
//    std::vector<std::vector<cv::Vec3i>> scalePoints;
//    std::vector<std::vector<std::vector<float>>> scaleDescriptors;
//    std::vector<cv::Mat3b> video;
//    extractor.extractLocalFeatures(scalePoints, scaleDescriptors, video);
//    auto endd = std::chrono::system_clock::now();
//    std::cout << "extract features: "
//              << std::chrono::duration_cast<std::chrono::milliseconds>(endd - begin).count()
//              << std::endl;
//    if (extractor.isEnd()) {
//        break;
//    }
//    // std::cout << "convert type" << std::endl;
//    std::vector<std::vector<FeaturePtr>> scaleFeatures(scalePoints.size());
//    for (int scaleIndex = 0; scaleIndex < scalePoints.size(); ++scaleIndex) {
//        scaleFeatures.at(scaleIndex).reserve(scalePoints[scaleIndex].size());
//        for (int i = 0; i < scalePoints[scaleIndex].size(); ++i) {
//            cv::Vec3i point = scalePoints.at(scaleIndex).at(i);
//            std::vector<Eigen::MatrixXf> channelFeatures(extractor.N_CHANNELS_);
//            int nChannelFeatures =
//                    scaleDescriptors.at(scaleIndex).at(i).size() / extractor.N_CHANNELS_;
//            for (int channelIndex = 0; channelIndex < channelFeatures.size(); ++channelIndex)
//            {
//                Eigen::MatrixXf feature(1, nChannelFeatures);
//                for (int featureIndex = 0; featureIndex < nChannelFeatures; ++featureIndex) {
//                    int index = channelIndex * nChannelFeatures + featureIndex;
//                    feature.coeffRef(0, featureIndex) =
//                    scaleDescriptors[scaleIndex][i][index];
//                }
//                channelFeatures.at(channelIndex) = feature;
//            }
//            auto feature = std::make_shared<randomforests::STIPNode::FeatureType>(
//                    channelFeatures, point, cv::Vec3i(), std::make_pair(0.0, 0.0), 0);
//            scaleFeatures.at(scaleIndex).push_back(feature);
//        }
//    }

//    std::cout << "calculate votes" << std::endl;
//    std::vector<std::vector<VoteInfo>> votesInfo;
//    for (int scaleIndex = 0; scaleIndex < scaleFeatures.size(); ++scaleIndex) {
//        if (scaleFeatures.at(scaleIndex).empty()) {
//            continue;
//        }

//        calculateVotes(scaleFeatures.at(scaleIndex), scaleIndex, votesInfo);
//    }

//    std::vector<int> overClassCounts(parameters_.getNumberOfPositiveClasses());
//    for (const auto& featureVotesInfo : votesInfo) {
//        std::vector<int> classCounts(parameters_.getNumberOfPositiveClasses());
//        for (const auto& voteInfo : featureVotesInfo) {
//            ++classCounts.at(voteInfo.getClassLabel());
//            ++overClassCounts.at(voteInfo.getClassLabel());
//        }
//        auto it = std::max_element(std::begin(classCounts), std::end(classCounts));
//        int maxLabel = it - std::begin(classCounts);

//        // for (double x : classCounts) {
//        //    std::cout << x << ", ";
//        //}
//        // std::cout << std::endl;
//    }
//    std::cout << std::endl;
//    std::vector<double> prob(parameters_.getNumberOfPositiveClasses());
//    int sum = std::accumulate(std::begin(overClassCounts), std::end(overClassCounts), 0);
//    std::transform(std::begin(overClassCounts), std::end(overClassCounts), std::begin(prob),
//                   [sum](int x) { return static_cast<double>(x) / sum; });
//    for (double p : prob) {
//        std::cout << p << ", ";
//    }
//    std::cout << std::endl << std::endl;
//}
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

    std::cout << "votes" << std::endl;
    std::vector<std::vector<Cuboid>> detectionCuboids(parameters_.getNumberOfPositiveClasses());
    for (int t = minT; t < maxT; ++t) {
        std::cout << "t: " << t << std::endl;

        std::cout << "voting" << std::endl;
        std::vector<std::vector<VoteInfo>> votesInfo;
        for (int scaleIndex = 0; scaleIndex < scaleFeatures.size(); ++scaleIndex) {
            if (scaleFeatures.at(scaleIndex).empty() ||
                scaleFeatures.at(scaleIndex).count(t) == 0) {
                continue;
            }

            calculateVotes(scaleFeatures.at(scaleIndex).at(t), scaleIndex, votesInfo);
        }
        std::cout << "input" << std::endl;
        inputInVotingSpace(votesInfo);

        std::cout << "mm" << std::endl;
        std::vector<std::pair<std::size_t, std::size_t>> minMaxRanges;
        getMinMaxVotingT(votesInfo, minMaxRanges);

        std::cout << "post" << std::endl;
        for (int classLabel = 0; classLabel < detectionCuboids.size(); ++classLabel) {
            // std::vector<Cuboid> cuboids = calculateCuboids(
            //        localMaxima.at(classLabel), parameters_.getAverageAspectRatio(classLabel),
            //        parameters_.getAverageDuration(classLabel));
            // std::copy(std::begin(cuboids), std::end(cuboids),
            //          std::back_inserter(detectionCuboids.at(classLabel)));
            // std::sort(std::begin(detectionCuboids.at(classLabel)),
            //          std::end(detectionCuboids.at(classLabel)),
            //          [](const Cuboid& a, const Cuboid& b) {
            //              return a.getLocalMaximum().getValue() < b.getLocalMaximum().getValue();
            //          });
            // detectionCuboids.at(classLabel) =
            //        performNonMaximumSuppression(detectionCuboids.at(classLabel));
        }

        for (int classLabel = 0; classLabel < votingSpaces_.size(); ++classLabel) {
            deleteOldVotes(classLabel, minMaxRanges.at(classLabel).second);
        }
    }

    std::cout << "output process" << std::endl;
    detectionResults.resize(detectionCuboids.size());
    for (int classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
        for (const auto& cuboid : detectionCuboids.at(classLabel)) {
            cv::Vec4f localMaximumPoint = cuboid.getLocalMaximum().getPoint();
            if (localMaximumPoint(S) > std::numeric_limits<float>::epsilon()) {
                localMaximumPoint(S) = parameters_.getBaseScale() / localMaximumPoint(S);
            } else {
                localMaximumPoint(S) = 0;
            }
            detectionResults.at(classLabel)
                    .emplace_back(
                            LocalMaximum(localMaximumPoint, cuboid.getLocalMaximum().getValue()));
        }
    }
}

void HoughForests::detect(LocalFeatureExtractor& extractor,
                          std::vector<std::vector<DetectionResult>>& detectionResults) {
    std::cout << "initialize" << std::endl;
    initialize();

    std::vector<cv::Mat3b> video;
    // int fps = extractor.getFPS();
    int fps = 20;
    std::size_t videoBeginT = 0;
    std::vector<std::vector<Cuboid>> detectionCuboids(parameters_.getNumberOfPositiveClasses());
    std::vector<std::unordered_map<int, std::vector<Cuboid>>> visualizationDetectionCuboids(
            parameters_.getNumberOfPositiveClasses());
    bool isEnded = false;

    // std::thread visualizationThread(
    //       [this, &video, fps, &videoBeginT, &visualizationDetectionCuboids, &isEnded]() {
    //           visualizeParallel(video, fps, videoBeginT, visualizationDetectionCuboids,
    //           isEnded);
    //       });
    while (true) {
        auto begin = std::chrono::system_clock::now();
        // std::cout << "read" << std::endl;
        std::vector<std::vector<cv::Vec3i>> scalePoints;
        std::vector<std::vector<std::vector<float>>> scaleDescriptors;
        {
            std::lock_guard<std::mutex> lock(m_);
            extractor.extractLocalFeatures(scalePoints, scaleDescriptors, video, videoBeginT);
        }
        auto featEnd = std::chrono::system_clock::now();
        std::cout << "extract features: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(featEnd - begin).count()
                  << std::endl;
        if (extractor.isEnded()) {
            isEnded = true;
            break;
        }

        // std::cout << "convert type" << std::endl;
        std::vector<std::vector<FeaturePtr>> scaleFeatures;
        scaleFeatures.reserve(scalePoints.size());
        for (int scaleIndex = 0; scaleIndex < scalePoints.size(); ++scaleIndex) {
            scaleFeatures.push_back(convertFeatureFormats(scalePoints.at(scaleIndex),
                                                          scaleDescriptors.at(scaleIndex),
                                                          extractor.N_CHANNELS_));
        }
        auto convertEnd = std::chrono::system_clock::now();
        std::cout << "convert features: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(convertEnd - featEnd)
                             .count()
                  << std::endl;

        auto voteBegin = std::chrono::system_clock::now();
        std::vector<std::pair<std::size_t, std::size_t>> minMaxRanges;
        votingProcess(scaleFeatures, minMaxRanges);
        auto voteEnd = std::chrono::system_clock::now();
        std::cout << "voting process: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(voteEnd - voteBegin)
                             .count()
                  << std::endl;

        auto postStart = std::chrono::system_clock::now();
        updateDetectionCuboids(minMaxRanges, detectionCuboids);
        for (int classLabel = 0; classLabel < detectionCuboids.size(); ++classLabel) {
            {
                std::lock_guard<std::mutex> lock(m_);
                visualizationDetectionCuboids.at(classLabel).clear();
                for (const auto& cuboid : detectionCuboids.at(classLabel)) {
                    int beginT = cuboid.getBeginT();
                    visualizationDetectionCuboids.at(classLabel)[beginT].push_back(cuboid);
                }
            }
        }
        auto postEnd = std::chrono::system_clock::now();
        std::cout << "post: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(postEnd - postStart)
                             .count()
                  << std::endl;

        for (int classLabel = 0; classLabel < votingSpaces_.size(); ++classLabel) {
            deleteOldVotes(classLabel, minMaxRanges.at(classLabel).second);
        }

        auto end = std::chrono::system_clock::now();
        std::cout << "one cycle: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                  << std::endl
                  << std::endl;
    }
    // visualizationThread.join();

    std::cout << "output process" << std::endl;
    detectionResults.resize(detectionCuboids.size());
    for (int classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
        for (const auto& cuboid : detectionCuboids.at(classLabel)) {
            cv::Vec4f localMaximumPoint = cuboid.getLocalMaximum().getPoint();
            if (localMaximumPoint(S) > std::numeric_limits<float>::epsilon()) {
                localMaximumPoint(S) = parameters_.getBaseScale() / localMaximumPoint(S);
            } else {
                localMaximumPoint(S) = 0;
            }
            detectionResults.at(classLabel)
                    .emplace_back(
                            LocalMaximum(localMaximumPoint, cuboid.getLocalMaximum().getValue()));
        }
    }
}

void HoughForests::initialize() {
    votingSpaces_.clear();
    std::vector<int> steps = {parameters_.getTemporalStep(), parameters_.getSpatialStep(),
                              parameters_.getSpatialStep()};
    std::vector<double> scales = parameters_.getScales();
    for (int i = 0; i < parameters_.getNumberOfPositiveClasses(); ++i) {
        votingSpaces_.emplace_back(parameters_.getWidth(), parameters_.getHeight(), scales.size(),
                                   scales, steps, parameters_.getBinSizes(), parameters_.getSigma(),
                                   parameters_.getTau(), parameters_.getScaleBandwidth(),
                                   parameters_.getVotesDeleteStep(),
                                   parameters_.getVotesBufferLength());
    }
}

std::vector<HoughForests::FeaturePtr> HoughForests::convertFeatureFormats(
        const std::vector<cv::Vec3i>& points, const std::vector<std::vector<float>>& descriptors,
        int nChannels) const {
    std::vector<FeaturePtr> features;
    features.reserve(points.size());
    for (int i = 0; i < points.size(); ++i) {
        cv::Vec3i point = points.at(i);
        std::vector<Eigen::MatrixXf> channelFeatures(nChannels);
        int nChannelFeatures = descriptors.at(i).size() / nChannels;
        for (int channelIndex = 0; channelIndex < channelFeatures.size(); ++channelIndex) {
            Eigen::MatrixXf feature(1, nChannelFeatures);
            for (int featureIndex = 0; featureIndex < nChannelFeatures; ++featureIndex) {
                int index = channelIndex * nChannelFeatures + featureIndex;
                feature.coeffRef(0, featureIndex) = descriptors[i][index];
            }
            channelFeatures.at(channelIndex) = feature;
        }
        auto feature = std::make_shared<randomforests::STIPNode::FeatureType>(
                channelFeatures, point, cv::Vec3i(), std::make_pair(0.0, 0.0), 0);
        features.push_back(feature);
    }
    return features;
}

void HoughForests::votingProcess(const std::vector<std::vector<FeaturePtr>>& scaleFeatures,
                                 std::vector<std::pair<std::size_t, std::size_t>>& minMaxRanges) {
    minMaxRanges.resize(parameters_.getNumberOfPositiveClasses());
    for (auto& oneClassRange : minMaxRanges) {
        int minT = std::numeric_limits<std::size_t>::max();
        int maxT = 0;
        oneClassRange = std::make_pair(minT, maxT);
    }
    for (int scaleIndex = 0; scaleIndex < scaleFeatures.size(); ++scaleIndex) {
        if (scaleFeatures.at(scaleIndex).empty()) {
            continue;
        }
        auto s1 = std::chrono::system_clock::now();
        std::vector<std::vector<VoteInfo>> votesInfo(scaleFeatures.at(scaleIndex).size());
        calculateVotes(scaleFeatures.at(scaleIndex), scaleIndex, votesInfo);
        auto s2 = std::chrono::system_clock::now();

        inputInVotingSpace(votesInfo);
        auto s3 = std::chrono::system_clock::now();

        getMinMaxVotingT(votesInfo, minMaxRanges);
        auto s4 = std::chrono::system_clock::now();
    }
}

void HoughForests::calculateVotes(const std::vector<FeaturePtr>& features, int scaleIndex,
                                  std::vector<std::vector<VoteInfo>>& votesInfo) const {
    using Task = std::function<void()>;
    std::queue<Task> tasks;
    for (int featureIndex = 0; featureIndex < features.size(); ++featureIndex) {
        tasks.push([this, featureIndex, &features, scaleIndex, &votesInfo]() {
            std::vector<LeafPtr> leavesData;
            randomForests_.match(features.at(featureIndex), leavesData);
            calculateVotes(features.at(featureIndex), scaleIndex, leavesData,
                           votesInfo.at(featureIndex));
        });
    }
    thread::threadProcess(tasks, nThreads_);
}

void HoughForests::calculateVotes(const FeaturePtr& feature, int scaleIndex,
                                  const std::vector<LeafPtr>& leavesData,
                                  std::vector<VoteInfo>& votesInfo) const {
    for (const auto& leafData : leavesData) {
        auto featuresInfo = leafData->getFeatureInfo();
        if (featuresInfo.size() > 300) {
            continue;
        }
        // std::cout << featuresInfo.size() << std::endl;
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
    // std::cout << "n_votes: " << votesInfo.size() << std::endl;
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

void HoughForests::getMinMaxVotingT(
        const std::vector<std::vector<VoteInfo>>& votesInfo,
        std::vector<std::pair<std::size_t, std::size_t>>& minMaxRanges) const {
    for (const auto& oneFeatureVotesInfo : votesInfo) {
        for (const auto& voteInfo : oneFeatureVotesInfo) {
            int t = voteInfo.getVotingPoint()(T);
            t = std::max(t, 0);
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

void HoughForests::updateDetectionCuboids(
        const std::vector<std::pair<std::size_t, std::size_t>>& minMaxRanges,
        std::vector<std::vector<Cuboid>>& detectionCuboids) const {
    using Task = std::function<void()>;
    std::queue<Task> tasks;
    for (int classLabel = 0; classLabel < minMaxRanges.size(); ++classLabel) {
        tasks.push([this, classLabel, &minMaxRanges, &detectionCuboids] {
            updateDetectionCuboids(classLabel, minMaxRanges.at(classLabel),
                                   detectionCuboids.at(classLabel));
        });
    }
    thread::threadProcess(tasks, nThreads_);
}

void HoughForests::updateDetectionCuboids(int classLabel,
                                          const std::pair<std::size_t, std::size_t>& minMaxRanges,
                                          std::vector<Cuboid>& detectionCuboids) const {
    auto s1 = std::chrono::system_clock::now();
    auto gridPoints = votingSpaces_.at(classLabel).getOriginalGridPoints();
    auto votingScores = votingSpaces_.at(classLabel).getGridVotingScores();
    auto s2 = std::chrono::system_clock::now();
    double threshold = parameters_.getScoreThreshold(classLabel);
    LocalMaxima overThresholdPoints;
    for (int pointIndex = 0; pointIndex < gridPoints.size(); ++pointIndex) {
        if (votingScores.at(pointIndex) > threshold) {
            overThresholdPoints.push_back(
                    LocalMaximum(gridPoints.at(pointIndex), votingScores.at(pointIndex)));
        }
    }
    auto s3 = std::chrono::system_clock::now();
    std::vector<Cuboid> cuboids =
            calculateCuboids(overThresholdPoints, parameters_.getAverageAspectRatio(classLabel),
                             parameters_.getAverageDuration(classLabel));
    std::copy(std::begin(cuboids), std::end(cuboids), std::back_inserter(detectionCuboids));
    std::sort(std::begin(detectionCuboids), std::end(detectionCuboids),
              [](const Cuboid& a, const Cuboid& b) {
                  return a.getLocalMaximum().getValue() < b.getLocalMaximum().getValue();
              });
    auto s4 = std::chrono::system_clock::now();
    detectionCuboids = performNonMaximumSuppression(detectionCuboids);
    auto s5 = std::chrono::system_clock::now();
    // std::cout << "get grids: " << std::chrono::duration_cast<std::chrono::milliseconds>(s2 -
    // s1).count() << std::endl;
    // std::cout << "thresholding: " << std::chrono::duration_cast<std::chrono::milliseconds>(s3
    // - s2).count() << std::endl;
    // std::cout << "calc cuboids: " << std::chrono::duration_cast<std::chrono::milliseconds>(s4
    // - s3).count() << std::endl;
    // std::cout << "nms: " << std::chrono::duration_cast<std::chrono::milliseconds>(s5 -
    // s4).count() << std::endl;
}

std::vector<HoughForests::Cuboid> HoughForests::calculateCuboids(const LocalMaxima& localMaxima,
                                                                 double averageAspectRatio,
                                                                 int averageDuration) const {
    std::vector<Cuboid> cuboids;
    cuboids.reserve(localMaxima.size());
    int baseScale = parameters_.getBaseScale();
    std::transform(
            std::begin(localMaxima), std::end(localMaxima), std::back_inserter(cuboids),
            [averageAspectRatio, averageDuration, baseScale](const LocalMaximum& localMaximum) {
                return Cuboid(localMaximum, baseScale, averageAspectRatio, averageDuration);
            });
    return cuboids;
}

std::vector<HoughForests::Cuboid> HoughForests::performNonMaximumSuppression(
        const std::vector<Cuboid>& cuboids) const {
    std::vector<int> indices(cuboids.size());
    std::iota(std::begin(indices), std::end(indices), 0);
    double threshold = parameters_.getIoUThreshold();
    std::vector<Cuboid> afterCuboids;
    while (!indices.empty()) {
        int index = indices.back();
        afterCuboids.push_back(cuboids.at(index));

        auto removeIt = std::remove_if(
                std::begin(indices), std::end(indices),
                [&cuboids, index, threshold](int candidateIndex) {
                    double iou = cuboids.at(index).computeIoU(cuboids.at(candidateIndex));
                    return iou > threshold;
                });
        indices.erase(removeIt, std::end(indices));
    }

    return afterCuboids;
}

void HoughForests::deleteOldVotes(int classLabel, std::size_t voteMaxT) {
    if (votingSpaces_.at(classLabel).binT(voteMaxT) < votingSpaces_.at(classLabel).getMaxT()) {
        return;
    }

    std::cout << "delete votes: class " << classLabel << std::endl;
    votingSpaces_.at(classLabel).deleteOldVotes();
}

void HoughForests::visualizeParallel(
        std::vector<cv::Mat3b>& video, int fps, const std::size_t& videoBeginT,
        const std::vector<std::unordered_map<int, std::vector<Cuboid>>>& detectionCuboids,
        const bool& isEnded) {
    using namespace std::chrono;
    double millisecPerFrame = 1.0 / fps * 1000;
    double opencvWaitKeyTime = 15;
    while (!isEnded) {
        if (!video.empty()) {
            // auto start = system_clock::now();
            std::vector<cv::Mat3b> visVideo;
            visVideo.reserve(video.size());
            {
                std::lock_guard<std::mutex> lock(m_);
                for (int t = 0; t < video.size(); ++t) {
                    cv::Mat3b visFrame = video.at(t).clone();
                    int visT = t + videoBeginT;
                    for (int classLabel = 0; classLabel < detectionCuboids.size(); ++classLabel) {
                        int duration = parameters_.getAverageDuration(classLabel);
                        for (int t2 = visT - duration; t2 < visT; ++t2) {
                            if (detectionCuboids.at(classLabel).count(t2) == 0) {
                                continue;
                            }

                            for (const auto& cuboid : detectionCuboids.at(classLabel).at(t2)) {
                                cv::rectangle(visFrame, cuboid.getRect(), cv::Scalar(0, 0, 255), 3);
                                cv::putText(visFrame, std::to_string(classLabel),
                                            cuboid.getRect().tl(), cv::FONT_HERSHEY_PLAIN, 2.5,
                                            cv::Scalar(0, 0, 255));
                            }
                        }
                    }
                    visVideo.push_back(visFrame);
                }
                video.clear();
            }
            // auto end = system_clock::now();
            // std::cout << "calc vis: " <<
            // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() <<
            // std::endl;

            for (const auto& frame : visVideo) {
                double sleepTime = std::max(0.0, millisecPerFrame - opencvWaitKeyTime);
                std::this_thread::sleep_for(milliseconds(static_cast<long>(sleepTime)));

                cv::imshow("result", frame);
                cv::waitKey(1);
            }
        }
    }
}

void HoughForests::visualize(const std::vector<cv::Mat3b>& video, std::size_t videoBeginT,
                             const std::vector<std::vector<Cuboid>>& detectionCuboids) const {
    int videoEndT = videoBeginT + video.size();
    std::unordered_map<int, std::vector<std::pair<int, cv::Rect>>> visualizeMap;
    for (int classLabel = 0; classLabel < detectionCuboids.size(); ++classLabel) {
        for (const auto& cuboid : detectionCuboids.at(classLabel)) {
            cv::Rect rect = cuboid.getRect();
            std::pair<int, int> range = cuboid.getRange();
            for (int visT = range.first; visT < range.second; ++visT) {
                visualizeMap[visT].push_back(std::make_pair(classLabel, rect));
            }
        }
    }

    for (int t = 0; t < video.size(); ++t) {
        int visT = t + videoBeginT;
        cv::Mat visFrame = video.at(t).clone();
        if (visualizeMap.count(visT) != 0) {
            for (const auto& vis : visualizeMap.at(visT)) {
                cv::rectangle(visFrame, vis.second, cv::Scalar(0, 0, 255), 5);
                cv::putText(visFrame, std::to_string(vis.first), vis.second.tl(),
                            cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 0, 255));
            }
        }
        cv::imshow("vis", visFrame);
        cv::waitKey(5);
    }
}

void HoughForests::visualize(const std::vector<cv::Mat3b>& video, std::size_t videoBeginT,
                             const std::vector<LocalMaxima>& localMaxima) const {
    int videoEndT = videoBeginT + video.size();
    std::unordered_map<int, std::vector<std::pair<int, cv::Point>>> visualizeMap;
    for (int classLabel = 0; classLabel < localMaxima.size(); ++classLabel) {
        for (const auto& localMaximum : localMaxima.at(classLabel)) {
            cv::Vec4f maximumPoint = localMaximum.getPoint();
            int t = maximumPoint(0);
            cv::Point point(maximumPoint(2), maximumPoint(1));

            for (int visT = t - 60; visT < t + 60; ++visT) {
                if (visualizeMap.count(visT) == 0) {
                    visualizeMap[visT] = {std::make_pair(classLabel, point)};
                } else {
                    visualizeMap[visT].push_back(std::make_pair(classLabel, point));
                }
            }
        }
    }

    for (int t = 0; t < video.size(); ++t) {
        int visT = t + videoBeginT;
        cv::Mat visFrame = video.at(t).clone();
        if (visualizeMap.count(visT) != 0) {
            for (const auto& vis : visualizeMap.at(visT)) {
                cv::circle(visFrame, vis.second, 10, cv::Scalar(0, 0, 255), -1);
                cv::putText(visFrame, std::to_string(vis.first), vis.second, cv::FONT_HERSHEY_PLAIN,
                            2.0, cv::Scalar(0, 0, 255));
            }
        }
        cv::imshow("vis", visFrame);
        cv::waitKey(5);
    }
}

std::vector<float> HoughForests::getVotingSpace(int classLabel) const {
    return votingSpaces_.at(classLabel).getGridVotingScores();
}

void HoughForests::visualize(const std::vector<std::vector<float>>& votingSpaces) const {
    auto it = std::max_element(std::begin(votingSpaces), std::end(votingSpaces),
                               [](const std::vector<float>& a, const std::vector<float>& b) {
                                   return *std::max_element(std::begin(a), std::end(a)) <
                                          *std::max_element(std::begin(b), std::end(b));
                               });
    float maxValue = *std::max_element(std::begin(*it), std::end(*it));
    for (int t = 0; t < parameters_.getVotesBufferLength(); t += parameters_.getTemporalStep()) {
        std::vector<cv::Mat1f> visSpaces(votingSpaces.size());
        for (auto& vis : visSpaces) {
            vis = cv::Mat1f(parameters_.getHeight() / parameters_.getSpatialStep(),
                            parameters_.getWidth() / parameters_.getSpatialStep());
            vis = 0.0;
        }
        int index = 0;
        for (int y = 0; y < visSpaces.front().rows; ++y) {
            for (int x = 0; x < visSpaces.front().cols; ++x) {
                for (int classLabel = 0; classLabel < votingSpaces.size(); ++classLabel) {
                    visSpaces.at(classLabel)(y, x) = votingSpaces.at(classLabel).at(index);
                }
                ++index;
            }
        }
        for (int classLabel = 0; classLabel < visSpaces.size(); ++classLabel) {
            visSpaces.at(classLabel) /= maxValue;
            cv::imshow(std::to_string(classLabel), visSpaces.at(classLabel));
        }
        cv::waitKey();
    }
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