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

void HoughForests::detect(LocalFeatureExtractor& extractor, cv::VideoCapture& capture, int fps,
                          bool isVisualizationEnabled, const cv::Size& visualizationSize,
                          std::vector<std::vector<DetectionResult>>& detectionResults) {
    std::cout << "initialize" << std::endl;
    initialize();

    std::deque<cv::Mat3b> video;
    std::vector<std::vector<Cuboid>> fixedDetectionCuboids(
            parameters_.getNumberOfPositiveClasses());
    std::vector<std::vector<Cuboid>> detectionCuboids(parameters_.getNumberOfPositiveClasses());
    std::vector<std::unordered_map<int, std::vector<Cuboid>>> visualizationDetectionCuboids(
            parameters_.getNumberOfPositiveClasses());
    bool isEnded = false;
    bool isFirstRead = true;

    std::thread videoHandlerThread =
            std::thread([this, &capture, &video, fps, isVisualizationEnabled, &visualizationSize,
                         &visualizationDetectionCuboids, &isEnded]() {
                videoHandler(capture, video, fps, isVisualizationEnabled, visualizationSize,
                             visualizationDetectionCuboids, isEnded);
            });
    while (true) {
        std::cout << "t feature: " << extractor.getStoredFeatureBeginT() << std::endl;

        // auto begin = std::chrono::system_clock::now();
        // std::cout << "read" << std::endl;
        int nFrames = isFirstRead ? extractor.getLocalDuration() : extractor.getTStep();
        if (!waitReading(video, isEnded, nFrames)) {
            break;
        }
        std::vector<cv::Mat3b> inputVideo;
        {
            std::lock_guard<std::mutex> lock(videoLock_);

            if (isFirstRead) {
                inputVideo.push_back(video.front().clone());
                extractor.setWidth(inputVideo.front().cols);
                extractor.setHeight(inputVideo.front().rows);
                isFirstRead = false;
            }
            for (int i = 0; i < nFrames; ++i) {
                inputVideo.push_back(video.front().clone());
                video.pop_front();
            }
        }
        // auto readEnd = std::chrono::system_clock::now();
        // std::cout << "read: "
        //<< std::chrono::duration_cast<std::chrono::milliseconds>(readEnd - begin).count()
        //<< std::endl;

        std::vector<std::vector<cv::Vec3i>> scalePoints;
        std::vector<std::vector<std::vector<float>>> scaleDescriptors;
        extractor.extractLocalFeatures(inputVideo, scalePoints, scaleDescriptors);
        // auto featEnd = std::chrono::system_clock::now();
        // std::cout << "extract features: "
        //         << std::chrono::duration_cast<std::chrono::milliseconds>(featEnd -
        //         readEnd).count()
        //         << std::endl;

        std::vector<std::vector<FeaturePtr>> scaleFeatures;
        scaleFeatures.reserve(scalePoints.size());
        for (int scaleIndex = 0; scaleIndex < scalePoints.size(); ++scaleIndex) {
            scaleFeatures.push_back(convertFeatureFormats(scalePoints.at(scaleIndex),
                                                          scaleDescriptors.at(scaleIndex),
                                                          extractor.N_CHANNELS_));
        }

        // auto voteBegin = std::chrono::system_clock::now();
        std::vector<std::pair<std::size_t, std::size_t>> minMaxRanges;
        votingProcess(scaleFeatures, minMaxRanges);
        // auto voteEnd = std::chrono::system_clock::now();
        // std::cout << "voting process: "
        //         << std::chrono::duration_cast<std::chrono::milliseconds>(voteEnd - voteBegin)
        //                    .count()
        //         << std::endl;

        // auto postStart = std::chrono::system_clock::now();
        updateDetectionCuboids(minMaxRanges, detectionCuboids);
        for (int classLabel = 0; classLabel < detectionCuboids.size(); ++classLabel) {
            {
                std::lock_guard<std::mutex> lock(detectionLock_);
                visualizationDetectionCuboids.at(classLabel).clear();
                for (const auto& cuboid : detectionCuboids.at(classLabel)) {
                    int beginT = cuboid.getBeginT();
                    visualizationDetectionCuboids.at(classLabel)[beginT].push_back(cuboid);
                }
            }
        }
        // auto postEnd = std::chrono::system_clock::now();
        // std::cout << "post: "
        //         << std::chrono::duration_cast<std::chrono::milliseconds>(postEnd - postStart)
        //                    .count()
        //         << std::endl;

        for (int classLabel = 0; classLabel < votingSpaces_.size(); ++classLabel) {
            deleteOldVotes(classLabel, minMaxRanges.at(classLabel).second);
            fixOldDetectionCuboids(detectionCuboids.at(classLabel),
                                   fixedDetectionCuboids.at(classLabel),
                                   extractor.getStoredFeatureBeginT());
        }

        // auto end = std::chrono::system_clock::now();
        // std::cout << "one cycle: "
        //         << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
        //         << std::endl
        //         << std::endl;
    }
    videoHandlerThread.join();

    std::cout << "output process" << std::endl;
    detectionResults.resize(detectionCuboids.size());
    for (int classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
        std::copy(std::begin(detectionCuboids.at(classLabel)),
                  std::end(detectionCuboids.at(classLabel)),
                  std::back_inserter(fixedDetectionCuboids.at(classLabel)));
        std::sort(std::begin(fixedDetectionCuboids.at(classLabel)),
                  std::end(fixedDetectionCuboids.at(classLabel)),
                  [](const Cuboid& a, const Cuboid& b) {
                      return a.getLocalMaximum().getValue() > b.getLocalMaximum().getValue();
                  });

        for (const auto& cuboid : fixedDetectionCuboids.at(classLabel)) {
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

bool HoughForests::waitReading(const std::deque<cv::Mat3b>& video, const bool& isEnded,
                               int nFrames) {
    while (true) {
        std::lock_guard<std::mutex> lock(videoLock_);
        if (video.size() >= nFrames) {
            return true;
        }
        if (isEnded) {
            return false;
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
        if (featuresInfo.size() > parameters_.getInvalidLeafSizeThreshold()) {
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

void HoughForests::fixOldDetectionCuboids(std::vector<Cuboid>& detectionCuboids,
                                          std::vector<Cuboid>& fixedDetectionCuboids,
                                          std::size_t videoBeginT) const {
    if (videoBeginT < parameters_.getVotesBufferLength()) {
        return;
    }

    std::size_t tThreshold = videoBeginT - parameters_.getVotesBufferLength();
    std::copy_if(std::begin(detectionCuboids), std::end(detectionCuboids),
                 std::back_inserter(fixedDetectionCuboids),
                 [tThreshold](const Cuboid& cuboid) { return cuboid.getEndT() < tThreshold; });
    auto removeIt = std::remove_if(
            std::begin(detectionCuboids), std::end(detectionCuboids),
            [tThreshold](const Cuboid& cuboid) { return cuboid.getEndT() < tThreshold; });
    detectionCuboids.erase(removeIt, std::end(detectionCuboids));
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

void HoughForests::videoHandler(
        cv::VideoCapture& capture, std::deque<cv::Mat3b>& video, int fps,
        bool isVisualizationEnabled, const cv::Size& visualizationSize,
        const std::vector<std::unordered_map<int, std::vector<Cuboid>>>& detectionCuboids,
        bool& isEnded) {
    using namespace std::chrono;
    const int STEP = 20;

    milliseconds perFrame(static_cast<long long>(1000.0 / fps));
    std::size_t t = 0;
    milliseconds sec(0);
    while (!isEnded) {
        auto begin = high_resolution_clock::now();
        if ((t % STEP) == 0) {
            std::cout << "t: " << t
                      << ", fps: " << 1000.0 / (static_cast<double>(sec.count()) / STEP)
                      << std::endl;
            sec = milliseconds(0);
        }

        cv::Mat frame;
        capture >> frame;

        if (frame.empty()) {
            std::lock_guard<std::mutex> lock(videoLock_);
            isEnded = true;
            break;
        }

        {
            std::lock_guard<std::mutex> lock(videoLock_);
            video.push_back(frame.clone());
        }

        if (isVisualizationEnabled) {
            for (int classLabel = 0; classLabel < detectionCuboids.size(); ++classLabel) {
                std::lock_guard<std::mutex> lock(detectionLock_);

                int visualizationOffset = fps / 2;
                int duration = parameters_.getAverageDuration(classLabel) + visualizationOffset;
                for (int t2 = t - duration; t2 < t; ++t2) {
                    if (detectionCuboids.at(classLabel).count(t2) == 0) {
                        continue;
                    }
                    cv::Mat1b tMat(1, 1);
                    tMat = static_cast<double>(t2 - (t - duration)) / duration * 255;
                    cv::Mat3b color;
                    cv::applyColorMap(tMat, color, cv::COLORMAP_JET);
                    for (const auto& cuboid : detectionCuboids.at(classLabel).at(t2)) {
                        cv::rectangle(frame, cuboid.getRect(), color(0, 0), 3);
                        cv::putText(frame, std::to_string(classLabel), cuboid.getRect().tl(),
                                    cv::FONT_HERSHEY_PLAIN, 2.5, color(0, 0));
                    }
                }

                cv::Mat1f smallSpace =
                        votingSpaces_.at(classLabel).getVotingSpace(std::max(0, int(t)));
                cv::Mat1f originalSpace;
                cv::resize(smallSpace, originalSpace, frame.size(), 0.0, 0.0, cv::INTER_NEAREST);
                originalSpace /= parameters_.getScoreThreshold(classLabel);
                cv::imshow("class: " + std::to_string(classLabel), originalSpace);
            }

            cv::imshow("Original Size", frame);
            cv::Mat resizedFrame;
            cv::resize(frame, resizedFrame, visualizationSize);
            cv::imshow("Large Size", resizedFrame);
            auto key = cv::waitKey(1);
            if (key == 'q') {
                std::lock_guard<std::mutex> lock(videoLock_);
                isEnded = true;
            }
        }

        auto end = begin + milliseconds(perFrame);
        std::this_thread::sleep_until(end);

        ++t;
        auto trueEnd = high_resolution_clock::now();
        sec += duration_cast<milliseconds>(trueEnd - begin);
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