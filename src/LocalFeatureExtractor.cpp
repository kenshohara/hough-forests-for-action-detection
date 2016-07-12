#include "LocalFeatureExtractor.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/superres/optical_flow.hpp>

#include <array>
#include <iostream>
#include <numeric>

namespace nuisken {
namespace houghforests {

const int LocalFeatureExtractor::N_CHANNELS_ = 6;

void LocalFeatureExtractor::extractLocalFeatures(std::vector<std::vector<cv::Vec3i>>& scalePoints,
                                                 std::vector<std::vector<Descriptor>> descriptors) {
    readOriginalScaleVideo();
    generateScaledVideos();
    for (int scaleIndex = 0; scaleIndex < scales_.size(); ++scaleIndex) {
        extractFeatures(scaleIndex, 1, scaleVideos_.at(scaleIndex).size());
        std::vector<cv::Vec3i> points;
        std::vector<std::vector<float>> descriptors;
        denseSampling(scaleIndex, points, descriptors);
    }
}

void LocalFeatureExtractor::readOriginalScaleVideo() {
    int nFrames = tStep_;
    if (scaleVideos_.front().empty()) {
        cv::Mat firstFrame;
        videoCapture_ >> firstFrame;
        cv::cvtColor(firstFrame, firstFrame, cv::COLOR_BGR2GRAY);
        scaleVideos_.front().push_back(
                firstFrame);  // add dummy frame for t_derivative and optical flow
        scaleVideos_.front().push_back(firstFrame);

        nFrames = localDuration_ - 1;

        width_ = firstFrame.cols;
        height_ = firstFrame.rows;
        storedStartT_ = 0;
    }

    for (int i = 0; i < nFrames; ++i) {
        cv::Mat frame;
        videoCapture_ >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        scaleVideos_.front().push_back(frame);
    }
}

void LocalFeatureExtractor::generateScaledVideos() {
    for (int scaleIndex = 1; scaleIndex < scales_.size(); ++scaleIndex) {
        int start = scaleVideos_.at(scaleIndex).size();
        for (int i = start; i < scaleVideos_.front().size(); ++i) {
            cv::Mat scaledFrame;
            cv::resize(scaleVideos_.front().at(i), scaledFrame, cv::Size(), scales_.at(scaleIndex),
                       scales_.at(scaleIndex), cv::INTER_CUBIC);
            scaleVideos_.at(scaleIndex).push_back(scaledFrame);
        }
    }
}

void LocalFeatureExtractor::extractFeatures(int scaleIndex, int startFrame, int endFrame) {
    extractIntensityFeature(scaleChannelFeatures_.at(scaleIndex).at(0), scaleIndex, startFrame,
                            endFrame);
    extractXDerivativeFeature(scaleChannelFeatures_.at(scaleIndex).at(1), scaleIndex, startFrame,
                              endFrame);
    extractYDerivativeFeature(scaleChannelFeatures_.at(scaleIndex).at(2), scaleIndex, startFrame,
                              endFrame);
    extractTDerivativeFeature(scaleChannelFeatures_.at(scaleIndex).at(3), scaleIndex, startFrame,
                              endFrame);
    extractFlowFeature(scaleChannelFeatures_.at(scaleIndex).at(4),
                       scaleChannelFeatures_.at(scaleIndex).at(5), scaleIndex, startFrame,
                       endFrame);
}

void LocalFeatureExtractor::deleteOldData() {
    for (auto& video : scaleVideos_) {
        video = {video.back()};
    }

    for (int scaleIndex = 0; scaleIndex < scales_.size(); scaleIndex) {
        int width = width_ * scales_.at(scaleIndex);
        int height = height_ * scales_.at(scaleIndex);

        int featureIndex = calculateFeatureIndex(0, 0, tStep_, width, height);
        for (auto& features : scaleChannelFeatures_.at(scaleIndex)) {
            auto beginIt = std::begin(features);
            auto deleteEndIt = beginIt + featureIndex;
            features.erase(beginIt, deleteEndIt);
        }
    }
    storedStartT_ += tStep_;
}

void LocalFeatureExtractor::extractIntensityFeature(Feature& features, int scaleIndex,
                                                    int startFrame, int endFrame) {
    for (int t = startFrame; t < endFrame; ++t) {
        Feature oneFrameFeature = extractIntensityFeature(scaleVideos_.at(scaleIndex).at(t));
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));
    }
}

void LocalFeatureExtractor::extractXDerivativeFeature(Feature& features, int scaleIndex,
                                                      int startFrame, int endFrame) {
    for (int t = startFrame; t < endFrame; ++t) {
        Feature oneFrameFeature = extractXDerivativeFeature(scaleVideos_.at(scaleIndex).at(t));
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));
    }
}

void LocalFeatureExtractor::extractYDerivativeFeature(Feature& features, int scaleIndex,
                                                      int startFrame, int endFrame) {
    for (int t = startFrame; t < endFrame; ++t) {
        Feature oneFrameFeature = extractYDerivativeFeature(scaleVideos_.at(scaleIndex).at(t));
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));
    }
}

void LocalFeatureExtractor::extractTDerivativeFeature(Feature& features, int scaleIndex,
                                                      int startFrame, int endFrame) {
    cv::Mat prev = scaleVideos_.at(scaleIndex).at(startFrame - 1);
    for (int t = startFrame; t < endFrame; ++t) {
        cv::Mat next = scaleVideos_.at(scaleIndex).at(t);

        Feature oneFrameFeature = extractTDerivativeFeature(prev, next);
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));

        prev = next;
    }
}

void LocalFeatureExtractor::extractFlowFeature(Feature& xFeatures, Feature& yFeatures,
                                               int scaleIndex, int startFrame, int endFrame) {
    cv::Mat prev = scaleVideos_.at(scaleIndex).at(startFrame - 1);
    for (int t = startFrame; t < endFrame; ++t) {
        cv::Mat next = scaleVideos_.at(scaleIndex).at(t);
        std::vector<Feature> feature = extractFlowFeature(prev, next);
        std::copy(std::begin(feature.at(0)), std::end(feature.at(0)),
                  std::back_inserter(xFeatures));
        std::copy(std::begin(feature.at(1)), std::end(feature.at(1)),
                  std::back_inserter(yFeatures));

        prev = next;
    }
}

LocalFeatureExtractor::Feature LocalFeatureExtractor::extractIntensityFeature(
        const cv::Mat1b& frame) const {
    cv::Mat feature = frame.reshape(0, 1);
    feature.convertTo(feature, CV_32F);

    return feature;
}

LocalFeatureExtractor::Feature LocalFeatureExtractor::extractXDerivativeFeature(
        const cv::Mat1b& frame) const {
    cv::Mat dst;
    cv::Sobel(frame, dst, CV_32F, 1, 0);

    cv::Mat feature = dst.reshape(0, 1);
    return feature;
}

LocalFeatureExtractor::Feature LocalFeatureExtractor::extractYDerivativeFeature(
        const cv::Mat1b& frame) const {
    cv::Mat dst;
    cv::Sobel(frame, dst, CV_32F, 0, 1);

    cv::Mat feature = dst.reshape(0, 1);
    return feature;
}

LocalFeatureExtractor::Feature LocalFeatureExtractor::extractTDerivativeFeature(
        const cv::Mat1b& prev, const cv::Mat1b& next) const {
    cv::Mat floatPrev;
    cv::Mat floatNext;
    prev.convertTo(floatPrev, CV_32F);
    next.convertTo(floatNext, CV_32F);

    cv::Mat diff = floatNext - floatPrev;

    cv::Mat feature = diff.reshape(0, 1);
    return feature;
}

std::vector<LocalFeatureExtractor::Feature> LocalFeatureExtractor::extractFlowFeature(
        const cv::Mat1b& prev, const cv::Mat1b& next) const {
    auto flow = cv::superres::createOptFlow_Farneback();

    cv::Mat1f flowX;
    cv::Mat1f flowY;
    flow->calc(prev, next, flowX, flowY);

    flowX = flowX.reshape(0, 1);
    flowY = flowY.reshape(0, 1);
    return std::vector<Feature>{flowX, flowY};
}
void LocalFeatureExtractor::denseSampling(int scaleIndex, std::vector<cv::Vec3i>& points,
                                          std::vector<Descriptor>& descriptors) const {
    int width = width_ * scales_.at(scaleIndex);
    int xEnd = width - localWidth_;
    int height = height_ * scales_.at(scaleIndex);
    int yEnd = height - localHeight_;

    for (int y = 0; y <= yEnd; y += yStep_) {
        for (int x = 0; x <= xEnd; xStep_) {
            points.emplace_back(storedStartT_ + (localDuration_ / 2), y + (localHeight_ / 2),
                                x + (localWidth_ / 2));

            Descriptor neighborhoodFeatures =
                    getLocalFeature(scaleIndex, cv::Vec3i(storedStartT_, y, x), width, height);
            descriptors.push_back(neighborhoodFeatures);
        }
    }
}

LocalFeatureExtractor::Descriptor LocalFeatureExtractor::getLocalFeature(
        int scaleIndex, const cv::Vec3i& topLeftPoint, int width, int height) const {
    Descriptor localFeature;
    int nLocalElements = localWidth_ * localHeight_ * localDuration_;
    localFeature.reserve(nLocalElements * N_CHANNELS_);
    for (int channelIndex = 0; channelIndex < N_CHANNELS_; ++channelIndex) {
        for (int t = 0; t < localDuration_; ++t) {
            for (int y = 0; y < localHeight_; ++y) {
                for (int x = 0; x < localWidth_; ++x) {
                    int featureIndex =
                            calculateFeatureIndex(x + topLeftPoint(X), y + topLeftPoint(Y),
                                                  t + topLeftPoint(T), width, height);
                    localFeature.push_back(
                            scaleChannelFeatures_.at(scaleIndex).at(channelIndex).at(featureIndex));
                }
            }
        }
    }

    return localFeature;
}

int LocalFeatureExtractor::calculateFeatureIndex(int x, int y, int t, int width, int height) const {
    return (t * width * height) + (y * width) + x;
}
}
}