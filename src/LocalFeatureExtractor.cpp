#include "LocalFeatureExtractor.h"
#include "ThreadProcess.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/superres/optical_flow.hpp>

#include <array>
#include <iostream>
#include <numeric>

namespace nuisken {
namespace houghforests {

const int LocalFeatureExtractor::N_CHANNELS_ = 4;

void LocalFeatureExtractor::makeLocalSizeOdd(int& size) const {
    if ((size % 2) == 0) {
        size++;
    }
}

void LocalFeatureExtractor::extractLocalFeatures(
        std::vector<std::vector<cv::Vec3i>>& scalePoints,
        std::vector<std::vector<Descriptor>>& scaleDescriptors) {
    readOriginalScaleVideo();
    extraction(scalePoints, scaleDescriptors);
}

void LocalFeatureExtractor::extractLocalFeatures(
        const ColorVideo& video, std::vector<std::vector<cv::Vec3i>>& scalePoints,
        std::vector<std::vector<Descriptor>>& scaleDescriptors) {
    inputNewScaleVideo(video);
    extraction(scalePoints, scaleDescriptors);
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
    }

    for (int i = 0; i < nFrames; ++i) {
        cv::Mat frame;
        videoCapture_ >> frame;
        if (frame.empty()) {
            isEnded_ = true;
            break;
        }
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        scaleVideos_.front().push_back(frame);
    }
}

void LocalFeatureExtractor::inputNewScaleVideo(const ColorVideo& video) {
    for (const auto& frame : video) {
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        scaleVideos_.front().push_back(grayFrame);
    }
}

void LocalFeatureExtractor::extraction(std::vector<std::vector<cv::Vec3i>>& scalePoints,
                                       std::vector<std::vector<Descriptor>>& scaleDescriptors) {
    generateScaledVideos();
    for (int scaleIndex = 0; scaleIndex < scales_.size(); ++scaleIndex) {
        int beginFrame = 1;
        int endFrame = scaleVideos_[scaleIndex].size();
        extractFeatures(scaleIndex, beginFrame, endFrame);
        if (scaleIndex == 0) {
            nStoredFeatureFrames_ += endFrame - beginFrame;
        }
        if (nStoredFeatureFrames_ < localDuration_) {
            isEnded_ = true;
            return;
        }

        std::vector<cv::Vec3i> points;
        std::vector<std::vector<float>> descriptors;
        denseSampling(scaleIndex, points, descriptors);
        scalePoints.push_back(points);
        scaleDescriptors.push_back(descriptors);
    }

    deleteOldData();
}

void LocalFeatureExtractor::generateScaledVideos() {
    for (int scaleIndex = 1; scaleIndex < scales_.size(); ++scaleIndex) {
        int begin = scaleVideos_[scaleIndex].size();
        for (int i = begin; i < scaleVideos_.front().size(); ++i) {
            cv::Mat scaledFrame;
            cv::resize(scaleVideos_.front()[i], scaledFrame, cv::Size(), scales_[scaleIndex],
                       scales_[scaleIndex], cv::INTER_CUBIC);
            scaleVideos_[scaleIndex].push_back(scaledFrame);
        }
    }
}

void LocalFeatureExtractor::extractFeatures(int scaleIndex, int beginFrame, int endFrame) {
    extractIntensityFeature(scaleChannelFeatures_[scaleIndex][0],
                            scaleChannelIntegrals_[scaleIndex][0], scaleIndex, beginFrame,
                            endFrame);
    extractXDerivativeFeature(scaleChannelFeatures_[scaleIndex][1],
                              scaleChannelIntegrals_[scaleIndex][1], scaleIndex, beginFrame,
                              endFrame);
    extractYDerivativeFeature(scaleChannelFeatures_[scaleIndex][2],
                              scaleChannelIntegrals_[scaleIndex][2], scaleIndex, beginFrame,
                              endFrame);
    extractTDerivativeFeature(scaleChannelFeatures_[scaleIndex][3],
                              scaleChannelIntegrals_[scaleIndex][3], scaleIndex, beginFrame,
                              endFrame);
    // extractFlowFeature(scaleChannelFeatures_[scaleIndex][4],
    //					scaleChannelFeatures_[scaleIndex][5],
    //					scaleIndex, beginFrame, endFrame);
}

void LocalFeatureExtractor::deleteOldData() {
    if (localDuration_ <= tStep_) {
        scaleVideos_ = std::vector<Video>(scales_.size());
        scaleChannelFeatures_ =
                std::vector<MultiChannelFeature>(scales_.size(), MultiChannelFeature(N_CHANNELS_));
        storedFeatureBeginT_ += tStep_;
        nStoredFeatureFrames_ = 0;

        return;
    }

    for (auto& video : scaleVideos_) {
        video = {video.back()};
    }

    for (int scaleIndex = 0; scaleIndex < scales_.size(); ++scaleIndex) {
        int width = width_ * scales_[scaleIndex];
        int height = height_ * scales_[scaleIndex];

        for (int channelIndex = 0; channelIndex < N_CHANNELS_; ++channelIndex) {
            auto beginIt = std::begin(scaleChannelFeatures_[scaleIndex][channelIndex]);
            auto deleteEndIt = beginIt + tStep_;
            scaleChannelFeatures_[scaleIndex][channelIndex].erase(beginIt, deleteEndIt);

            auto integralBeginIt = std::begin(scaleChannelIntegrals_[scaleIndex][channelIndex]);
            auto integralDeleteEndIt = integralBeginIt + tStep_;
            scaleChannelIntegrals_[scaleIndex][channelIndex].erase(integralBeginIt,
                                                                   integralDeleteEndIt);
        }
    }
    storedFeatureBeginT_ += tStep_;
    nStoredFeatureFrames_ -= tStep_;
}

void LocalFeatureExtractor::extractIntensityFeature(Feature& features, Feature& integrals,
                                                    int scaleIndex, int beginFrame, int endFrame) {
    features.reserve(features.size() + (endFrame - beginFrame));
    integrals.reserve(integrals.size() + (endFrame - beginFrame));
    for (int t = beginFrame; t < endFrame; ++t) {
        cv::Mat1f oneFrameFeature;
        extractIntensityFeature(scaleVideos_[scaleIndex][t], oneFrameFeature);
        features.push_back(oneFrameFeature);

        cv::Mat oneFrameIntegral;
        cv::integral(cv::Mat(oneFrameFeature), oneFrameIntegral, CV_32F);
        integrals.push_back(oneFrameIntegral);
    }
}

void LocalFeatureExtractor::extractXDerivativeFeature(Feature& features, Feature& integrals,
                                                      int scaleIndex, int beginFrame,
                                                      int endFrame) {
    features.reserve(features.size() + (endFrame - beginFrame));
    integrals.reserve(integrals.size() + (endFrame - beginFrame));
    for (int t = beginFrame; t < endFrame; ++t) {
        cv::Mat1f oneFrameFeature;
        extractXDerivativeFeature(scaleVideos_[scaleIndex][t], oneFrameFeature);
        features.push_back(oneFrameFeature);

        cv::Mat oneFrameIntegral;
        cv::integral(cv::Mat(oneFrameFeature), oneFrameIntegral, CV_32F);
        integrals.push_back(oneFrameIntegral);
    }
}

void LocalFeatureExtractor::extractYDerivativeFeature(Feature& features, Feature& integrals,
                                                      int scaleIndex, int beginFrame,
                                                      int endFrame) {
    features.reserve(features.size() + (endFrame - beginFrame));
    integrals.reserve(integrals.size() + (endFrame - beginFrame));
    for (int t = beginFrame; t < endFrame; ++t) {
        cv::Mat1f oneFrameFeature;
        extractYDerivativeFeature(scaleVideos_[scaleIndex][t], oneFrameFeature);
        features.push_back(oneFrameFeature);

        cv::Mat oneFrameIntegral;
        cv::integral(cv::Mat(oneFrameFeature), oneFrameIntegral, CV_32F);
        integrals.push_back(oneFrameIntegral);
    }
}

void LocalFeatureExtractor::extractTDerivativeFeature(Feature& features, Feature& integrals,
                                                      int scaleIndex, int beginFrame,
                                                      int endFrame) {
    features.reserve(features.size() + (endFrame - beginFrame));
    integrals.reserve(integrals.size() + (endFrame - beginFrame));

    cv::Mat prev = scaleVideos_[scaleIndex][beginFrame - 1];
    for (int t = beginFrame; t < endFrame; ++t) {
        cv::Mat next = scaleVideos_[scaleIndex][t];

        cv::Mat1f oneFrameFeature;
        extractTDerivativeFeature(prev, next, oneFrameFeature);
        features.push_back(oneFrameFeature);

        cv::Mat oneFrameIntegral;
        cv::integral(cv::Mat(oneFrameFeature), oneFrameIntegral, CV_32F);
        integrals.push_back(oneFrameIntegral);

        prev = next;
    }
}

void LocalFeatureExtractor::extractFlowFeature(Feature& xFeatures, Feature& yFeatures,
                                               Feature& xIntegrals, Feature& yIntegrals,
                                               int scaleIndex, int beginFrame, int endFrame) {
    xFeatures.reserve(xFeatures.size() + (endFrame - beginFrame));
    yFeatures.reserve(yFeatures.size() + (endFrame - beginFrame));
    xIntegrals.reserve(xIntegrals.size() + (endFrame - beginFrame));
    yIntegrals.reserve(yIntegrals.size() + (endFrame - beginFrame));

    cv::Mat prev = scaleVideos_[scaleIndex][beginFrame - 1];
    for (int t = beginFrame; t < endFrame; ++t) {
        cv::Mat next = scaleVideos_[scaleIndex][t];
        std::vector<cv::Mat1f> features;
        extractFlowFeature(prev, next, features);
        xFeatures.push_back(features.at(0));
        yFeatures.push_back(features.at(1));

        cv::Mat xIntegral;
        cv::integral(cv::Mat(features.at(0)), xIntegral, CV_32F);
        cv::Mat yIntegral;
        cv::integral(cv::Mat(features.at(1)), yIntegral, CV_32F);
        xIntegrals.push_back(xIntegral);
        yIntegrals.push_back(yIntegral);

        prev = next;
    }
}

void LocalFeatureExtractor::extractIntensityFeature(const cv::Mat1b& frame,
                                                    cv::Mat1f& feature) const {
    frame.convertTo(feature, CV_32F);
}

void LocalFeatureExtractor::extractXDerivativeFeature(const cv::Mat1b& frame,
                                                      cv::Mat1f& feature) const {
    cv::Sobel(frame, feature, CV_32F, 1, 0);
}

void LocalFeatureExtractor::extractYDerivativeFeature(const cv::Mat1b& frame,
                                                      cv::Mat1f& feature) const {
    cv::Sobel(frame, feature, CV_32F, 0, 1);
}

void LocalFeatureExtractor::extractTDerivativeFeature(const cv::Mat1b& prev, const cv::Mat1b& next,
                                                      cv::Mat1f& feature) const {
    cv::Mat floatPrev;
    cv::Mat floatNext;
    prev.convertTo(floatPrev, CV_32F);
    next.convertTo(floatNext, CV_32F);

    feature = floatNext - floatPrev;
}

void LocalFeatureExtractor::extractFlowFeature(const cv::Mat1b& prev, const cv::Mat1b& next,
                                               std::vector<cv::Mat1f>& features) const {
    auto flow = cv::superres::createOptFlow_Farneback();

    cv::Mat1f flowX;
    cv::Mat1f flowY;
    flow->calc(prev, next, flowX, flowY);

    features.push_back(flowX);
    features.push_back(flowY);
}

void LocalFeatureExtractor::denseSampling(int scaleIndex, std::vector<cv::Vec3i>& points,
                                          std::vector<Descriptor>& descriptors) const {
    int width = width_ * scales_[scaleIndex];
    int xEnd = width - localWidth_;
    int height = height_ * scales_[scaleIndex];
    int yEnd = height - localHeight_;

    for (int y = 0; y <= yEnd; y += yStep_) {
        for (int x = 0; x <= xEnd; x += xStep_) {
            points.emplace_back(storedFeatureBeginT_ + (localDuration_ / 2), y + (localHeight_ / 2),
                                x + (localWidth_ / 2));

            Descriptor neighborhoodFeatures;
            getDescriptor(scaleIndex, cv::Vec3i(0, y, x), width, height, neighborhoodFeatures);
            descriptors.push_back(neighborhoodFeatures);
        }
    }
}

void LocalFeatureExtractor::getDescriptor(int scaleIndex, const cv::Vec3i& topLeftPoint, int width,
                                          int height, Descriptor& descriptor) const {
    int nBlockElements = xBlockSize_ * yBlockSize_ * tBlockSize_;
    int nXBlocks = localWidth_ / xBlockSize_;
    int nYBlocks = localHeight_ / yBlockSize_;
    int nTBlocks = localDuration_ / tBlockSize_;
    int nPooledElements = nXBlocks * nYBlocks * nTBlocks;
    descriptor.resize(nPooledElements * N_CHANNELS_);
    int blockIndex = 0;
    for (int tBlockIndex = 0; tBlockIndex < nTBlocks; ++tBlockIndex) {
        int tBegin = topLeftPoint(T) + tBlockSize_ * tBlockIndex;
        int tEnd = tBegin + tBlockSize_;
        for (int yBlockIndex = 0; yBlockIndex < nYBlocks; ++yBlockIndex) {
            int yBegin = topLeftPoint(Y) + yBlockSize_ * yBlockIndex;
            int yEnd = yBegin + yBlockSize_;
            for (int xBlockIndex = 0; xBlockIndex < nXBlocks; ++xBlockIndex) {
                int xBegin = topLeftPoint(X) + xBlockSize_ * xBlockIndex;
                int xEnd = xBegin + xBlockSize_;

                pooling(scaleIndex, blockIndex, nPooledElements, nBlockElements, xBegin, xEnd,
                        yBegin, yEnd, tBegin, tEnd, descriptor);
                blockIndex++;
            }
        }
    }
}

void LocalFeatureExtractor::pooling(int scaleIndex, int blockIndex, int nPooledElements,
                                    int nBlockElements, int xBegin, int xEnd, int yBegin, int yEnd,
                                    int tBegin, int tEnd, Descriptor& pooledDescriptor) const {
    for (int channelIndex = 0; channelIndex < N_CHANNELS_; ++channelIndex) {
        double sumPooling = 0.0;
        for (int t = tBegin; t < tEnd; ++t) {
            cv::Mat1f integralMat = scaleChannelIntegrals_[scaleIndex][channelIndex][t];
            sumPooling += integralMat(yEnd, xEnd) - integralMat(yBegin, xEnd) -
                          integralMat(yEnd, xBegin) + integralMat(yBegin, xBegin);
        }
        pooledDescriptor[channelIndex * nPooledElements + blockIndex] = sumPooling / nBlockElements;
    }
}

void LocalFeatureExtractor::visualizeDenseFeature(const std::vector<cv::Vec3i>& points,
                                                  const std::vector<Descriptor>& features,
                                                  int width, int height, int duration) const {
    std::vector<cv::Mat1f> video(duration);
    for (auto& frame : video) {
        frame.create(height, width);
        frame = 0.0;
    }

    int xRange = localWidth_ / 2;
    int yRange = localHeight_ / 2;
    int tRange = localDuration_ / 2;

    for (int i = 0; i < points.size(); ++i) {
        cv::Vec3i point = points[i];

        int featureIndex = 0;
        for (int t = point(T) - tRange; t <= point(T) + tRange; ++t) {
            for (int y = point(Y) - yRange; y <= point(Y) + yRange; ++y) {
                for (int x = point(X) - xRange; x <= point(X) + xRange; ++x) {
                    video[t](y, x) = features[i][featureIndex];
                    ++featureIndex;
                }
            }
        }
    }

    for (int i = 0; i < video.size(); ++i) {
        std::cout << i << std::endl;

        cv::Mat frame = video[i];
        frame = frame.reshape(0, height);
        cv::normalize(frame, frame, 1.0, 0.0, cv::NORM_MINMAX);

        cv::imshow("", frame);
        cv::waitKey(0);
    }
}

void LocalFeatureExtractor::visualizeDenseFeature(const Descriptor& feature) const {
    int index = 0;
    for (int j = 0; j < localDuration_; ++j) {
        cv::Mat1f local(localHeight_, localWidth_);
        for (int y = 0; y < localHeight_; ++y) {
            for (int x = 0; x < localWidth_; ++x) {
                local(y, x) = feature.at(index++);
            }
        }
        cv::normalize(local, local, 0.0, 1.0, cv::NORM_MINMAX);
        cv::imshow("local", local);
        cv::waitKey(0);
    }
}

void LocalFeatureExtractor::visualizePooledDenseFeature(
        const std::vector<cv::Vec3i>& points, const std::vector<Descriptor>& features) const {
    int duration = 200;
    std::vector<cv::Mat1f> video(duration);
    for (auto& frame : video) {
        frame.create(height_, width_);
        frame = 0.0;
    }

    int xRange = localWidth_ / 2;
    int yRange = localHeight_ / 2;
    int tRange = localDuration_ / 2;

    for (int i = 0; i < points.size(); ++i) {
        cv::Vec3i point = points[i];

        int featureIndex = 0;
        for (int t = point(T) - tRange; t <= point(T) + tRange; t += tBlockSize_) {
            for (int y = point(Y) - yRange; y <= point(Y) + yRange; y += yBlockSize_) {
                for (int x = point(X) - xRange; x <= point(X) + xRange; x += xBlockSize_) {
                    for (int tt = 0; tt < tBlockSize_; ++tt) {
                        for (int yy = 0; yy < xBlockSize_; ++yy) {
                            for (int xx = 0; xx < xBlockSize_; ++xx) {
                                video[t + tt](y + yy, x + xx) = features[i][featureIndex];
                            }
                        }
                    }
                    ++featureIndex;
                }
            }
        }
    }

    for (int i = 0; i < video.size(); ++i) {
        std::cout << i << std::endl;

        cv::Mat frame = video[i];
        frame = frame.reshape(0, height_);
        cv::normalize(frame, frame, 1.0, 0.0, cv::NORM_MINMAX);

        cv::imshow("", frame);
        cv::waitKey(0);
    }
}

void LocalFeatureExtractor::visualizePooledDenseFeature(const Descriptor& feature) const {
    int index = 0;
    for (int j = 0; j < localDuration_ / tBlockSize_; ++j) {
        cv::Mat1f local(localHeight_ / yBlockSize_, localWidth_ / xBlockSize_);
        for (int y = 0; y < localHeight_ / yBlockSize_; ++y) {
            for (int x = 0; x < localWidth_ / xBlockSize_; ++x) {
                local(y, x) = feature.at(index++);
            }
        }
        cv::normalize(local, local, 0.0, 1.0, cv::NORM_MINMAX);
        cv::Mat1f resizedLocal;
        cv::resize(local, resizedLocal, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);
        cv::imshow("local", resizedLocal);
        cv::waitKey(0);
    }
}
}
}