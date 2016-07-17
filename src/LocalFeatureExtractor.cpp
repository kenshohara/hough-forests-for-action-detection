#include "LocalFeatureExtractor.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/superres/optical_flow.hpp>

#include <array>
#include <iostream>
#include <numeric>

namespace nuisken {
namespace houghforests {

const int LocalFeatureExtractor::N_CHANNELS_ = 4;
const int LocalFeatureExtractor::N_HOG_BINS_ = 9;

void LocalFeatureExtractor::makeLocalSizeOdd(int& size) const {
    if ((size % 2) == 0) {
        size++;
    }
}

void LocalFeatureExtractor::extractLocalFeatures(
        std::vector<std::vector<cv::Vec3i>>& scalePoints,
        std::vector<std::vector<Descriptor>>& scaleDescriptors) {
    extractLocalFeatures(scalePoints, scaleDescriptors, ColorVideo{});
}

void LocalFeatureExtractor::extractLocalFeatures(
        std::vector<std::vector<cv::Vec3i>>& scalePoints,
        std::vector<std::vector<Descriptor>>& scaleDescriptors, ColorVideo& usedVideo) {
    readOriginalScaleVideo();
    generateScaledVideos();
    for (int scaleIndex = 0; scaleIndex < scales_.size(); ++scaleIndex) {
        int beginFrame = 1;
        int endFrame = scaleVideos_[scaleIndex].size();
        extractFeatures(scaleIndex, beginFrame, endFrame);
        if (scaleIndex == 0) {
            nStoredFeatureFrames_ += endFrame - beginFrame;
        }
        if (nStoredFeatureFrames_ < localDuration_) {
            isEnd_ = true;
            return;
        }

        std::vector<cv::Vec3i> points;
        std::vector<std::vector<float>> descriptors;
        denseSampling(scaleIndex, points, descriptors);
        scalePoints.push_back(points);
        scaleDescriptors.push_back(descriptors);
    }

    for (const auto& frame : colorVideo_) {
        usedVideo.push_back(frame.clone());
    }

    deleteOldData();
}

void LocalFeatureExtractor::readOriginalScaleVideo() {
    int nFrames = tStep_;
    if (scaleVideos_.front().empty()) {
        cv::Mat firstFrame;
        videoCapture_ >> firstFrame;
        colorVideo_.push_back(firstFrame.clone());
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
            isEnd_ = true;
            break;
        }
        colorVideo_.push_back(frame.clone());
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        scaleVideos_.front().push_back(frame);
    }
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
    extractIntensityFeature(scaleChannelFeatures_[scaleIndex][0], scaleIndex, beginFrame, endFrame);
    extractXDerivativeFeature(scaleChannelFeatures_[scaleIndex][1], scaleIndex, beginFrame,
                              endFrame);
    extractYDerivativeFeature(scaleChannelFeatures_[scaleIndex][2], scaleIndex, beginFrame,
                              endFrame);
    extractTDerivativeFeature(scaleChannelFeatures_[scaleIndex][3], scaleIndex, beginFrame,
                              endFrame);
    // extractFlowFeature(scaleChannelFeatures_[scaleIndex][4],
    // scaleChannelFeatures_[scaleIndex][5],
    //                   scaleIndex, beginFrame, endFrame);
    // extractHOGFeature(scaleChannelFeatures_[scaleIndex], scaleIndex, beginFrame, endFrame);
}

void LocalFeatureExtractor::deleteOldData() {
    if (localDuration_ <= tStep_) {
        scaleVideos_ = std::vector<Video>(scales_.size());
        scaleChannelFeatures_ =
                std::vector<MultiChannelFeature>(scales_.size(), MultiChannelFeature(N_CHANNELS_));
        storedColorVideoStartT_ += tStep_;
        storedFeatureStartT_ += tStep_;
        nStoredFeatureFrames_ = 0;

        return;
    }

    storedColorVideoStartT_ += colorVideo_.size();
    colorVideo_.clear();
    for (auto& video : scaleVideos_) {
        video = {video.back()};
    }

    for (int scaleIndex = 0; scaleIndex < scales_.size(); ++scaleIndex) {
        int width = width_ * scales_[scaleIndex];
        int height = height_ * scales_[scaleIndex];

        int featureIndex = calculateFeatureIndex(0, 0, tStep_, width, height);
        for (auto& features : scaleChannelFeatures_[scaleIndex]) {
            auto beginIt = std::begin(features);
            auto deleteEndIt = beginIt + featureIndex;
            features.erase(beginIt, deleteEndIt);
        }
    }
    storedFeatureStartT_ += tStep_;
    nStoredFeatureFrames_ -= tStep_;
}

void LocalFeatureExtractor::extractIntensityFeature(Feature& features, int scaleIndex,
                                                    int beginFrame, int endFrame) {
    for (int t = beginFrame; t < endFrame; ++t) {
        Feature oneFrameFeature = extractIntensityFeature(scaleVideos_[scaleIndex][t]);
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));
    }
}

void LocalFeatureExtractor::extractXDerivativeFeature(Feature& features, int scaleIndex,
                                                      int beginFrame, int endFrame) {
    for (int t = beginFrame; t < endFrame; ++t) {
        Feature oneFrameFeature = extractXDerivativeFeature(scaleVideos_[scaleIndex][t]);
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));
    }
}

void LocalFeatureExtractor::extractYDerivativeFeature(Feature& features, int scaleIndex,
                                                      int beginFrame, int endFrame) {
    for (int t = beginFrame; t < endFrame; ++t) {
        Feature oneFrameFeature = extractYDerivativeFeature(scaleVideos_[scaleIndex][t]);
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));
    }
}

void LocalFeatureExtractor::extractTDerivativeFeature(Feature& features, int scaleIndex,
                                                      int beginFrame, int endFrame) {
    cv::Mat prev = scaleVideos_[scaleIndex][beginFrame - 1];
    for (int t = beginFrame; t < endFrame; ++t) {
        cv::Mat next = scaleVideos_[scaleIndex][t];

        Feature oneFrameFeature = extractTDerivativeFeature(prev, next);
        std::copy(std::begin(oneFrameFeature), std::end(oneFrameFeature),
                  std::back_inserter(features));

        prev = next;
    }
}

void LocalFeatureExtractor::extractFlowFeature(Feature& xFeatures, Feature& yFeatures,
                                               int scaleIndex, int beginFrame, int endFrame) {
    cv::Mat prev = scaleVideos_[scaleIndex][beginFrame - 1];
    for (int t = beginFrame; t < endFrame; ++t) {
        cv::Mat next = scaleVideos_[scaleIndex][t];
        std::vector<Feature> feature = extractFlowFeature(prev, next);
        std::copy(std::begin(feature[0]), std::end(feature[0]), std::back_inserter(xFeatures));
        std::copy(std::begin(feature[1]), std::end(feature[1]), std::back_inserter(yFeatures));

        prev = next;
    }
}

void LocalFeatureExtractor::extractHOGFeature(std::vector<Feature>& features, int scaleIndex,
                                              int beginFrame, int endFrame) {
    for (int t = beginFrame; t < endFrame; ++t) {
        std::vector<Feature> oneFrameFeatures = extractHOGFeature(scaleVideos_[scaleIndex][t]);
        for (int i = 0; i < oneFrameFeatures.size(); ++i) {
            std::copy(std::begin(oneFrameFeatures.at(i)), std::end(oneFrameFeatures.at(i)),
                      std::back_inserter(features.at(i)));
        }
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

std::vector<LocalFeatureExtractor::Feature> LocalFeatureExtractor::extractHOGFeature(
        const cv::Mat1b& frame) const {
    std::vector<cv::Mat> channels(N_HOG_BINS_);
    for (auto& channel : channels) {
        channel.create(frame.rows, frame.cols, CV_8U);
    }
    cv::Mat I_x;
    cv::Mat I_y;

    // |I_x|, |I_y|
    Sobel(frame, I_x, CV_16S, 1, 0, 3);
    Sobel(frame, I_y, CV_16S, 0, 1, 3);

    cv::Mat orientations(frame.rows, frame.cols, CV_8U);
    cv::Mat magnitudes(frame.rows, frame.cols, CV_8U);

    int rows = I_x.rows;
    int cols = I_y.cols;

    if (I_x.isContinuous() && I_y.isContinuous() && orientations.isContinuous() &&
        magnitudes.isContinuous()) {
        cols *= rows;
        rows = 1;
    }

    for (int y = 0; y < rows; y++) {
        short* ptr_Ix = I_x.ptr<short>(y);
        short* ptr_Iy = I_y.ptr<short>(y);
        uchar* ptr_out = orientations.ptr<uchar>(y);
        for (int x = 0; x < cols; x++) {
            // Avoid division by zero
            float tx = (float)ptr_Ix[x] + (float)copysign(0.000001f, (float)ptr_Ix[x]);
            // Scaling [-pi/2 pi/2] -> [0 80*pi]
            ptr_out[x] = cv::saturate_cast<uchar>(
                    (atan((float)ptr_Iy[x] / tx) + 3.14159265f / 2.0f) * 80);
        }
    }

    // Magnitude of gradients
    for (int y = 0; y < rows; y++) {
        short* ptr_Ix = I_x.ptr<short>(y);
        short* ptr_Iy = I_y.ptr<short>(y);
        uchar* ptr_out = magnitudes.ptr<uchar>(y);
        for (int x = 0; x < cols; x++) {
            ptr_out[x] = cv::saturate_cast<uchar>(sqrt((float)ptr_Ix[x] * (float)ptr_Ix[x] +
                                                       (float)ptr_Iy[x] * (float)ptr_Iy[x]));
        }
    }

    // 9-bin HOG feature stored at vImg[7] - vImg[15]
    hog_.extractOBin(orientations, magnitudes, channels, 0);

    std::vector<Feature> hogFeatures(channels.size());
    for (int i = 0; i < hogFeatures.size(); ++i) {
        channels.at(i) = channels.at(i).reshape(0, 1);
        channels[i].copyTo(hogFeatures.at(i));
    }
    return hogFeatures;
}

void LocalFeatureExtractor::denseSampling(int scaleIndex, std::vector<cv::Vec3i>& points,
                                          std::vector<Descriptor>& descriptors) const {
    int width = width_ * scales_[scaleIndex];
    int xEnd = width - localWidth_;
    int height = height_ * scales_[scaleIndex];
    int yEnd = height - localHeight_;

    for (int y = 0; y <= yEnd; y += yStep_) {
        for (int x = 0; x <= xEnd; x += xStep_) {
            points.emplace_back(storedFeatureStartT_ + (localDuration_ / 2), y + (localHeight_ / 2),
                                x + (localWidth_ / 2));

            Descriptor neighborhoodFeatures =
                    getDescriptor(scaleIndex, cv::Vec3i(0, y, x), width, height);
            descriptors.push_back(neighborhoodFeatures);
        }
    }
}

LocalFeatureExtractor::Descriptor LocalFeatureExtractor::getDescriptor(
        int scaleIndex, const cv::Vec3i& topLeftPoint, int width, int height) const {
    Descriptor descriptor;
    int nLocalElements = localWidth_ * localHeight_ * localDuration_;
    int nPooledElements = xBlockSize_ * yBlockSize_ * tBlockSize_;
    descriptor.reserve(nPooledElements * N_CHANNELS_);
    for (int channelIndex = 0; channelIndex < N_CHANNELS_; ++channelIndex) {
        Descriptor channelDescriptor;
        channelDescriptor.reserve(nLocalElements);
        for (int t = 0; t < localDuration_; ++t) {
            for (int y = 0; y < localHeight_; ++y) {
                for (int x = 0; x < localWidth_; ++x) {
                    int featureIndex =
                            calculateFeatureIndex(x + topLeftPoint(X), y + topLeftPoint(Y),
                                                  t + topLeftPoint(T), width, height);
                    channelDescriptor.push_back(
                            scaleChannelFeatures_[scaleIndex][channelIndex][featureIndex]);
                }
            }
        }
        Descriptor averagePooledChannelDescriptor = pooling(channelDescriptor, AVERAGE);
        Descriptor maxPooledChannelDescriptor = pooling(channelDescriptor, MAX);
        std::copy(std::begin(averagePooledChannelDescriptor),
                  std::end(averagePooledChannelDescriptor), std::back_inserter(descriptor));
        std::copy(std::begin(maxPooledChannelDescriptor), std::end(maxPooledChannelDescriptor),
                  std::back_inserter(descriptor));
    }

    return descriptor;
}

void LocalFeatureExtractor::denseSamplingHOG(int scaleIndex, std::vector<cv::Vec3i>& points,
                                             std::vector<Descriptor>& descriptors) const {
    int width = width_ * scales_[scaleIndex];
    int xEnd = width - localWidth_;
    int height = height_ * scales_[scaleIndex];
    int yEnd = height - localHeight_;

    for (int y = 0; y <= yEnd; y += yStep_) {
        for (int x = 0; x <= xEnd; x += xStep_) {
            points.emplace_back(storedFeatureStartT_ + (localDuration_ / 2), y + (localHeight_ / 2),
                                x + (localWidth_ / 2));

            Descriptor neighborhoodFeatures =
                    getHOGDescriptor(scaleIndex, cv::Vec3i(0, y, x), width, height);
            descriptors.push_back(neighborhoodFeatures);
        }
    }
}

LocalFeatureExtractor::Descriptor LocalFeatureExtractor::getHOGDescriptor(
        int scaleIndex, const cv::Vec3i& topLeftPoint, int width, int height) const {
    std::vector<Descriptor> channelDescriptors(N_HOG_BINS_);
    int nLocalElements = localWidth_ * localHeight_ * localDuration_;
    int nPooledElements = xBlockSize_ * yBlockSize_ * tBlockSize_;
    channelDescriptors.reserve(nPooledElements * N_HOG_BINS_);
    for (int channelIndex = 0; channelIndex < N_HOG_BINS_; ++channelIndex) {
        channelDescriptors.at(channelIndex).reserve(nLocalElements);
        for (int t = 0; t < localDuration_; ++t) {
            for (int y = 0; y < localHeight_; ++y) {
                for (int x = 0; x < localWidth_; ++x) {
                    int featureIndex =
                            calculateFeatureIndex(x + topLeftPoint(X), y + topLeftPoint(Y),
                                                  t + topLeftPoint(T), width, height);
                    channelDescriptors.at(channelIndex)
                            .push_back(
                                    scaleChannelFeatures_[scaleIndex][channelIndex][featureIndex]);
                }
            }
        }
    }
    Descriptor hogDescriptor = calculateHistogram(channelDescriptors);
    return hogDescriptor;
}

int LocalFeatureExtractor::calculateFeatureIndex(int x, int y, int t, int width, int height) const {
    return (t * width * height) + (y * width) + x;
}

LocalFeatureExtractor::Descriptor LocalFeatureExtractor::calculateHistogram(
        const std::vector<Descriptor>& binValues) const {
    Descriptor hogDescriptor;
    int xSize = localWidth_ / xBlockSize_;
    int ySize = localHeight_ / yBlockSize_;
    int tSize = localDuration_ / tBlockSize_;
    hogDescriptor.reserve(xSize * ySize * tSize * N_HOG_BINS_);
    for (int tBlockIndex = 0; tBlockIndex < tSize; ++tBlockIndex) {
        for (int yBlockIndex = 0; yBlockIndex < ySize; ++yBlockIndex) {
            for (int xBlockIndex = 0; xBlockIndex < xSize; ++xBlockIndex) {
                int beginX = xBlockIndex * xBlockSize_;
                int beginY = yBlockIndex * yBlockSize_;
                int beginT = tBlockIndex * tBlockSize_;
                Descriptor blockHistogram =
                        calculateBlockHistogram(binValues, beginX, beginY, beginT);
                std::copy(std::begin(blockHistogram), std::end(blockHistogram),
                          std::back_inserter(hogDescriptor));
            }
        }
    }
    return hogDescriptor;
}

LocalFeatureExtractor::Descriptor LocalFeatureExtractor::calculateBlockHistogram(
        const std::vector<Descriptor>& binValues, int beginX, int beginY, int beginT) const {
    Descriptor blockHistogram(N_HOG_BINS_, 0.f);
    for (int t = 0; t < tBlockSize_; ++t) {
        for (int y = 0; y < yBlockSize_; ++y) {
            int featureIndex = calculateFeatureIndex(beginX, beginY + y, beginT + t, localWidth_,
                                                     localHeight_);
            for (int x = 0; x < xBlockSize_; ++x, ++featureIndex) {
                for (int binIndex = 0; binIndex < N_HOG_BINS_; ++binIndex) {
                    blockHistogram[binIndex] += binValues[binIndex][featureIndex];
                }
            }
        }
    }
    cv::normalize(blockHistogram, blockHistogram, 1.0, 0.0, cv::NORM_L2);
    return blockHistogram;
}

LocalFeatureExtractor::Descriptor LocalFeatureExtractor::pooling(const Descriptor& descriptor,
                                                                 PoolingType type) const {
    Descriptor pooledDescriptor;
    int xSize = localWidth_ / xBlockSize_;
    int ySize = localHeight_ / yBlockSize_;
    int tSize = localDuration_ / tBlockSize_;
    pooledDescriptor.reserve(xSize * ySize * tSize);
    for (int tBlockIndex = 0; tBlockIndex < tSize; ++tBlockIndex) {
        for (int yBlockIndex = 0; yBlockIndex < ySize; ++yBlockIndex) {
            for (int xBlockIndex = 0; xBlockIndex < xSize; ++xBlockIndex) {
                int beginX = xBlockIndex * xBlockSize_;
                int beginY = yBlockIndex * yBlockSize_;
                int beginT = tBlockIndex * tBlockSize_;
                if (type == AVERAGE) {
                    pooledDescriptor.push_back(averagePooling(descriptor, beginX, beginY, beginT));
                } else {
                    pooledDescriptor.push_back(maxPooling(descriptor, beginX, beginY, beginT));
                }
            }
        }
    }
    return pooledDescriptor;
}

float LocalFeatureExtractor::averagePooling(const Descriptor& descriptor, int beginX, int beginY,
                                            int beginT) const {
    float sum = 0.0;
    for (int t = 0; t < tBlockSize_; ++t) {
        for (int y = 0; y < yBlockSize_; ++y) {
            int featureIndex = calculateFeatureIndex(beginX, beginY + y, beginT + t, localWidth_,
                                                     localHeight_);
            for (int x = 0; x < xBlockSize_; ++x) {
                sum += descriptor[featureIndex++];
            }
        }
    }
    return sum / (xBlockSize_ * yBlockSize_ * tBlockSize_);
}

float LocalFeatureExtractor::maxPooling(const Descriptor& descriptor, int beginX, int beginY,
                                        int beginT) const {
    float max = 0.0;
    for (int t = 0; t < tBlockSize_; ++t) {
        for (int y = 0; y < yBlockSize_; ++y) {
            int featureIndex = calculateFeatureIndex(beginX, beginY + y, beginT + t, localWidth_,
                                                     localHeight_);
            for (int x = 0; x < xBlockSize_; ++x, ++featureIndex) {
                if (descriptor[featureIndex] > max) {
                    max = descriptor[featureIndex];
                }
            }
        }
    }
    return max;
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

    // for (int i = 0; i < points.size(); ++i) {
    //    int index = 0;
    //    for (int j = 0; j < localDuration_; ++j) {
    //        std::cout << localHeight_ * localWidth_ * localDuration_ << ", " <<
    //        features.at(i).size() << std::endl;
    //        cv::Mat1f local(localHeight_, localWidth_);
    //        for (int y = 0; y < localHeight_; ++y) {
    //            for (int x = 0; x < localWidth_; ++x) {
    //                local(y, x) = features.at(i).at(index++);
    //            }
    //        }
    //        cv::normalize(local, local, 0.0, 1.0, cv::NORM_MINMAX);
    //        cv::imshow("local", local);
    //        cv::waitKey(0);
    //    }
    //}

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
}
}