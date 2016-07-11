#include "LocalFeatureExtractor.h"

#include <zlib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/superres/optical_flow.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <array>
#include <numeric>

namespace nuisken {
namespace houghforests {

const int LocalFeatureExtractor::IO_STEP_ = 1000000;
//const int LocalFeatureExtractor::FEATURE_NUM_ = 6;
const int LocalFeatureExtractor::FEATURE_NUM_ = 1;

void LocalFeatureExtractor::save(const std::string& outputFilePath, const std::vector<float>& feature,
                              int width, int height, int duration) const {
    gzFile file = gzopen(outputFilePath.c_str(), "wb");
    std::array<std::size_t, 4> sizes = {feature.size(), width, height, duration};
    gzwrite(file, sizes.data(), sizes.size() * sizeof(std::size_t));

    for (int i = 0; i < feature.size(); i += IO_STEP_) {
        int writeSize = 0;
        if ((i + IO_STEP_) < feature.size()) {
            writeSize = IO_STEP_;
        } else {
            writeSize = feature.size() - i;
        }
        gzwrite(file, feature.data() + i, writeSize * sizeof(float));
    }
    gzclose(file);
}

void LocalFeatureExtractor::saveHeader(const std::string& outputFilePath, 
                                    const std::string& tmpDataFilePath, 
                                    int dims, int width, int height, int duration) const {
    gzFile tmpFile = gzopen(tmpDataFilePath.c_str(), "rb");
    gzFile file = gzopen(outputFilePath.c_str(), "wb");
    std::array<std::size_t, 4> sizes = {dims, width, height, duration};

    gzwrite(file, sizes.data(), sizes.size() * sizeof(std::size_t));
    
    for (int i = 0; i < dims; i += IO_STEP_) {
        int dataSize = 0;
        if ((i + IO_STEP_) < dims) {
            dataSize = IO_STEP_;
        } else {
            dataSize = dims - i;
        }

        std::vector<float> tmpData(dataSize);
        gzread(tmpFile, tmpData.data(), dataSize * sizeof(float));
        gzwrite(file, tmpData.data(), dataSize * sizeof(float));
    }
    
    gzclose(file);
    gzclose(tmpFile);
}

void LocalFeatureExtractor::saveAppend(const std::string& outputFilePath,
                                    const std::vector<float>& feature) const {
    gzFile file = gzopen(outputFilePath.c_str(), "ab");
    for (int i = 0; i < feature.size(); i += IO_STEP_) {
        int writeSize = 0;
        if ((i + IO_STEP_) < feature.size()) {
            writeSize = IO_STEP_;
        } else {
            writeSize = feature.size() - i;
        }
        gzwrite(file, feature.data() + i, writeSize * sizeof(float));
    }
    gzclose(file);
}

void LocalFeatureExtractor::saveDenseFeatures(const std::string& outputFilePath, 
                                           const std::vector<cv::Vec3i>& points,
                                           const std::vector<std::vector<float>>& features,
                                           int width, int height, int duration) const {
    gzFile file = gzopen(outputFilePath.c_str(), "wb");
    std::array<std::size_t, 3> sizes = {width, height, duration};
    gzwrite(file, sizes.data(), sizes.size() * sizeof(std::size_t));

    std::array<std::size_t, 2> dataInfo = {features.size(), features.front().size()};
    gzwrite(file, dataInfo.data(), dataInfo.size() * sizeof(std::size_t));

    for (int i = 0; i < points.size(); ++i) {
        gzwrite(file, points.at(i).val, points.at(i).rows * sizeof(int));
        gzwrite(file, features.at(i).data(), features.at(i).size() * sizeof(float));
    }
    gzclose(file);
}

void LocalFeatureExtractor::saveAppendDenseFeatures(const std::string& outputFilePath,
                                                 const std::vector<cv::Vec3i>& points,
                                                 const std::vector<std::vector<float>>& features) const {
    gzFile file = gzopen(outputFilePath.c_str(), "ab");
    for (int i = 0; i < points.size(); ++i) {
        gzwrite(file, points.at(i).val, points.at(i).rows * sizeof(int));
        gzwrite(file, features.at(i).data(), features.at(i).size() * sizeof(float));
    }
    gzclose(file);
}

void LocalFeatureExtractor::saveHeaderDenseFeatures(const std::string& outputFilePath,
                                                 const std::string& tmpDataFilePath,
                                                 int width, int height, int duration, 
                                                 int featureNums, int featureDims) const {
    gzFile tmpFile = gzopen(tmpDataFilePath.c_str(), "rb");
    gzFile file = gzopen(outputFilePath.c_str(), "wb");
    std::array<std::size_t, 5> dataInfo = {width, height, duration, featureNums, featureDims};

    gzwrite(file, dataInfo.data(), dataInfo.size() * sizeof(std::size_t));

    for (int i = 0; i < featureNums; ++i) {
        cv::Vec3i tmpPoint;
        std::vector<float> tmpData(featureDims);
        gzread(tmpFile, tmpPoint.val, tmpPoint.rows * sizeof(int));
        gzread(tmpFile, tmpData.data(), featureDims * sizeof(float));
        gzwrite(file, tmpPoint.val, tmpPoint.rows * sizeof(int));
        gzwrite(file, tmpData.data(), featureDims * sizeof(float));
    }

    gzclose(file);
    gzclose(tmpFile);
}

void LocalFeatureExtractor::load(const std::string& inputFilePath, std::vector<float>& feature,
                              int& width, int& height, int& duration) const {
    gzFile file = gzopen(inputFilePath.c_str(), "rb");
    std::array<std::size_t, 4> sizes;
    gzread(file, sizes.data(), sizes.size() * sizeof(std::size_t));

    width = sizes.at(1);
    height = sizes.at(2);
    duration = sizes.at(3);

    int dataDims = sizes.at(0);
    feature = std::vector<float>(dataDims);

    for (int i = 0; i < feature.size(); i += IO_STEP_) {
        int readSize = 0;
        if ((i + IO_STEP_) < feature.size()) {
            readSize = IO_STEP_;
        } else {
            readSize = feature.size() - i;
        }
        gzread(file, feature.data() + i, readSize * sizeof(float));
    }
    gzclose(file);
}

void LocalFeatureExtractor::loadDenseFeatures(const std::string& inputFilePath, 
                                           std::vector<cv::Vec3i>& points,
                                           std::vector<std::vector<float>>& features,
                                           int& width, int& height, int& duration) const {
    gzFile file = gzopen(inputFilePath.c_str(), "rb");
    std::array<std::size_t, 3> sizes;
    gzread(file, sizes.data(), sizes.size() * sizeof(std::size_t));
    width = sizes.at(0);
    height = sizes.at(1);
    duration = sizes.at(2);

    std::array<std::size_t, 2> dataInfo;
    gzread(file, dataInfo.data(), dataInfo.size() * sizeof(std::size_t));

    int featureNums = dataInfo.at(0);
    int featureDims = dataInfo.at(1);

    points.resize(featureNums);
    features.resize(featureNums);
    for (int i = 0; i < featureNums; ++i) {
        gzread(file, points.at(i).val, points.at(i).rows * sizeof(int));

        features.at(i).resize(featureDims);
        gzread(file, features.at(i).data(), featureDims * sizeof(float));
    }
    gzclose(file);
}

std::vector<std::vector<float>> LocalFeatureExtractor::extractFeature(const std::vector<cv::Mat1b>& video,
                                                                   const FeatureType featureType,
                                                                   int startFrame, int endFrame) const {
    if (endFrame == -1) {
        endFrame = video.size();
    }
    std::vector<cv::Mat1b> targetVideo;
    targetVideo.reserve(endFrame - startFrame);
    for (int t = startFrame; t < endFrame; ++t) {
        targetVideo.push_back(video.at(t));
    }

    switch (featureType) {
    case INTENSITY:
        return std::vector<std::vector<float>>{extractIntensityFeature(targetVideo)};
    case X_DERIVATIVE:
        return std::vector<std::vector<float>>{extractXDerivativeFeature(targetVideo)};
    case Y_DERIVATIVE:
        return std::vector<std::vector<float>>{extractYDerivativeFeature(targetVideo)};
    case T_DERIVATIVE:
        return std::vector<std::vector<float>>{extractTDerivativeFeature(targetVideo)};
    case FLOW:
        return extractFlowFeature(targetVideo);
    default:
        return std::vector<std::vector<float>>();
    }
}

std::vector<std::vector<float>> LocalFeatureExtractor::extractAllFeatures(const std::vector<cv::Mat1b>& video,
                                                                       int startFrame, int endFrame) const {
    if (endFrame == -1) {
        endFrame = video.size();
    }
    std::vector<cv::Mat1b> targetVideo;
    targetVideo.reserve(endFrame - startFrame);
    for (int t = startFrame; t < endFrame; ++t) {
        targetVideo.push_back(video.at(t));
    }

    auto intensity = extractIntensityFeature(targetVideo);
    //auto xDerivative = extractXDerivativeFeature(targetVideo);
    //auto yDerivative = extractYDerivativeFeature(targetVideo);
    //auto tDerivative = extractTDerivativeFeature(targetVideo);
    //auto flow = extractFlowFeature(targetVideo);

    //return {intensity, xDerivative, yDerivative, tDerivative, flow.at(0), flow.at(1)};
    return {intensity};
}

void LocalFeatureExtractor::denseSampling(const std::vector<float>& features,
                                       std::vector<cv::Vec3i>& points,
                                       std::vector<std::vector<float>>& descriptors,
                                       const std::vector<int>& neighborhoodSizes,
                                       const std::vector<int>& samplingStrides,
                                       int width, int height, int duration) const {
    for (int t = 0; t <= (duration - neighborhoodSizes.at(T)); t += samplingStrides.at(T)) {
        for (int y = 0; y <= (height - neighborhoodSizes.at(Y)); y += samplingStrides.at(Y)) {
            for (int x = 0; x <= (width - neighborhoodSizes.at(X)); x += samplingStrides.at(X)) {
                cv::Vec3i centerPoint(t + (neighborhoodSizes.at(T) / 2),
                                      y + (neighborhoodSizes.at(Y) / 2),
                                      x + (neighborhoodSizes.at(X) / 2));
                points.push_back(centerPoint);

                std::vector<float> neighborhoodFeatures = getNeighborhoodFeatures(features, cv::Vec3i(t, y, x),
                                                                                  neighborhoodSizes, width, height);
                descriptors.push_back(neighborhoodFeatures);
            }
        }
    }
}

std::vector<float> LocalFeatureExtractor::getNeighborhoodFeatures(const std::vector<float>& features,
                                                               const cv::Vec3i& topLeftPoint,
                                                               const std::vector<int>& neighborhoodSizes,
                                                               int width, int height) const {
    std::vector<float> neighborhoodFeatures;
    int neighborhoodSize = std::accumulate(std::begin(neighborhoodSizes), 
                                           std::end(neighborhoodSizes), 
                                           1, std::multiplies<int>());
    neighborhoodFeatures.reserve(neighborhoodSize);
    for (int t = 0; t < neighborhoodSizes.at(T); ++t) {
        for (int y = 0; y < neighborhoodSizes.at(Y); ++y) {
            for (int x = 0; x < neighborhoodSizes.at(X); ++x) {
                int featureIndex = calculateFeatureIndex(x + topLeftPoint(X), 
                                                         y + topLeftPoint(Y), 
                                                         t + topLeftPoint(T),
                                                         width, height);
                neighborhoodFeatures.push_back(features.at(featureIndex));
            }
        }
    }

    return neighborhoodFeatures;
}

int LocalFeatureExtractor::calculateFeatureIndex(int x, int y, int t, 
                                              int width, int height) const {
    return (t * width * height) + (y * width) + x;
}

std::vector<float> LocalFeatureExtractor::extractIntensityFeature(const cv::Mat1b& frame) const {
    cv::Mat feature = frame.reshape(0, 1);
    feature.convertTo(feature, CV_32F);

    return feature;
}

std::vector<float> LocalFeatureExtractor::extractXDerivativeFeature(const cv::Mat1b& frame) const {
    cv::Mat dst;
    cv::Sobel(frame, dst, CV_32F, 1, 0);

    cv::Mat feature = dst.reshape(0, 1);
    return feature;
}

std::vector<float> LocalFeatureExtractor::extractYDerivativeFeature(const cv::Mat1b& frame) const {
    cv::Mat dst;
    cv::Sobel(frame, dst, CV_32F, 0, 1);

    cv::Mat feature = dst.reshape(0, 1);
    return feature;
}

std::vector<float> LocalFeatureExtractor::extractTDerivativeFeature(const cv::Mat1b& prev, 
                                                                 const cv::Mat1b& next) const {
    cv::Mat floatPrev;
    cv::Mat floatNext;
    prev.convertTo(floatPrev, CV_32F);
    next.convertTo(floatNext, CV_32F);

    cv::Mat diff = floatNext - floatPrev;

    cv::Mat feature = diff.reshape(0, 1);
    return feature;
}

std::vector<std::vector<float>> LocalFeatureExtractor::extractFlowFeature(const cv::Mat1b& prev,
                                                                       const cv::Mat1b& next) const {
    auto flow = cv::superres::createOptFlow_Farneback();

    cv::Mat1f flowX;
    cv::Mat1f flowY;
    flow->calc(prev, next, flowX, flowY);

    flowX = flowX.reshape(0, 1);
    flowY = flowY.reshape(0, 1);
    return std::vector<std::vector<float>>{flowX, flowY};
}

std::vector<float> LocalFeatureExtractor::extractIntensityFeature(const std::vector<cv::Mat1b>& video) const {
    std::vector<float> feature;
    for (const auto& frame : video) {
        std::vector<float> tmpFeature = extractIntensityFeature(frame);
        std::copy(std::begin(tmpFeature), std::end(tmpFeature), 
                  std::back_inserter(feature));
    }
    return feature;
}

std::vector<float> LocalFeatureExtractor::extractXDerivativeFeature(const std::vector<cv::Mat1b>& video) const {
    std::vector<float> feature;
    for (const auto& frame : video) {
        std::vector<float> tmpFeature = extractXDerivativeFeature(frame);
        std::copy(std::begin(tmpFeature), std::end(tmpFeature),
                  std::back_inserter(feature));
    }
    return feature;
}

std::vector<float> LocalFeatureExtractor::extractYDerivativeFeature(const std::vector<cv::Mat1b>& video) const {
    std::vector<float> feature;
    for (const auto& frame : video) {
        std::vector<float> tmpFeature = extractYDerivativeFeature(frame);
        std::copy(std::begin(tmpFeature), std::end(tmpFeature),
                  std::back_inserter(feature));
    }
    return feature;
}

std::vector<float> LocalFeatureExtractor::extractTDerivativeFeature(const std::vector<cv::Mat1b>& video) const {
    std::vector<float> feature;
    cv::Mat prev = video[0];
    std::vector<float> zeros(video[0].total());
    std::copy(std::begin(zeros), std::end(zeros),
              std::back_inserter(feature));
    for (int i = 1; i < video.size(); ++i) {
        cv::Mat next = video[i];

        std::vector<float> tmpFeature = extractTDerivativeFeature(prev, next);
        std::copy(std::begin(tmpFeature), std::end(tmpFeature),
                  std::back_inserter(feature));

        prev = next;
    }

    return feature;
}

std::vector<std::vector<float>> LocalFeatureExtractor::extractFlowFeature(const std::vector<cv::Mat1b>& video) const {
    std::vector<float> feature1;
    std::vector<float> feature2;
    cv::Mat prev = video[0];
    std::vector<float> zeros(video[0].total());
    std::copy(std::begin(zeros), std::end(zeros),
              std::back_inserter(feature1));
    std::copy(std::begin(zeros), std::end(zeros),
              std::back_inserter(feature2));
    for (int i = 1; i < video.size(); ++i) {
        cv::Mat next = video[i];
        std::vector<std::vector<float>> tmpFeature = extractFlowFeature(prev, next);
        std::copy(std::begin(tmpFeature.at(0)), std::end(tmpFeature.at(0)),
                  std::back_inserter(feature1));
        std::copy(std::begin(tmpFeature.at(1)), std::end(tmpFeature.at(1)),
                  std::back_inserter(feature2));

        prev = next;
    }

    return std::vector<std::vector<float>>{feature1, feature2};
}

void LocalFeatureExtractor::visualizeFeature(const std::vector<float>& feature,
                                          int width, int height) const {
    int frameSize = width * height;
    for (int i = 0; i < feature.size(); i += frameSize) {
        auto frameBegin = std::begin(feature);
        std::advance(frameBegin, i);
        auto frameEnd = frameBegin;
        std::advance(frameEnd, frameSize);
        std::vector<float> frameFeature(frameBegin, frameEnd);

        cv::Mat frame(frameFeature);
        frame = frame.reshape(0, height);
        cv::normalize(frame, frame, 1.0, 0.0, cv::NORM_MINMAX);

        cv::imshow("", frame);
        cv::waitKey(10);
    }
}

void LocalFeatureExtractor::visualizeDenseFeature(const std::vector<cv::Vec3i>& points,
                                               const std::vector<std::vector<float>>& features,
                                               int width, int height, int duration,
                                               const std::vector<int>& neighborhoodSizes) const {
    std::vector<cv::Mat1f> video(duration);
    for (auto& frame : video) {
        frame.create(height, width);
        frame = 0.0;
    }

    int xRange = neighborhoodSizes.at(X) / 2;
    int yRange = neighborhoodSizes.at(Y) / 2;
    int tRange = neighborhoodSizes.at(T) / 2;

    for (int i = 0; i < points.size(); ++i) {
        cv::Vec3i point = points.at(i);
        
        int featureIndex = 0;
        for (int t = point(T) - tRange; t <= point(T) + tRange; ++t) {
            for (int y = point(Y) - yRange; y <= point(Y) + yRange; ++y) {
                for (int x = point(X) - xRange; x <= point(X) + xRange; ++x) {
                    video.at(t)(y, x) = features.at(i).at(featureIndex);
                    ++featureIndex;
                }
            }
        }
    }

    for (int i = 0; i < video.size(); ++i) {
        cv::Mat frame = video.at(i);
        frame = frame.reshape(0, height);
        cv::normalize(frame, frame, 1.0, 0.0, cv::NORM_MINMAX);

        cv::imshow("", frame);
        cv::waitKey(10);
    }
}

}
}