#pragma once

#include "opencv2/core/core.hpp"

#include <vector>

namespace nuisken {
namespace houghforests {

class LocalFeatureExtractor {
public:
    enum AXIS {
        X = 2,
        Y = 1,
        T = 0
    };

    enum FeatureType {
        INTENSITY,
        X_DERIVATIVE,
        Y_DERIVATIVE,
        T_DERIVATIVE,
        FLOW
    };
    static const int FEATURE_NUM_;

private:
    static const int IO_STEP_;

public:
    std::vector<std::vector<float>> extractFeature(const std::vector<cv::Mat1b>& video, 
                                                   const FeatureType featureType,
                                                   int startFrame = 0, int endFrame = -1) const;
    std::vector<std::vector<float>> extractAllFeatures(const std::vector<cv::Mat1b>& video,
                                                       int startFrame = 0, int endFrame = -1) const;

    void denseSampling(const std::vector<float>& features,
                       std::vector<cv::Vec3i>& points, 
                       std::vector<std::vector<float>>& descriptors,
                       const std::vector<int>& neighborhoodSizes,
                       const std::vector<int>& samplingStrides,
                       int width, int height, int duration) const;

    void save(const std::string& outputFilePath, const std::vector<float>& feature,
              int width, int height, int duration) const;
    void saveHeader(const std::string& outputFilePath, const std::string& tmpDataFilePath, 
                    int dims, int width, int height, int duration) const;
    void saveAppend(const std::string& outputFilePath, const std::vector<float>& feature) const;
    void saveDenseFeatures(const std::string& outputFilePath, 
                           const std::vector<cv::Vec3i>& points,
                           const std::vector<std::vector<float>>& features,
                           int width, int height, int duration) const;
    void saveAppendDenseFeatures(const std::string& outputFilePath,
                                 const std::vector<cv::Vec3i>& points,
                                 const std::vector<std::vector<float>>& features) const;
    void saveHeaderDenseFeatures(const std::string& outputFilePath, const std::string& tmpDataFilePath,
                                 int width, int height, int duration, int featureNums, int featureDims) const;
    void load(const std::string& inputFilePath, std::vector<float>& feature,
              int& width, int& height, int& duration) const;
    void loadDenseFeatures(const std::string& inputFilePath, std::vector<cv::Vec3i>& points,
                           std::vector<std::vector<float>>& features,
                           int& width, int& height, int& duration) const;
    void visualizeFeature(const std::vector<float>& feature,
                          int width, int height) const;
    void visualizeDenseFeature(const std::vector<cv::Vec3i>& points,
                               const std::vector<std::vector<float>>& features,
                               int width, int height, int duration,
                               const std::vector<int>& neighborhoodSizes) const;

    static std::vector<std::string> getFeatureNames() {
        return std::vector<std::string>{"intensity", "x_derivative", "y_derivative", "t_derivative", "x_flow", "y_flow"};
    }

private:
    std::vector<float> extractIntensityFeature(const std::vector<cv::Mat1b>& video) const;
    std::vector<float> extractXDerivativeFeature(const std::vector<cv::Mat1b>& video) const;
    std::vector<float> extractYDerivativeFeature(const std::vector<cv::Mat1b>& video) const;
    std::vector<float> extractTDerivativeFeature(const std::vector<cv::Mat1b>& video) const;
    std::vector<std::vector<float>> extractFlowFeature(const std::vector<cv::Mat1b>& video) const;

    std::vector<float> extractIntensityFeature(const cv::Mat1b& frame) const;
    std::vector<float> extractXDerivativeFeature(const cv::Mat1b& frame) const;
    std::vector<float> extractYDerivativeFeature(const cv::Mat1b& frame) const;
    std::vector<float> extractTDerivativeFeature(const cv::Mat1b& prev, const cv::Mat1b& next) const;
    std::vector<std::vector<float>> extractFlowFeature(const cv::Mat1b& prev, const cv::Mat1b& next) const;

    std::vector<float> getNeighborhoodFeatures(const std::vector<float>& features,
                                               const cv::Vec3i& centerPoint,
                                               const std::vector<int>& neighborhoodSizes,
                                               int width, int height) const;
    int calculateFeatureIndex(int x, int y, int t, int width, int height) const;
};

}
}