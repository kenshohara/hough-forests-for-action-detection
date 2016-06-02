#include "ActionHoughDataHandler.h"

namespace nuisken {
namespace houghforests {

ActionHoughDataHandler::ActionHoughDataHandler(const std::vector<std::size_t>& sizes,
                                               const std::vector<double>& scales, double sigma,
                                               double tau, double scaleBandwidth)
        : sizes_(sizes),
          scales_(scales),
          sigma_(sigma),
          tau_(tau),
          scaleBandwidth_(scaleBandwidth),
          isBuild_(false),
          maxIndex_(0),
          minIndex_(std::numeric_limits<std::size_t>::max()) {
    scaleData_.resize(scales.size());
}

void ActionHoughDataHandler::addInput(const cv::Vec3i& point, int scaleIndex, double weight) {
    for (int i = 0; i < VIDEO_DIMENSION_SIZE_; ++i) {
        if (point[i] < 0 || point[i] >= sizes_.at(i)) {
            return;
        }
    }

    std::size_t index = computeIndex(point);
    scaleData_.at(scaleIndex)[index] += weight;

    if (index > maxIndex_) {
        maxIndex_ = index;
    }
    if (index < minIndex_) {
        minIndex_ = index;
    }
}

void ActionHoughDataHandler::buildTree() {
    std::vector<std::array<float, DIMENSION_SIZE_>> data;
    std::vector<double> weights;
    data.reserve(getDataPointsSize());
    weights.reserve(getDataPointsSize());

    for (int scaleIndex = 0; scaleIndex < scaleData_.size(); ++scaleIndex) {
        for (const auto& indexAndValue : scaleData_.at(scaleIndex)) {
            std::array<float, DIMENSION_SIZE_> point;
            cv::Vec3i vecPoint = computePoint(indexAndValue.first);
            for (int i = 0; i < VIDEO_DIMENSION_SIZE_; ++i) {
                point.at(i) = vecPoint(i);
            }
            point.back() = scales_.at(scaleIndex);
            data.push_back(point);
            weights.push_back(indexAndValue.second);
        }
    }

    std::vector<double> bandwidths = {tau_, sigma_, scaleBandwidth_};
    std::vector<int> bandDimensions = {TEMPORAL_DIMENSION_SIZE_, SPATIAL_DIMENSION_SIZE_,
                                       SCALE_DIMENSION_SIZE_};

    if (!data.empty()) {
        kernelDensityEstimation_ = std::make_unique<KDE>(data, weights, bandwidths, bandDimensions);
        kernelDensityEstimation_->buildTree();
        isBuild_ = true;
    }
}

std::vector<std::vector<cv::Vec3i>> ActionHoughDataHandler::getDataPoints() const {
    std::vector<std::vector<cv::Vec3i>> scaleDataPoints(scales_.size());

    for (int scaleIndex = 0; scaleIndex < scales_.size(); ++scaleIndex) {
        for (const auto& indexAndValue : scaleData_.at(scaleIndex)) {
            cv::Vec3i point = computePoint(indexAndValue.first);
            scaleDataPoints.at(scaleIndex).push_back(point);
        }
    }

    return scaleDataPoints;
}

std::vector<std::vector<float>> ActionHoughDataHandler::getWeights() const {
    std::vector<std::vector<float>> scaleWeights;
    scaleWeights.resize(scales_.size());

    for (int scaleIndex = 0; scaleIndex < scales_.size(); ++scaleIndex) {
        for (const auto& indexAndValue : scaleData_.at(scaleIndex)) {
            scaleWeights.at(scaleIndex).push_back(indexAndValue.second);
        }
    }

    return scaleWeights;
}

std::array<float, ActionHoughDataHandler::DIMENSION_SIZE_>
ActionHoughDataHandler::convertFromVecToArray(const cv::Vec4f& point) const {
    std::array<float, DIMENSION_SIZE_> arrayPoint;
    for (int i = 0; i < DIMENSION_SIZE_; ++i) {
        arrayPoint[i] = point[i];
    }

    return arrayPoint;
}

cv::Vec4f ActionHoughDataHandler::convertFromArrayToVec(
        const std::array<float, DIMENSION_SIZE_>& point) const {
    cv::Vec4f vecPoint;
    for (int i = 0; i < DIMENSION_SIZE_; ++i) {
        vecPoint[i] = point[i];
    }

    return vecPoint;
}

std::size_t ActionHoughDataHandler::computeIndex(const cv::Vec3i& point) const {
    std::size_t index = point(0) * sizes_.at(1) * sizes_.at(2) + point(1) * sizes_.at(2) + point(2);
    return index;
}

cv::Vec3i ActionHoughDataHandler::computePoint(std::size_t index) const {
    int t = index / (sizes_.at(1) * sizes_.at(2));
    int y = index / sizes_.at(2) % sizes_.at(1);
    int x = index % sizes_.at(2);

    return cv::Vec3i(t, y, x);
}
}
}