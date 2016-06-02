#include "ActionHoughQuickShift.h"
#include "RandomGenerator.h"

#include <boost/timer.hpp>
#include <boost/dynamic_bitset.hpp>

#include <limits>
#include <random>
#include <deque>

namespace nuisken {
namespace houghforests {

ActionHoughQuickShift::ActionHoughQuickShift(const std::vector<std::size_t>& sizes,
                                             const std::vector<double>& scales, int maxIteration,
                                             double threshold)
        : ActionHoughDataHandler(sizes, scales, 0.0, 0.0, 0.0),
          threshold_(threshold),
          maxIteration_(maxIteration) {}

ActionHoughQuickShift::ActionHoughQuickShift(const std::vector<std::size_t>& sizes,
                                             const std::vector<double>& scales, double sigma,
                                             double tau, double scaleBandwidth, int maxIteration,
                                             double threshold)
        : ActionHoughDataHandler(sizes, scales, sigma, tau, scaleBandwidth),
          threshold_(threshold),
          maxIteration_(maxIteration) {}

double ActionHoughQuickShift::estimateDensity(const cv::Vec4f& point) const {
    if (isBuild()) {
        std::array<float, DIMENSION_SIZE_> pointArray = convertFromVecToArray(point);
        return kernelDensityEstimation_->estimateDensity(pointArray);
    } else {
        return 0.0;
    }
}

std::array<float, ActionHoughQuickShift::DIMENSION_SIZE_>
ActionHoughQuickShift::calculateWeightedSum(const std::array<float, DIMENSION_SIZE_>& point,
                                            double& density) const {
    if (isBuild()) {
        return kernelDensityEstimation_->calculateWeightedSum(point, density);
    } else {
        return std::array<float, DIMENSION_SIZE_>();
    }
}

cv::Vec4f ActionHoughQuickShift::calculateWeightedMean(const cv::Vec4f& point) const {
    double density;
    return calculateWeightedMean(point, density);
}

cv::Vec4f ActionHoughQuickShift::calculateWeightedMean(const cv::Vec4f& point,
                                                       double& density) const {
    if (isBuild()) {
        std::array<float, DIMENSION_SIZE_> pointArray = convertFromVecToArray(point);
        std::array<float, DIMENSION_SIZE_> weightedMeanArray =
                kernelDensityEstimation_->calculateWeightedMean(pointArray, density);
        return convertFromArrayToVec(weightedMeanArray);
    } else {
        return cv::Vec4f();
    }
}

LocalMaxima ActionHoughQuickShift::findMode(const std::vector<cv::Vec4f>& gridPoints) const {
    std::vector<double> densities;
    densities.reserve(gridPoints.size());
    for (int i = 0; i < gridPoints.size(); ++i) {
        densities.push_back(estimateDensity(gridPoints.at(i)));
    }

    return findMode(gridPoints, densities);
}

LocalMaxima ActionHoughQuickShift::findMode(const std::vector<cv::Vec4f>& gridPoints,
                                            const std::vector<double>& densities) const {
    std::vector<std::array<float, DIMENSION_SIZE_>> arrayPoints;
    arrayPoints.reserve(gridPoints.size());
    for (const auto& vecPoint : gridPoints) {
        std::array<float, DIMENSION_SIZE_> point;
        for (int i = 0; i < DIMENSION_SIZE_; ++i) {
            point.at(i) = vecPoint(i);
        }
        arrayPoints.push_back(point);
    }

    KDE nearestNeighborSearch(arrayPoints,
                              std::vector<double>{getTau(), getSigma(), getScaleBandwidth()},
                              std::vector<int>{TEMPORAL_DIMENSION_SIZE_, SPATIAL_DIMENSION_SIZE_,
                                               SCALE_DIMENSION_SIZE_});
    nearestNeighborSearch.buildTree();

    std::vector<int> links(gridPoints.size());
    std::fill(std::begin(links), std::end(links), -1);
    for (int i = 0; i < gridPoints.size(); ++i) {
        std::vector<Match> matches = nearestNeighborSearch.findNeighborPoints(arrayPoints.at(i));
        std::sort(std::begin(matches), std::end(matches),
                  [](const Match& a, const Match& b) { return a.second < b.second; });

        for (int j = 0; j < matches.size(); ++j) {
            double kernel = nearestNeighborSearch.calculateKernel(
                    arrayPoints.at(i), arrayPoints.at(matches.at(j).first));
            if (kernel < std::numeric_limits<double>::epsilon()) {
                continue;
            }

            if (densities.at(i) < densities.at(matches.at(j).first)) {
                links.at(i) = matches.at(j).first;
                break;
            }
        }
    }

    std::vector<std::array<float, DIMENSION_SIZE_>> localMaximumPoints;
    std::vector<double> localMaximumDensities;
    for (int i = 0; i < links.size(); ++i) {
        if (links.at(i) == -1) {
            std::array<float, DIMENSION_SIZE_> point;
            for (int j = 0; j < DIMENSION_SIZE_; ++j) {
                point.at(j) = gridPoints.at(i)(j);
            }
            localMaximumPoints.push_back(point);
            localMaximumDensities.push_back(densities.at(i));
        }
    }

    KDE localMaximaNeighborSearch(localMaximumPoints,
                                  std::vector<double>{getTau(), getSigma(), getScaleBandwidth()},
                                  std::vector<int>{TEMPORAL_DIMENSION_SIZE_,
                                                   SPATIAL_DIMENSION_SIZE_, SCALE_DIMENSION_SIZE_});
    localMaximaNeighborSearch.buildTree();

    LocalMaxima localMaxima;
    for (int i = 0; i < localMaximumPoints.size(); ++i) {
        bool isAdded = true;
        std::vector<Match> matches =
                localMaximaNeighborSearch.findNeighborPoints(localMaximumPoints.at(i));
        for (int j = 0; j < matches.size(); ++j) {
            double kernel = localMaximaNeighborSearch.calculateKernel(
                    localMaximumPoints.at(i), localMaximumPoints.at(matches.at(j).first));
            double density = localMaximumDensities.at(i);
            double neighborDensity = localMaximumDensities.at(matches.at(j).first);
            if (kernel > std::numeric_limits<double>::epsilon() &&
                (density - neighborDensity) < std::numeric_limits<double>::epsilon() &&
                matches.at(j).second > std::numeric_limits<float>::epsilon() &&
                i > matches.at(j).first) {
                isAdded = false;
            }
        }

        if (isAdded) {
            cv::Vec4f point;
            for (int j = 0; j < DIMENSION_SIZE_; ++j) {
                point(j) = localMaximumPoints.at(i).at(j);
            }
            LocalMaximum localMaximum(point, localMaximumDensities.at(i));
            localMaxima.push_back(localMaximum);
        }
    }

    return localMaxima;
}

LocalMaximum ActionHoughQuickShift::findMode(const cv::Vec4f& initialPoint) const {
    std::array<float, DIMENSION_SIZE_> point = convertFromVecToArray(initialPoint);
    std::array<float, DIMENSION_SIZE_> meanPoint;
    int iteration = 0;
    double distance = 0.0;
    do {
        meanPoint = kernelDensityEstimation_->calculateWeightedMean(point);

        distance = 0.0;
        for (int i = 0; i < DIMENSION_SIZE_; ++i) {
            distance += (point[i] - meanPoint[i]) * (point[i] - meanPoint[i]);
        }
        distance = std::sqrt(distance);
        ++iteration;
        point = meanPoint;
    } while (iteration < maxIteration_ && distance >= threshold_);

    double density = kernelDensityEstimation_->estimateDensity(meanPoint);
    auto convergentPoint = convertFromArrayToVec(meanPoint);
    return LocalMaximum(convergentPoint, density);
}

double ActionHoughQuickShift::calculateKernel(const cv::Vec4f& point1,
                                              const cv::Vec4f& point2) const {
    return kernelDensityEstimation_->calculateKernel(convertFromVecToArray(point1),
                                                     convertFromVecToArray(point2));
}
}
}