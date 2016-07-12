#include "LocalMaximaFinder.h"

#include <chrono>

namespace nuisken {
namespace houghforests {

LocalMaxima LocalMaximaFinder::findLocalMaxima(const VotingSpace& votingSpace,
                                               double scoreThreshold, std::size_t voteStartT,
                                               std::size_t voteEndT) const {
    int bandwidthRange = 3.0 * tau_;
    int findStartT = voteStartT - bandwidthRange;
    int findEndT = voteEndT + bandwidthRange;

    std::vector<Point> gridPoints = getGridPoints(findStartT, findEndT, 0, votingSpace.getHeight(),
                                                  0, votingSpace.getWidth(), 0, scales_.size());
    return findLocalMaxima(votingSpace, scoreThreshold, voteStartT, voteEndT, gridPoints);
}

LocalMaxima LocalMaximaFinder::findLocalMaxima(const VotingSpace& votingSpace,
                                               double scoreThreshold, std::size_t voteStartT,
                                               std::size_t voteEndT,
                                               const std::vector<Point>& gridPoints) const {
    std::vector<std::array<float, DIMENSION_SIZE_>> votingPoints;
    std::vector<float> weights;
    votingSpace.getVotes(votingPoints, weights, voteStartT, voteEndT);
    if (votingPoints.empty()) {
        return {};
    }

    std::vector<double> bandwidths = {tau_, sigma_, scaleBandwidth_};
    std::vector<int> bandDimensions = {TEMPORAL_DIMENSION_SIZE_, SPATIAL_DIMENSION_SIZE_,
                                       SCALE_DIMENSION_SIZE_};

    KDE kde(votingPoints, weights, bandwidths, bandDimensions);
    kde.buildTree();

    std::vector<double> densities;
    densities.reserve(gridPoints.size());
    for (int i = 0; i < gridPoints.size(); ++i) {
        densities.push_back(kde.estimateDensity(gridPoints.at(i)));
    }

    std::vector<int> links(gridPoints.size());
    std::fill(std::begin(links), std::end(links), -1);
    for (int i = 0; i < gridPoints.size(); ++i) {
        std::vector<Match> matches = kde.findNeighborPoints(gridPoints.at(i));
        std::sort(std::begin(matches), std::end(matches),
                  [](const Match& a, const Match& b) { return a.second < b.second; });

        for (int j = 0; j < matches.size(); ++j) {
            double kernel =
                    kde.calculateKernel(gridPoints.at(i), gridPoints.at(matches.at(j).first));
            if (kernel < std::numeric_limits<double>::epsilon()) {
                continue;
            }

            if (densities.at(i) < densities.at(matches.at(j).first)) {
                links.at(i) = matches.at(j).first;
                break;
            }
        }
    }

    LocalMaxima localMaxima;
    std::vector<Point> localMaximumPoints;
    std::vector<double> localMaximumDensities;
    for (int i = 0; i < links.size(); ++i) {
        if (links.at(i) == -1 && densities.at(i) > scoreThreshold) {
            localMaxima.push_back(refineLocalMaximum(kde, gridPoints.at(i)));
        }
    }

    return localMaxima;
}

LocalMaximum LocalMaximaFinder::refineLocalMaximum(const KDE& kde,
                                                   const Point& localMaximumPoint) const {
    Point point = localMaximumPoint;
    Point meanPoint;
    int iteration = 0;
    double distance = 0.0;
    do {
        meanPoint = kde.calculateWeightedMean(point);

        distance = 0.0;
        for (int i = 0; i < DIMENSION_SIZE_; ++i) {
            distance += (point[i] - meanPoint[i]) * (point[i] - meanPoint[i]);
        }
        distance = std::sqrt(distance);
        ++iteration;
        point = meanPoint;
    } while (iteration < maxIteration_ && distance >= threshold_);

    double density = kde.estimateDensity(meanPoint);
    return LocalMaximum(cv::Vec4f(meanPoint.data()), density);
}

std::vector<LocalMaximaFinder::Point> LocalMaximaFinder::getGridPoints(
        std::size_t startT, std::size_t endT, std::size_t startY, std::size_t endY,
        std::size_t startX, std::size_t endX, std::size_t startSIndex,
        std::size_t endSIndex) const {
    std::vector<Point> gridPoints;
    for (std::size_t t = startT; t < endT; t += steps_.at(T)) {
        for (std::size_t y = startY; y < endY; y += steps_.at(Y)) {
            for (std::size_t x = startX; x < endX; x += steps_.at(X)) {
                for (std::size_t s = startSIndex; s < endSIndex; ++s) {
                    gridPoints.emplace_back(Point{static_cast<float>(t), static_cast<float>(y),
                                                  static_cast<float>(x),
                                                  static_cast<float>(scales_.at(s))});
                }
            }
        }
    }

    return gridPoints;
}

LocalMaxima LocalMaximaFinder::combineNeighborLocalMaxima(const LocalMaxima& localMaxima) const {
    if (localMaxima.empty()) {
        return {};
    }

    LocalMaxima sortedMaxima = localMaxima;
    std::sort(std::begin(sortedMaxima), std::end(sortedMaxima),
              [](const LocalMaximum& a, const LocalMaximum& b) {
                  return a.getValue() > b.getValue();
              });

    std::vector<Point> localMaximumPoints;
    std::vector<float> weights;
    localMaximumPoints.reserve(sortedMaxima.size());
    weights.reserve(sortedMaxima.size());
    for (int i = 0; i < sortedMaxima.size(); ++i) {
        Point point;
        cv::Vec4i vecPoint = sortedMaxima.at(i).getPoint();
        for (int d = 0; d < point.size(); ++d) {
            point.at(d) = vecPoint(d);
        }
        localMaximumPoints.push_back(point);
        weights.push_back(sortedMaxima.at(i).getValue());
    }

    std::vector<double> bandwidths = {tau_, sigma_, scaleBandwidth_};
    std::vector<int> bandDimensions = {TEMPORAL_DIMENSION_SIZE_, SPATIAL_DIMENSION_SIZE_,
                                       SCALE_DIMENSION_SIZE_};

    KDE kde(localMaximumPoints, weights, bandwidths, bandDimensions);
    kde.buildTree();

    std::vector<int> indices;
    for (int i = 0; i < localMaximumPoints.size(); ++i) {
        std::vector<Match> matches = kde.findNeighborPoints(localMaximumPoints.at(i));
        std::sort(std::begin(matches), std::end(matches),
                  [](const Match& a, const Match& b) { return a.second < b.second; });

        bool doesNeighborExist = false;
        for (int j = 0; j < matches.size(); ++j) {
            if (std::find(std::begin(indices), std::end(indices), matches.at(j).first) !=
                std::end(indices)) {
                doesNeighborExist =
                        isNeighbor(cv::Vec4f(localMaximumPoints.at(i).data()),
                                   cv::Vec4f(localMaximumPoints.at(matches.at(j).first).data()));
                if (doesNeighborExist) {
                    break;
                }
            }
        }
        if (!doesNeighborExist) {
            indices.push_back(i);
        }
    }

    LocalMaxima combinedMaxima;
    combinedMaxima.reserve(indices.size());
    for (int index : indices) {
        combinedMaxima.push_back(sortedMaxima.at(index));
    }
    return combinedMaxima;
}

bool LocalMaximaFinder::isNeighbor(const cv::Vec4f& a, const cv::Vec4f& b) const {
    double distance = cv::norm(a - b);
    if (distance < 10.0) {
        return true;
    } else {
        return false;
    }
}
}
}