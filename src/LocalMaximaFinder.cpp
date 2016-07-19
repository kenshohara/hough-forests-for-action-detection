#include "LocalMaximaFinder.h"

#include <algorithm>
#include <chrono>

namespace nuisken {
namespace houghforests {

LocalMaxima LocalMaximaFinder::findLocalMaxima(const VotingSpace& votingSpace,
                                               double scoreThreshold, std::size_t voteStartT,
                                               std::size_t voteEndT) const {
    std::size_t bandwidthRange = 3.0 * tau_;
    std::size_t findStartT = (voteStartT < bandwidthRange) ? 0 : voteStartT - bandwidthRange;
    findStartT = votingSpace.discretizePoint(findStartT);
    std::size_t findEndT = voteEndT + bandwidthRange;
    findEndT = votingSpace.discretizePoint(findEndT);

    std::size_t tStep = steps_.at(T) * votingSpace.getDiscretizeRatio();
    std::size_t yStep = steps_.at(Y) * votingSpace.getDiscretizeRatio();
    std::size_t xStep = steps_.at(X) * votingSpace.getDiscretizeRatio();
    std::vector<Point> gridPoints =
            getGridPoints(findStartT, findEndT, tStep, 0, votingSpace.getHeight(), yStep, 0,
                          votingSpace.getWidth(), xStep, 0, scales_.size());
    return findLocalMaxima(votingSpace, scoreThreshold, votingSpace.discretizePoint(voteStartT),
                           votingSpace.discretizePoint(voteEndT), gridPoints);
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
    double tau = tau_ * votingSpace.getDiscretizeRatio();
    double sigma = sigma_ * votingSpace.getDiscretizeRatio();
    std::vector<double> bandwidths = {tau, sigma, scaleBandwidth_};
    std::vector<int> bandDimensions = {TEMPORAL_DIMENSION_SIZE_, SPATIAL_DIMENSION_SIZE_,
                                       SCALE_DIMENSION_SIZE_};
    KDE voteKde(votingPoints, weights, bandwidths, bandDimensions);
    voteKde.buildTree();

    // std::cout << "estimate densities" << std::endl;
    std::vector<double> densities;
    densities.reserve(gridPoints.size());
    for (int i = 0; i < gridPoints.size(); ++i) {
        densities.push_back(voteKde.estimateDensity(gridPoints.at(i)));
    }

    KDE gridKde(gridPoints, bandwidths, bandDimensions);
    gridKde.buildTree();

    // std::cout << "quick shift" << std::endl;
    std::vector<int> links(gridPoints.size());
    std::fill(std::begin(links), std::end(links), -1);
    for (int i = 0; i < gridPoints.size(); ++i) {
        std::vector<Match> matches = gridKde.findNeighborPoints(gridPoints.at(i));
        std::sort(std::begin(matches), std::end(matches),
                  [](const Match& a, const Match& b) { return a.second < b.second; });

        for (int j = 0; j < matches.size(); ++j) {
            double kernel =
                    gridKde.calculateKernel(gridPoints.at(i), gridPoints.at(matches.at(j).first));
            if (kernel < std::numeric_limits<double>::epsilon()) {
                continue;
            }

            if (densities.at(i) < densities.at(matches.at(j).first)) {
                links.at(i) = matches.at(j).first;
                break;
            }
        }
    }

    // std::cout << "mean shift" << std::endl;
    LocalMaxima localMaxima;
    for (int i = 0; i < links.size(); ++i) {
        if (links.at(i) == -1 && densities.at(i) > scoreThreshold) {
            localMaxima.push_back(refineLocalMaximum(voteKde, gridPoints.at(i)));
        }
    }

    for (auto& localMaximum : localMaxima) {
        cv::Vec4f point = localMaximum.getPoint();
        cv::Vec3i pointWithoutScale(point(T), point(Y), point(X));
        cv::Vec3i originalPointWithoutScale = votingSpace.calculateOriginalPoint(pointWithoutScale);
        localMaximum.setPoint(cv::Vec4f(originalPointWithoutScale(T), originalPointWithoutScale(Y),
                                        originalPointWithoutScale(X), point(3)));
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
        std::size_t beginT, std::size_t endT, std::size_t stepT, std::size_t beginY,
        std::size_t endY, std::size_t stepY, std::size_t beginX, std::size_t endX,
        std::size_t stepX, std::size_t beginSIndex, std::size_t endSIndex) const {
    std::vector<Point> gridPoints;
    for (std::size_t t = beginT; t < endT; t += stepT) {
        for (std::size_t y = beginY; y < endY; y += stepY) {
            for (std::size_t x = beginX; x < endX; x += stepX) {
                for (std::size_t s = beginSIndex; s < endSIndex; ++s) {
                    gridPoints.emplace_back(Point{static_cast<float>(t), static_cast<float>(y),
                                                  static_cast<float>(x),
                                                  static_cast<float>(scales_.at(s))});
                }
            }
        }
    }

    return gridPoints;
}

}
}