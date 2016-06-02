#include "LocalMaximaFinder.h"

namespace nuisken {
namespace houghforests {

LocalMaxima LocalMaximaFinder::findLocalMaxima(const VotingSpace& votingSpace,
                                               std::size_t voteStartT, std::size_t voteEndT) const {
    int bandwidthRange = 3.0 * tau_;
    int findStartT = voteStartT - bandwidthRange;
    int findEndT = voteEndT + bandwidthRange;

    return findLocalMaxima(votingSpace,
                           getGridPoints(findStartT, findEndT, 0, votingSpace.getHeight(), 0,
                                         votingSpace.getWidth(), 0, scales_.size()));
}

LocalMaxima LocalMaximaFinder::findLocalMaxima(const VotingSpace& votingSpace,
                                               const std::vector<Point>& gridPoints) const {
    std::vector<std::array<float, DIMENSION_SIZE_>> votingPoints;
    std::vector<float> weights;
    votingSpace.getVotes(votingPoints, weights, 0, 0);
    if (!votingPoints.empty()) {
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
        if (links.at(i) == -1) {
            localMaxima.push_back(refineLocalMaximum(kde, gridPoints.at(i)));
        }
    }

    return localMaxima;
}

LocalMaximum LocalMaximaFinder::refineLocalMaximum(const KDE& kde, const Point& localMaximumPoint) const {
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
                    Point p = {t, y, x, scales_.at(s)};
                    gridPoints.push_back(p);
                }
            }
        }
    }

    return gridPoints;
}
}
}