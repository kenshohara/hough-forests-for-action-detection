#include "LocalMaximaFinder.h"

#include <algorithm>
#include <chrono>

namespace nuisken {
namespace houghforests {

LocalMaxima LocalMaximaFinder::findLocalMaxima(const VotingSpace& votingSpace,
                                               double scoreThreshold, std::size_t voteBeginT,
                                               std::size_t voteEndT) const {
    std::size_t bandwidthRange = 3.0 * tau_;
    std::size_t findBeginT = (voteBeginT < bandwidthRange) ? 0 : voteBeginT - bandwidthRange;
    findBeginT = votingSpace.discretizePoint(findBeginT);
    findBeginT = std::max(findBeginT, votingSpace.getMinT());
    std::size_t findEndT = voteEndT + bandwidthRange;
    findEndT = votingSpace.discretizePoint(findEndT);
    findEndT = std::min(findEndT, votingSpace.getMaxT());
    voteBeginT = votingSpace.discretizePoint(voteBeginT);
    voteEndT = votingSpace.discretizePoint(voteEndT);

    std::vector<Point> gridPoints = votingSpace.getGridPoints(findBeginT, findEndT);
    std::vector<double> densities = votingSpace.getGridVotingScores(findBeginT, findEndT);

    double tau = tau_ * votingSpace.getDiscretizeRatio();
    double sigma = sigma_ * votingSpace.getDiscretizeRatio();
    std::vector<double> bandwidths = {tau, sigma, scaleBandwidth_};
    std::vector<int> bandDimensions = {TEMPORAL_DIMENSION_SIZE_, SPATIAL_DIMENSION_SIZE_,
                                       SCALE_DIMENSION_SIZE_};

    // std::cout << "quick shift" << std::endl;
    KDE gridKde(gridPoints, bandwidths, bandDimensions);
    gridKde.buildTree();
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
    std::vector<std::array<float, DIMENSION_SIZE_>> votingPoints;
    std::vector<float> weights;
    votingSpace.getVotes(votingPoints, weights, voteBeginT, voteEndT);
    if (votingPoints.empty()) {
        return {};
    }
    KDE voteKde(votingPoints, weights, bandwidths, bandDimensions);
    voteKde.buildTree();

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
}
}