#include "VotingSpace.h"
#include "Utils.h"

#include <iostream>

namespace nuisken {
namespace houghforests {

void VotingSpace::inputVote(const cv::Vec3i& point, std::size_t scaleIndex, float weight) {
    cv::Vec4i originalPoint(point.val);
    originalPoint(T) -= minT_;
    originalPoint(S) = scaleIndex;
    cv::Vec4i binnedPoint = binPoint(originalPoint);
    if (binnedPoint(T) < 0 || binnedPoint(T) >= votingSpace_.size[T] || binnedPoint(X) < 0 ||
        binnedPoint(X) >= votingSpace_.size[X] || binnedPoint(Y) < 0 ||
        binnedPoint(Y) >= votingSpace_.size[Y] || scaleIndex < 0 ||
        scaleIndex >= votingSpace_.size[S]) {
        return;
    }
    votingSpace_(binnedPoint) += weight;
}

void VotingSpace::deleteOldVotes() {
    std::vector<cv::Range> srcRanges = {
            cv::Range(deleteStep_, votingSpace_.size[T]), cv::Range(0, votingSpace_.size[Y]),
            cv::Range(0, votingSpace_.size[X]), cv::Range(0, votingSpace_.size[S])};
    std::vector<cv::Range> dstRanges = srcRanges;
    dstRanges.at(T) = cv::Range(0, votingSpace_.size[T] - deleteStep_);
    votingSpace_(srcRanges.data()).copyTo(votingSpace_(dstRanges.data()));
    std::vector<cv::Range> deleteRanges = srcRanges;
    deleteRanges.at(T) = cv::Range(votingSpace_.size[T] - deleteStep_, votingSpace_.size[T]);
    votingSpace_(deleteRanges.data()) = 0.0;

    minT_ += deleteStep_;
    maxT_ += deleteStep_;
}

std::vector<cv::Vec4f> VotingSpace::getOriginalGridPoints() const {
    std::vector<cv::Vec4f> originalGridPoints;
    originalGridPoints.reserve(gridPoints_.size());
    for (const auto& point : gridPoints_) {
        cv::Vec4i originalPoint = calculateOriginalPoint(point);
        originalPoint(T) += calculateOriginalT(minT_);
        cv::Vec4f originalScalePoint(originalPoint(T), originalPoint(Y), originalPoint(X),
                                     scales_.at(originalPoint(S)));
        originalGridPoints.push_back(originalScalePoint);
    }
    return originalGridPoints;
}

std::vector<float> VotingSpace::getGridVotingScores() const {
    std::vector<float> scores;
    scores.reserve(gridPoints_.size());
    for (const auto& point : gridPoints_) {
        scores.push_back(votingSpace_(point));
    }
    return scores;
}

cv::Vec4i VotingSpace::binPoint(const cv::Vec4i& originalPoint) const {
    cv::Vec4i binnedPoint = originalPoint;
    binnedPoint(T) /= binSizes_.at(T);
    binnedPoint(Y) /= binSizes_.at(Y);
    binnedPoint(X) /= binSizes_.at(X);
    return binnedPoint;
}

int VotingSpace::binT(int t) const { return t / binSizes_.at(T); }

cv::Vec4i VotingSpace::calculateOriginalPoint(const cv::Vec4i& binnedPoint) const {
    cv::Vec4i originalPoint = binnedPoint;
    originalPoint(T) *= binSizes_.at(T);
    originalPoint(Y) *= binSizes_.at(Y);
    originalPoint(X) *= binSizes_.at(X);
    return originalPoint;
}

int VotingSpace::calculateOriginalT(int binnedT) const { return binnedT * binSizes_.at(T); }

void VotingSpace::initializeGridPoints() {
    for (std::size_t t = 0; t < votingSpace_.size[T]; t += steps_.at(T)) {
        for (std::size_t y = 0; y < votingSpace_.size[Y]; y += steps_.at(Y)) {
            for (std::size_t x = 0; x < votingSpace_.size[X]; x += steps_.at(X)) {
                for (std::size_t s = 0; s < votingSpace_.size[S]; ++s) {
                    gridPoints_.emplace_back(t, y, x, s);
                }
            }
        }
    }
}
}
}