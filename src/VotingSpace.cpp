#include "VotingSpace.h"
#include "Utils.h"

#include <iostream>

namespace nuisken {
namespace houghforests {

void VotingSpace::inputVote(const cv::Vec3i& point, std::size_t scaleIndex, float weight) {
    cv::Vec3i discretizedPoint = discretizePoint(point);
    if (discretizedPoint(T) < static_cast<long long>(minT_) || discretizedPoint(X) < 0 ||
        discretizedPoint(X) >= width_ || discretizedPoint(Y) < 0 ||
        discretizedPoint(Y) >= height_ || scaleIndex < 0 || scaleIndex >= nScales_) {
        return;
    }
    std::size_t index = computeIndex(discretizedPoint, scaleIndex);
    newVotes_[index] += weight;
    allVotes_[index] += weight;
}

void VotingSpace::deleteOldVotes() {
    int deleteEndT = minT_ + deleteStep_;
    std::size_t beginIndex = computeIndex(cv::Vec3i(minT_, 0, 0), 0);
    std::size_t endIndex = computeIndex(cv::Vec3i(deleteEndT, 0, 0), 0);
    for (auto it = std::cbegin(allVotes_); it != std::cend(allVotes_);) {
        if (it->first >= beginIndex && it->first < endIndex) {
            allVotes_.erase(it++);
        } else {
            ++it;
        }
    }
    minT_ += deleteStep_;
    maxT_ += deleteStep_;

    for (auto& point : gridPoints_) {
        point.at(T) += deleteStep_;
    }
    std::size_t nDeletedGrids = computeGridIndex(deleteStep_);
    gridVotingScores_.erase(std::begin(gridVotingScores_),
                            std::begin(gridVotingScores_) + nDeletedGrids);
    for (std::size_t i = 0; i < nDeletedGrids; ++i) {
        gridVotingScores_.push_back(0.0);
    }
}

void VotingSpace::getVotes(std::vector<std::array<float, 4>>& votingPoints,
                           std::vector<float>& weights, int beginT, int endT) const {
    int beginIndex = computeIndex(cv::Vec3i(beginT, 0, 0), 0);
    int endIndex = computeIndex(cv::Vec3i(endT, 0, 0), 0);
    for (const auto& vote : allVotes_) {
        if (vote.first >= beginIndex && vote.first < endIndex) {
            votingPoints.push_back(computePoint(vote.first));
            weights.push_back(vote.second);
        }
    }
}

void VotingSpace::getNewVotes(std::vector<std::array<float, 4>>& votingPoints,
                              std::vector<float>& weights) const {
    for (const auto& vote : newVotes_) {
        votingPoints.push_back(computePoint(vote.first));
        weights.push_back(vote.second);
    }
}

void VotingSpace::renew() {
    std::vector<Point> votingPoints;
    std::vector<float> weights;
    getNewVotes(votingPoints, weights);
    newVotes_.clear();
    if (votingPoints.empty()) {
        return;
    }

    double tau = tau_ * getDiscretizeRatio();
    double sigma = sigma_ * getDiscretizeRatio();
    std::vector<double> bandwidths = {tau, sigma, scaleBandwidth_};
    std::vector<int> bandDimensions = {TEMPORAL_DIMENSION_SIZE_, SPATIAL_DIMENSION_SIZE_,
                                       SCALE_DIMENSION_SIZE_};
    KDE voteKde(votingPoints, weights, bandwidths, bandDimensions);
    voteKde.buildTree();

    for (int i = 0; i < gridPoints_.size(); ++i) {
        gridVotingScores_.at(i) += voteKde.estimateDensity(gridPoints_.at(i));
    }
}

std::size_t VotingSpace::discretizePoint(std::size_t originalPoint) const {
    return originalPoint * discretizeRatio_;
}

std::size_t VotingSpace::calculateOriginalPoint(std::size_t discretizedPoint) const {
    return discretizedPoint / discretizeRatio_;
}

cv::Vec3i VotingSpace::discretizePoint(const cv::Vec3i& originalPoint) const {
    return originalPoint * discretizeRatio_;
}

cv::Vec3i VotingSpace::calculateOriginalPoint(const cv::Vec3i& discretizedPoint) const {
    return discretizedPoint / discretizeRatio_;
}

std::size_t VotingSpace::computeIndex(const cv::Vec3i& point, std::size_t scaleIndex) const {
    std::size_t index = (point(0) * (height_ * width_ * nScales_)) +
                        (point(1) * (width_ * nScales_)) + (point(2) * nScales_) + scaleIndex;
    return index;
}

void VotingSpace::computePointAndScale(std::size_t index, cv::Vec3i& point,
                                       std::size_t& scaleIndex) const {
    std::size_t t = index / (height_ * width_ * nScales_);
    std::size_t y = index / (width_ * nScales_) % height_;
    std::size_t x = index / nScales_ % width_;
    point = cv::Vec3i(t, y, x);

    scaleIndex = index % nScales_;
}

std::array<float, 4> VotingSpace::computePoint(std::size_t index) const {
    std::size_t t = index / (height_ * width_ * nScales_);
    std::size_t y = index / (width_ * nScales_) % height_;
    std::size_t x = index / nScales_ % width_;

    std::size_t scaleIndex = index % nScales_;

    std::array<float, 4> point = {t, y, x, scales_.at(scaleIndex)};
    return point;
}

std::size_t VotingSpace::computeGridIndex(std::size_t t) const {
    std::size_t gridT = t / steps_.at(T);
    std::size_t height = height_ / steps_.at(Y);
    std::size_t width = width_ / steps_.at(X);
    std::size_t index = gridT * (height * width * nScales_);
    return index;
}

void VotingSpace::initializeGridPoints() {
    for (std::size_t t = minT_; t < maxT_; t += steps_.at(T)) {
        for (std::size_t y = 0; y < height_; y += steps_.at(Y)) {
            for (std::size_t x = 0; x < width_; x += steps_.at(X)) {
                for (std::size_t s = 0; s < nScales_; ++s) {
                    gridPoints_.emplace_back(Point{static_cast<float>(t), static_cast<float>(y),
                                                   static_cast<float>(x),
                                                   static_cast<float>(scales_.at(s))});
                }
            }
        }
    }
}
}
}