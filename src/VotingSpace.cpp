#include "VotingSpace.h"
#include "Utils.h"

namespace nuisken {
namespace houghforests {

void VotingSpace::inputVote(const cv::Vec3i& point, std::size_t scaleIndex, float weight) {
    if (point(T) < 0 || point(X) < 0 || point(X) >= width_ || point(Y) < 0 || point(Y) >= height_ ||
        scaleIndex < 0 || scaleIndex >= nScales_) {
        return;
    }

    std::size_t index = computeIndex(point, scaleIndex);
    votes_[index] += weight;
}

void VotingSpace::deleteOldVotes() {
    int deleteEndT = minT_ + deleteStep_;
    int startIndex = minT_ * height_ * width_ * nScales_;
    int endIndex = deleteEndT * height_ * width_ * nScales_;
    for (auto it = std::cbegin(votes_); it != std::cend(votes_); ++it) {
        if (it->first >= startIndex && it->first < endIndex) {
            votes_.erase(it++);
        } else {
            ++it;
        }
    }

    minT_ += deleteStep_;
    maxT_ += deleteStep_;
}

void VotingSpace::getVotes(std::vector<std::array<float, 4>>& votingPoints,
                           std::vector<float>& weights, int startT, int endT) const {
    int startIndex = startT * height_ * width_ * nScales_;
    int endIndex = endT * height_ * width_ * nScales_;
    for (const auto& vote : votes_) {
        if (vote.first >= startIndex && vote.first < endIndex) {
            votingPoints.push_back(computePoint(vote.first));
            weights.push_back(vote.second);
        }
    }
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
}
}