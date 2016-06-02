#include "VotingSpace.h"
#include "Utils.h"

namespace nuisken {
namespace houghforests {

void VotingSpace::addInput(const cv::Vec3i& point, std::size_t scaleIndex, float weight) {
    if (point(T) < 0 || point(X) < 0 || point(X) >= width_ || point(Y) < 0 || point(Y) >= height_ || scaleIndex < 0 || scaleIndex >= nScales_) {
        return;
    }
    
    std::size_t index = computeIndex(point, scaleIndex);
    votes_[index] += weight;
}

void VotingSpace::deleteOldVotes() {
    int deleteEndT = minT_ + deleteStep_;
    int startIndex = minT_ * height_ * width_* nScales_;
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

std::size_t VotingSpace::computeIndex(const cv::Vec3i& point, std::size_t scaleIndex) const {
    std::size_t index = (point(0) * (height_ * width_ * nScales_)) + (point(1) * (width_ * nScales_)) + (point(2) * nScales_) + scaleIndex;
    return index;
}

void VotingSpace::computePointAndScale(std::size_t index, cv::Vec3i& point, std::size_t& scaleIndex) const {
    int t = index / (height_ * width_ * nScales_);
    int y = index / (width_ * nScales_) % height_;
    int x = index / nScales_ % width_;
    point = cv::Vec3i(t, y, x);

    scaleIndex = index % nScales_;
}

}
}