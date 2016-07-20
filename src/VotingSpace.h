#ifndef VOTING_SPACE
#define VOTING_SPACE

#include "KernelDensityEstimation.h"

#include <opencv2/core/core.hpp>

#include <array>
#include <unordered_map>
#include <vector>

namespace nuisken {
namespace houghforests {

class VotingSpace {
   private:
    static const int DIMENSION_SIZE_ = 4;
    static const int SPATIAL_DIMENSION_SIZE_ = 2;
    static const int TEMPORAL_DIMENSION_SIZE_ = 1;
    static const int SCALE_DIMENSION_SIZE_ = 1;
    using Point = std::array<float, DIMENSION_SIZE_>;
    using KDE = KernelDensityEstimation<float, DIMENSION_SIZE_>;

    std::unordered_map<std::size_t, float> allVotes_;
    std::unordered_map<std::size_t, float> newVotes_;
    std::vector<double> scales_;
    std::size_t width_;
    std::size_t height_;
    std::size_t nScales_;
    std::vector<int> steps_;
    double sigma_;
    double tau_;
    double scaleBandwidth_;
    std::size_t maxT_;
    std::size_t minT_;
    std::size_t deleteStep_;
    double discretizeRatio_;

    std::vector<Point> gridPoints_;
    std::vector<double> gridVotingScores_;

   public:
    VotingSpace(std::size_t width, std::size_t height, std::size_t nScales,
                const std::vector<double>& scales, const std::vector<int>& steps, double sigma,
                double tau, double scaleBandwidth, std::size_t deleteStep, std::size_t bufferLength,
                double discretizeRatio)
            : width_(width * discretizeRatio),
              height_(height * discretizeRatio),
              nScales_(nScales),
              scales_(scales),
              steps_(steps),
              sigma_(sigma * discretizeRatio_),
              tau_(tau * discretizeRatio),
              scaleBandwidth_(scaleBandwidth),
              maxT_(bufferLength * discretizeRatio),
              minT_(0),
              deleteStep_(deleteStep * discretizeRatio),
              discretizeRatio_(discretizeRatio) {
        for (auto& step : steps_) {
            step *= discretizeRatio;
        }

        initializeGridPoints();
        gridVotingScores_ = std::vector<double>(gridPoints_.size(), 0.0);
    };
    ~VotingSpace(){};

    void inputVote(const cv::Vec3i& point, std::size_t scaleIndex, float weight);
    void deleteOldVotes();
    void getVotes(std::vector<std::array<float, 4>>& votingPoints, std::vector<float>& weights,
                  int beginT, int endT) const;
    void getNewVotes(std::vector<std::array<float, 4>>& votingPoints,
                     std::vector<float>& weights) const;
    void renew();

    void computePointAndScale(std::size_t index, cv::Vec3i& point, std::size_t& scaleIndex) const;
    std::array<float, 4> computePoint(std::size_t index) const;
    std::size_t computeIndex(const cv::Vec3i& point, std::size_t scaleIndex) const;
    std::size_t computeGridIndex(const cv::Vec3i& point, std::size_t scaleIndex) const;
    std::size_t discretizePoint(std::size_t originalPoint) const;
    cv::Vec3i discretizePoint(const cv::Vec3i& originalPoint) const;
    std::size_t calculateOriginalPoint(std::size_t discretizedPoint) const;
    cv::Vec3i calculateOriginalPoint(const cv::Vec3i& discretizedPoint) const;

    Point getGridPoint(std::size_t gridIndex) const { return gridPoints_.at(gridIndex); }
    std::vector<Point> getGridPoints() const { return gridPoints_; }
    std::vector<Point> getGridPoints(std::size_t beginT, std::size_t endT) const {
        std::size_t beginIndex = computeGridIndex(cv::Vec3i(beginT - minT_, 0, 0), 0);
        std::size_t endIndex = computeGridIndex(cv::Vec3i(endT - minT_, 0, 0), 0);
        std::vector<Point> gridPoints;
        gridPoints.reserve(endIndex - beginIndex);
        std::copy(std::begin(gridPoints_) + beginIndex, std::end(gridPoints_) + endIndex,
                  std::back_inserter(gridPoints));
        return gridPoints;
    }
    double getGridVotingScore(std::size_t gridIndex) const {
        return gridVotingScores_.at(gridIndex);
    }
    std::vector<double> getGridVotingScores() const { return gridVotingScores_; }
    std::vector<double> getGridVotingScores(std::size_t beginT, std::size_t endT) const {
        std::size_t beginIndex = computeGridIndex(cv::Vec3i(beginT - minT_, 0, 0), 0);
        std::size_t endIndex = computeGridIndex(cv::Vec3i(endT - minT_, 0, 0), 0);
        std::vector<double> votingScores;
        votingScores.reserve(endIndex - beginIndex);
        std::copy(std::begin(gridVotingScores_) + beginIndex,
                  std::end(gridVotingScores_) + endIndex, std::back_inserter(votingScores));
        return votingScores;
    }
    std::size_t getVotesCount() const { return allVotes_.size(); };
    std::size_t getWidth() const { return width_; }
    std::size_t getHeight() const { return height_; }
    std::size_t getMaxT() const { return maxT_; }
    std::size_t getMinT() const { return minT_; }
    double getDiscretizeRatio() const { return discretizeRatio_; }

   private:
    void initializeGridPoints();
};
}
}

#endif