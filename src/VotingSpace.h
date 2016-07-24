#ifndef VOTING_SPACE
#define VOTING_SPACE

#include "Utils.h"

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
    static const int S = 3;
    using Point = std::array<float, DIMENSION_SIZE_>;

    cv::Mat1f votingSpace_;
    std::unordered_map<std::size_t, float> allVotes_;
    std::unordered_map<std::size_t, float> newVotes_;
    std::vector<double> scales_;
    std::vector<int> steps_;
    std::vector<int> binSizes_;
    double sigma_;
    double tau_;
    double scaleBandwidth_;
    std::size_t maxT_;
    std::size_t minT_;
    std::size_t deleteStep_;

    std::vector<cv::Vec4i> gridPoints_;

   public:
    VotingSpace(std::size_t width, std::size_t height, std::size_t nScales,
                const std::vector<double>& scales, const std::vector<int>& steps,
                const std::vector<int>& binSizes, double sigma, double tau, double scaleBandwidth,
                std::size_t deleteStep, std::size_t bufferLength)
            : scales_(scales),
              steps_(steps),
              binSizes_(binSizes),
              sigma_(sigma),
              tau_(tau),
              scaleBandwidth_(scaleBandwidth),
              maxT_(bufferLength),
              minT_(0),
              deleteStep_(deleteStep) {
        for (int axis = 0; axis < steps_.size(); ++axis) {
            steps_.at(axis) /= binSizes_.at(axis);
        }
        sigma_ /= binSizes.at(X);
        tau_ /= binSizes.at(T);
        maxT_ /= binSizes.at(T);
        deleteStep_ /= binSizes.at(T);

        std::vector<int> sizes = {static_cast<int>(bufferLength / binSizes_.at(T)),
                                  static_cast<int>(height / binSizes_.at(Y)),
                                  static_cast<int>(width / binSizes_.at(X)),
                                  static_cast<int>(nScales)};
        votingSpace_.create(sizes.size(), sizes.data());
        votingSpace_ = 0.0;

        initializeGridPoints();
    };
    ~VotingSpace(){};

    void inputVote(const cv::Vec3i& point, std::size_t scaleIndex, float weight);
    void deleteOldVotes();

    std::vector<cv::Vec4f> getOriginalGridPoints() const;
    std::vector<float> getGridVotingScores() const;

    cv::Vec4i binPoint(const cv::Vec4i& originalPoint) const;
    int binT(int t) const;
    cv::Vec4i calculateOriginalPoint(const cv::Vec4i& discretizedPoint) const;
    int calculateOriginalT(int binnedT) const;

    std::size_t getMaxT() const { return maxT_; }
    std::size_t getMinT() const { return minT_; }

   private:
    void initializeGridPoints();
};
}
}

#endif