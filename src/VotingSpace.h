#ifndef VOTING_SPACE
#define VOTING_SPACE

#include <opencv2/core/core.hpp>

#include <array>
#include <unordered_map>
#include <vector>

namespace nuisken {
namespace houghforests {

class VotingSpace {
   private:
    std::unordered_map<std::size_t, float> votes_;
    std::vector<double> scales_;
    std::size_t width_;
    std::size_t height_;
    std::size_t nScales_;
    std::size_t maxT_;
    std::size_t minT_;
    std::size_t deleteStep_;
    double discretizeRatio_;

   public:
    VotingSpace(std::size_t width, std::size_t height, std::size_t nScales,
                const std::vector<double>& scales, std::size_t deleteStep, std::size_t bufferLength,
                double discretizeRatio)
            : width_(width * discretizeRatio),
              height_(height * discretizeRatio),
              nScales_(nScales),
              scales_(scales),
              maxT_(bufferLength * discretizeRatio),
              minT_(0),
              deleteStep_(deleteStep * discretizeRatio),
              discretizeRatio_(discretizeRatio){};
    ~VotingSpace(){};

    void inputVote(const cv::Vec3i& point, std::size_t scaleIndex, float weight);
    void deleteOldVotes();
    void getVotes(std::vector<std::array<float, 4>>& votingPoints, std::vector<float>& weights,
                  int beginT, int endT) const;

    void computePointAndScale(std::size_t index, cv::Vec3i& point, std::size_t& scaleIndex) const;
    std::array<float, 4> computePoint(std::size_t index) const;
    std::size_t computeIndex(const cv::Vec3i& point, std::size_t scaleIndex) const;
    std::size_t discretizePoint(std::size_t originalPoint) const;
    cv::Vec3i discretizePoint(const cv::Vec3i& originalPoint) const;
    std::size_t calculateOriginalPoint(std::size_t discretizedPoint) const;
    cv::Vec3i calculateOriginalPoint(const cv::Vec3i& discretizedPoint) const;

    std::size_t getVotesCount() const { return votes_.size(); };
    std::size_t getWidth() const { return width_; }
    std::size_t getHeight() const { return height_; }
    std::size_t getMaxT() const { return maxT_; }
    std::size_t getMinT() const { return minT_; }
    double getDiscretizeRatio() const { return discretizeRatio_; }
};
}
}

#endif