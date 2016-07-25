#ifndef HOUGH_FORESTS
#define HOUGH_FORESTS

#include "HoughForestsParameters.h"
#include "LocalFeatureExtractor.h"
#include "RandomForests.hpp"
#include "STIPNode.h"
#include "Storage.h"
#include "TreeParameters.h"
#include "Utils.h"
#include "VotingSpace.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <utility>

namespace nuisken {
namespace houghforests {

class HoughForests {
   private:
    using FeaturePtr = std::shared_ptr<randomforests::STIPNode::FeatureType>;
    using LeafPtr = std::shared_ptr<randomforests::STIPNode::LeafType>;
    using VoteInfo = storage::VoteInfo<3>;
    using FeatureVoteInfo = storage::FeatureVoteInfo<3>;
    using VotesInfoMap = storage::VotesInfoMap<3>;
    using DetectionResult = storage::DetectionResult<4>;
    using Cuboid = storage::SpaceTimeCuboid;

    const int S = 3;

   private:
    randomforests::RandomForests<randomforests::STIPNode> randomForests_;

    std::vector<VotingSpace> votingSpaces_;

    HoughForestsParameters parameters_;

    std::vector<std::vector<int>> trainingDataWidths_;

    int nThreads_;

    std::mutex m_;

   private:
    randomforests::STIPNode stipNode_;

   public:
    HoughForests(int nThreads = 1) : nThreads_(nThreads){};
    HoughForests(const randomforests::STIPNode& stipNode, const HoughForestsParameters& parameters,
                 int nThreads = 1)
            : stipNode_(stipNode),
              randomForests_(stipNode, parameters.getTreeParameters()),
              parameters_(parameters),
              nThreads_(nThreads){};
    virtual ~HoughForests(){};

    void HoughForests::train(const std::vector<FeaturePtr>& features);

    // void classify(LocalFeatureExtractor& extractor);
    void detect(LocalFeatureExtractor& extractor,
                std::vector<std::vector<DetectionResult>>& detectionResults);
    void detect(const std::vector<std::string>& featureFilePaths,
                std::vector<std::vector<DetectionResult>>& detectionResults);

    HoughForestsParameters getHoughForestsParameters() const { return parameters_; }

    randomforests::TreeParameters getTreeParameters() const {
        return parameters_.getTreeParameters();
    }

    void setTreeParameters(const randomforests::TreeParameters& treeParameters) {
        parameters_.setTreeParameters(treeParameters);
    }

    void setHoughForestsParameters(const HoughForestsParameters& parameters) {
        parameters_ = parameters;
    }

    void save(const std::string& directoryPath) const;
    void load(const std::string& directoryPath);

   private:
    void initialize();
    std::vector<FeaturePtr> convertFeatureFormats(
            const std::vector<cv::Vec3i>& points,
            const std::vector<std::vector<float>>& descriptors, int nChannels) const;
    void votingProcess(const std::vector<std::vector<FeaturePtr>>& scaleFeatures,
                       std::vector<std::pair<std::size_t, std::size_t>>& minMaxRanges);
    void calculateVotes(const std::vector<FeaturePtr>& features, int scaleIndex,
                        std::vector<std::vector<VoteInfo>>& votesInfo) const;
    void calculateVotes(const FeaturePtr& feature, int scaleIndex,
                        const std::vector<LeafPtr>& leavesData,
                        std::vector<VoteInfo>& votesInfo) const;
    cv::Vec3i calculateVotingPoint(const FeaturePtr& feature, double scale,
                                   const randomforests::STIPLeaf::FeatureInfo& featureInfo) const;
    void inputInVotingSpace(const std::vector<std::vector<VoteInfo>>& votesInfo);
    void getMinMaxVotingT(const std::vector<std::vector<VoteInfo>>& votesInfo,
                          std::vector<std::pair<std::size_t, std::size_t>>& minMaxRanges) const;
    void updateDetectionCuboids(
            const std::vector<std::pair<std::size_t, std::size_t>>& minMaxRanges,
            std::vector<std::vector<Cuboid>>& detectionCuboids) const;
    void updateDetectionCuboids(int classLabel,
                                const std::pair<std::size_t, std::size_t>& minMaxRanges,
                                std::vector<Cuboid>& detectionCuboids) const;
    std::vector<Cuboid> calculateCuboids(const LocalMaxima& localMaxima, double averageAspectRatio,
                                         int averageDuration) const;
    std::vector<Cuboid> performNonMaximumSuppression(const std::vector<Cuboid>& cuboids) const;
    void deleteOldVotes(int classLabel, std::size_t voteMaxT);
    std::vector<float> getVotingSpace(int classLabel) const;
    void visualizeParallel(
            std::vector<cv::Mat3b>& video, int fps, const std::size_t& videoBeginT,
            const std::vector<std::unordered_map<int, std::vector<Cuboid>>>& detectionCuboids,
            const bool& isEnded);
    void visualize(const std::vector<cv::Mat3b>& video, std::size_t videoBeginT,
                   const std::vector<std::vector<Cuboid>>& detectionCuboids) const;
    void visualize(const std::vector<cv::Mat3b>& video, std::size_t videoBeginT,
                   const std::vector<LocalMaxima>& localMaxima) const;
    void visualize(const std::vector<std::vector<float>>& votingSpaces) const;
};
}
}

#endif