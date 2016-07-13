#ifndef HOUGH_FORESTS
#define HOUGH_FORESTS

#include "HoughForestsParameters.h"
#include "LocalFeatureExtractor.h"
#include "LocalMaximaFinder.h"
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
    typedef std::shared_ptr<randomforests::STIPNode::FeatureType> FeaturePtr;
    typedef std::shared_ptr<randomforests::STIPNode::LeafType> LeafPtr;
    typedef storage::VoteInfo<3> VoteInfo;
    typedef storage::VotesInfoMap<3> VotesInfoMap;
    typedef storage::DetectionResult<4> DetectionResult;
    typedef storage::FeatureVoteInfo<3> FeatureVoteInfo;

    const int S = 3;

   private:
    randomforests::RandomForests<randomforests::STIPNode> randomForests_;

    std::vector<VotingSpace> votingSpaces_;
    LocalMaximaFinder finder_;

    HoughForestsParameters parameters_;

    std::vector<std::vector<int>> trainingDataWidths_;

    int nThreads_;

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

    void detect(LocalFeatureExtractor& extractor);
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

    // std::vector<float> getVotingSpace(int classLabel, int spatialStep, int durationStep,
    //                                  const std::vector<double>& scales) const;

    // std::vector<float> calculateScores(const std::vector<FeaturePtr>& features,
    //                                   const cv::Vec4i& calculationPosition);

    void save(const std::string& directoryPath) const;
    void load(const std::string& directoryPath);

   private:
    void initialize();
    void calculateVotes(const std::vector<FeaturePtr>& features, int scaleIndex,
                        std::vector<std::vector<VoteInfo>>& votesInfo) const;
    std::vector<VoteInfo> calculateVotes(const FeaturePtr& feature, int scaleIndex,
                                         const std::vector<LeafPtr>& leavesData) const;
    cv::Vec3i calculateVotingPoint(const FeaturePtr& feature, double scale,
                                   const randomforests::STIPLeaf::FeatureInfo& featureInfo) const;
    void inputInVotingSpace(const std::vector<std::vector<VoteInfo>>& votesInfo);
    void getMinMaxVotingT(const std::vector<std::vector<VoteInfo>>& votesInfo,
                          std::vector<std::pair<int, int>>& minMaxRanges) const;
    std::vector<LocalMaxima> findLocalMaxima(const std::vector<std::pair<int, int>>& minMaxRanges);
    LocalMaxima findLocalMaxima(VotingSpace& votingSpace, double scoreThreshold, int voteStartT,
                                int voteEndT);
    std::vector<LocalMaxima> thresholdLocalMaxima(std::vector<LocalMaxima> localMaxima) const;
    void deleteOldVotes(int classLabel, int voteMaxT);
    void visualize(const std::vector<cv::Mat3b>& video, int videoStartT,
                   const std::vector<LocalMaxima>& localMaxima) const;
};
}
}

#endif