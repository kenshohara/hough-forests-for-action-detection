#ifndef HOUGH_FORESTS
#define HOUGH_FORESTS

#include "RandomForests.hpp"
#include "HoughForestsParameters.h"
#include "TreeParameters.h"
#include "STIPNode.h"
#include "Storage.h"
#include "Utils.h"
#include "ActionHoughQuickShift.h"

#include <opencv2/core/core.hpp>

#include <map>
#include <tuple>
#include <utility>
#include <memory>
#include <array>

namespace nuisken {
namespace houghforests {

class HoughForests {
   private:
    typedef std::shared_ptr<randomforests::STIPNode::FeatureType> FeaturePtr;
    typedef std::shared_ptr<randomforests::STIPNode::LeafType> LeafPtr;
    typedef ActionHoughQuickShift MSType;
    typedef storage::VoteInfo<3> VoteInfo;
    typedef storage::VotesInfoMap<3> VotesInfoMap;
    typedef storage::DetectionResult<4> DetectionResult;
    typedef storage::FeatureVoteInfo<3> FeatureVoteInfo;

    const int S = 3;

   private:
    randomforests::RandomForests<randomforests::STIPNode> randomForests_;

    std::vector<std::unique_ptr<MSType>> meanShifts_;

    HoughForestsParameters parameters_;

    std::vector<std::vector<int>> trainingDataWidths_;

    int maxNumberOfThreads_;

   private:
    randomforests::STIPNode stipNode_;

   public:
    HoughForests(int maxNumberOfThreads = 1) : maxNumberOfThreads_(maxNumberOfThreads){};
    HoughForests(const randomforests::STIPNode& stipNode, const HoughForestsParameters& parameters,
                 int maxNumberOfThreads = 1)
            : stipNode_(stipNode),
              randomForests_(stipNode, parameters.getTreeParameters()),
              parameters_(parameters),
              maxNumberOfThreads_(maxNumberOfThreads){};
    virtual ~HoughForests(){};

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

    int getMaxNumberOfThreads() const { return maxNumberOfThreads_; }

    // void train(const std::vector<FeaturePtr>& features);

    std::vector<float> getVotingSpace(int classLabel, int spatialStep, int durationStep,
                                      const std::vector<double>& scales) const;

    std::vector<float> calculateScores(const std::vector<FeaturePtr>& features,
                                       const cv::Vec4i& calculationPosition);

    void save(const std::string& directoryPath) const;
    void load(const std::string& directoryPath);

   private:
    void calculateVotes(const std::vector<FeaturePtr>& features, int scaleIndex,
                        VotesInfoMap& votesInfoMap) const;
    void calculateVotesBasedOnOnePatch(const FeaturePtr& feature, int scaleIndex,
                                       const std::vector<LeafPtr>& votingData,
                                       VotesInfoMap& votesInfoMap) const;
    void inputToMeanShift(VotesInfoMap& votesInfoMap);
    cv::Vec3f calculateVotingPoint(const FeaturePtr& feature, double scale,
                                   const randomforests::STIPLeaf::FeatureInfo& featureInfo) const;
    void initializeMeanShifts();
    std::vector<LocalMaxima> findLocalMaxima();
    LocalMaxima findOneClassLocalMaxima(std::unique_ptr<MSType>& meanShift);
    std::vector<cv::Vec4f> prepareGridPoints() const;
    std::vector<LocalMaxima> verifyLocalMaxima(const VotesInfoMap& votesInfoMap,
                                               const std::vector<LocalMaxima>& localMaxima) const;
    std::vector<LocalMaxima> thresholdLocalMaxima(std::vector<LocalMaxima> localMaxima) const;
};
}
}

#endif