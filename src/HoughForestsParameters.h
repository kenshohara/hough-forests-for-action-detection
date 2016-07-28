#ifndef HOUGH_FORESTS_PARAMETERS
#define HOUGH_FORESTS_PARAMETERS

#include "TreeParameters.h"

#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <vector>

namespace nuisken {
namespace houghforests {

class HoughForestsParameters {
   private:
    std::size_t width_;
    std::size_t height_;
    std::vector<double> scales_;
    int baseScale_;

    int nClasses_;

    double sigma_;
    double tau_;
    double scaleBandwidth_;

    std::vector<int> binSizes_;

    int spatialStep_;
    int temporalStep_;

    int votesDeleteStep_;
    int votesBufferLength_;

    int invalidLeafSizeThreshold_;

    std::vector<double> scoreThresholds_;

    std::vector<std::size_t> averageDurations_;
    std::vector<double> averageAspectRatios_;

    double iouThreshold_;

    bool hasNegativeClass_;

    bool isBackprojection_;

    randomforests::TreeParameters treeParameters_;

   public:
    HoughForestsParameters(){};
    HoughForestsParameters(std::size_t width, std::size_t height, const std::vector<double>& scales,
                           int baseScale, int nClasses, double sigma, double tau,
                           double scaleBandwidth, int spatialStep, int temporalStep,
                           const std::vector<int>& binSizes, int votesDeleteStep,
                           int votesBufferLength, int invalidLeafSizeThreshold,
                           const std::vector<double>& scoreThresholds,
                           const std::vector<std::size_t>& averageDurations,
                           const std::vector<double>& averageAspectRatios, double iouThreshold,
                           bool hasNegativeClass, bool isBackprojection,
                           const randomforests::TreeParameters& treeParameters)
            : width_(width),
              height_(height),
              scales_(scales),
              baseScale_(baseScale),
              nClasses_(nClasses),
              sigma_(sigma),
              tau_(tau),
              scaleBandwidth_(scaleBandwidth),
              binSizes_(binSizes),
              spatialStep_(spatialStep),
              temporalStep_(temporalStep),
              votesDeleteStep_(votesDeleteStep),
              votesBufferLength_(votesBufferLength),
              invalidLeafSizeThreshold_(invalidLeafSizeThreshold),
              scoreThresholds_(scoreThresholds),
              averageDurations_(averageDurations),
              averageAspectRatios_(averageAspectRatios),
              iouThreshold_(iouThreshold),
              hasNegativeClass_(hasNegativeClass),
              isBackprojection_(isBackprojection),
              treeParameters_(treeParameters){};

    std::size_t getWidth() const { return width_; }

    std::size_t getHeight() const { return height_; }

    std::vector<double> getScales() const { return scales_; }

    double getScale(int index) const { return scales_.at(index); }

    int getBaseScale() const { return baseScale_; }

    double getSigma() const { return sigma_; }

    double getTau() const { return tau_; }

    double getScaleBandwidth() const { return scaleBandwidth_; }

    std::vector<int> getBinSizes() const { return binSizes_; }

    int getSpatialStep() const { return spatialStep_; }

    int getTemporalStep() const { return temporalStep_; }

    int getVotesDeleteStep() const { return votesDeleteStep_; }

    int getVotesBufferLength() const { return votesBufferLength_; }

    int getInvalidLeafSizeThreshold() const { return invalidLeafSizeThreshold_; }

    double getScoreThreshold(int classLabel) const { return scoreThresholds_.at(classLabel); };

    std::size_t getAverageDuration(int classLabel) const {
        return averageDurations_.at(classLabel);
    }

    double getAverageAspectRatio(int classLabel) const {
        return averageAspectRatios_.at(classLabel);
    }

    double getIoUThreshold() const { return iouThreshold_; }

    bool isBackprojection() const { return isBackprojection_; }

    bool hasNegativeClass() const { return hasNegativeClass_; }

    int getNumberOfClasses() const { return nClasses_; }

    int getNumberOfPositiveClasses() const {
        if (hasNegativeClass_) {
            return getNumberOfClasses() - 1;
        } else {
            return getNumberOfClasses();
        }
    }

    int getNegativeLabel() const {
        if (hasNegativeClass_) {
            return getNumberOfClasses() - 1;
        } else {
            return -1;
        }
    }

    randomforests::TreeParameters getTreeParameters() const { return treeParameters_; }

    void setTreeParameters(const randomforests::TreeParameters& treeParameters) {
        treeParameters_ = treeParameters;
    }
};
}
}

#endif