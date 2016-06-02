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
    std::vector<std::size_t> sizes_;
    std::vector<double> scales_;
    int baseScale_;

    int numberOfClasses_;

    double sigma_;
    double tau_;
    double scaleBandwidth_;

    int spatialStep_;
    int temporalStep_;

    int localMaximaSize_;
    bool hasNegativeClass_;

    bool isBackprojection_;

    randomforests::TreeParameters treeParameters_;

   public:
    HoughForestsParameters(){};
    HoughForestsParameters(const std::vector<std::size_t>& sizes, const std::vector<double>& scales,
                           int baseScale, int numberOfClasses, double sigma, double tau,
                           double scaleBandwidth, int spatialStep, int temporalStep,
                           int localMaximaSize, bool hasNegativeClass, bool isBackprojection,
                           const randomforests::TreeParameters& treeParameters)
            : sizes_(sizes),
              scales_(scales),
              baseScale_(baseScale),
              numberOfClasses_(numberOfClasses),
              sigma_(sigma),
              tau_(tau),
              scaleBandwidth_(scaleBandwidth),
              spatialStep_(spatialStep),
              temporalStep_(temporalStep),
              localMaximaSize_(localMaximaSize),
              hasNegativeClass_(hasNegativeClass),
              isBackprojection_(isBackprojection),
              treeParameters_(treeParameters){};

    std::vector<std::size_t> getSizes() const { return sizes_; }

    int getSize(int index) const { return sizes_.at(index); }

    std::vector<double> getScales() const { return scales_; }

    double getScale(int index) const { return scales_.at(index); }

    int getBaseScale() const { return baseScale_; }

    double getSigma() const { return sigma_; }

    double getTau() const { return tau_; }

    double getScaleBandwidth() const { return scaleBandwidth_; }

    double getSpatialStep() const { return spatialStep_; }

    double getTemporalStep() const { return temporalStep_; }

    int getLocalMaximaSize() const { return localMaximaSize_; }

    bool isBackprojection() const { return isBackprojection_; }

    bool hasNegativeClass() const { return hasNegativeClass_; }

    int getNumberOfClasses() const { return numberOfClasses_; }

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