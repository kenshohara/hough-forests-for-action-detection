#ifndef ACTION_HOUGH_QUICK_SHIFT
#define ACTION_HOUGH_QUICK_SHIFT

#include "Utils.h"
#include "ActionHoughDataHandler.h"
#include "Storage.h"

#include <boost/dynamic_bitset.hpp>

#include <opencv2/core/core.hpp>

#include <tuple>
#include <array>

namespace nuisken {
namespace houghforests {

class ActionHoughQuickShift : public ActionHoughDataHandler {
   private:
    typedef std::pair<std::uint32_t, float> Match;

   private:
    double threshold_;
    int maxIteration_;

   public:
    ActionHoughQuickShift(const std::vector<std::size_t>& sizes, const std::vector<double>& scales,
                          int maxIteration = 50, double threshold = 0.1);
    ActionHoughQuickShift(const std::vector<std::size_t>& sizes, const std::vector<double>& scales,
                          double sigma, double tau, double scaleBandwidth, int maxIteration = 50,
                          double threshold = 0.1);

    double estimateDensity(const cv::Vec4f& point) const;

    std::array<float, DIMENSION_SIZE_> calculateWeightedSum(
            const std::array<float, DIMENSION_SIZE_>& point, double& density) const;
    cv::Vec4f calculateWeightedMean(const cv::Vec4f& point) const;
    cv::Vec4f calculateWeightedMean(const cv::Vec4f& point, double& density) const;

    LocalMaxima findMode(const std::vector<cv::Vec4f>& gridPoints) const;
    LocalMaxima findMode(const std::vector<cv::Vec4f>& gridPoints,
                         const std::vector<double>& densities) const;
    LocalMaximum findMode(const cv::Vec4f& initialPoint) const;

    double calculateKernel(const cv::Vec4f& point1, const cv::Vec4f& point2) const;
};
}
}

#endif