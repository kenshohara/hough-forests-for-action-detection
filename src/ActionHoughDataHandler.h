#ifndef ACTION_HOUGH_DATA_HANDLER
#define ACTION_HOUGH_DATA_HANDLER

#include "KernelDensityEstimation.h"

#include <opencv2/core/core.hpp>

#include <unordered_map>
#include <memory>

namespace nuisken {
namespace houghforests {

class ActionHoughDataHandler {
   public:
    static const int DIMENSION_SIZE_ = 4;
    static const int VIDEO_DIMENSION_SIZE_ = 3;
    static const int SPATIAL_DIMENSION_SIZE_ = 2;
    static const int TEMPORAL_DIMENSION_SIZE_ = 1;
    static const int SCALE_DIMENSION_SIZE_ = 1;

    typedef KernelDensityEstimation<float, DIMENSION_SIZE_> KDE;

   protected:
    std::unique_ptr<KDE> kernelDensityEstimation_;

   private:
    typedef std::unordered_map<std::size_t, float> DataStorage;

    std::vector<DataStorage> scaleData_;
    std::vector<std::size_t> sizes_;
    std::vector<double> scales_;
    double sigma_;
    double tau_;
    double scaleBandwidth_;
    double threshold_;
    int maxIteration_;
    bool isBuild_;

    std::size_t maxIndex_;
    std::size_t minIndex_;

   public:
    ActionHoughDataHandler(const std::vector<std::size_t>& sizes, const std::vector<double>& scales,
                           double sigma, double tau, double scaleBandwidth);
    ~ActionHoughDataHandler(){};

    void addInput(const cv::Vec3i& point, int scaleIndex, double weight);

    virtual void buildTree();

    std::vector<std::vector<cv::Vec3i>> getDataPoints() const;
    std::vector<std::vector<float>> getWeights() const;

    std::vector<DataStorage> getData() const { return scaleData_; }

    void setData(const std::vector<DataStorage>& scaleData) { scaleData_ = scaleData; }

    std::vector<std::size_t> getSizes() const { return sizes_; }

    std::vector<double> getScales() const { return scales_; }

    int getDataPointsSize() const {
        return std::accumulate(std::begin(scaleData_), std::end(scaleData_), 0,
                               [](int sum, const DataStorage& a) { return sum + a.size(); });
    }

    bool isBuild() const { return isBuild_; }

    double getSigma() const { return sigma_; }

    double getTau() const { return tau_; }

    double getScaleBandwidth() const { return scaleBandwidth_; }

    void setSigma(double sigma) {
        sigma_ = sigma;
        if (kernelDensityEstimation_ != nullptr) {
            kernelDensityEstimation_->setBandwidths({tau_, sigma_, scaleBandwidth_});
        }
    }

    void setTau(double tau) {
        tau_ = tau;
        if (kernelDensityEstimation_ != nullptr) {
            kernelDensityEstimation_->setBandwidths({tau_, sigma_, scaleBandwidth_});
        }
    }

    void setScaleBandwidth(double scaleBandwidth) {
        scaleBandwidth_ = scaleBandwidth;
        if (kernelDensityEstimation_ != nullptr) {
            kernelDensityEstimation_->setBandwidths({tau_, sigma_, scaleBandwidth_});
        }
    }

    cv::Vec3i getMaxIndexPoint() const { return computePoint(maxIndex_); }

    cv::Vec3i getMinIndexPoint() const { return computePoint(minIndex_); }

    cv::Vec3i computePoint(std::size_t index) const;
    std::size_t computeIndex(const cv::Vec3i& point) const;

   protected:
    void setIsBuild(bool isBuild) { isBuild_ = isBuild; }

    std::array<float, DIMENSION_SIZE_> convertFromVecToArray(const cv::Vec4f& point) const;
    cv::Vec4f convertFromArrayToVec(const std::array<float, DIMENSION_SIZE_>& point) const;
};
}
}

#endif