#ifndef KERNEL_DENSITY_ESTIMATION
#define KERNEL_DENSITY_ESTIMATION

#include <nanoflann.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>

namespace nuisken {
/**
* カーネル密度推定
* (ガウシアンカーネル)
*/
template <typename T, std::uint32_t DIM>
class KernelDensityEstimation {
   private:
    template <typename T, std::uint32_t DIM>
    class DataAdaptor {
       private:
        typedef std::array<T, DIM> Point;

       private:
        std::vector<std::array<T, DIM>> data_;

       public:
        DataAdaptor(){};
        DataAdaptor(const std::vector<std::array<T, DIM>>& data) : data_(data){};

        void addNewData(const Point& newData) { data_.push_back(newData); }

        std::vector<Point> getData() const { return data_; }

        Point getData(int index) const { return data_.at(index); }

        // Must return the number of data points
        std::uint32_t kdtree_get_point_count() const { return data_.size(); }

        // Returns the distance between the vector "p1[0:size-1]" and the data point with index
        // "idx_p2" stored in the class:
        T kdtree_distance(const T* p1, const std::uint32_t idx_p2, std::uint32_t size) const {
            T distance = 0.0;
            for (int i = 0; i < DIM; ++i) {
                T diff = p1[i] - data_[idx_p2][i];
                distance += diff * diff;
            }

            return distance;
        }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        double kdtree_get_pt(const std::uint32_t idx, int dim) const { return data_[idx][dim]; }

        // Optional bounding-box computation: return false to default to a standard bbox computation
        // loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it
        //   can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point
        //   clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& bb) const {
            return false;
        }
    };

    typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<T, DataAdaptor<T, DIM>>, DataAdaptor<T, DIM>, DIM,
            std::uint32_t>
            KDTree;
    typedef std::array<T, DIM> Point;

   private:
    std::unique_ptr<KDTree> kdTree_;
    DataAdaptor<T, DIM> dataAdaptor_;
    std::vector<float> weights_;
    std::vector<double> bandwidths_;
    std::vector<int> bandDimensions_;
    std::vector<std::vector<double>> bandKernelLookupTable_;
    std::vector<double> bandIncrements_;
    double bandConstant_;
    double radius_;
    double sumWeights_;
    const int KDTREE_LEAF_SIZE_;
    const int LOOKUP_TABLE_SIZE_;
    const int CALCULATION_RANGE_;

    static const double PI;

   public:
    KernelDensityEstimation(const std::vector<Point>& data, const std::vector<double>& bandwidths,
                            const std::vector<int>& bandDimensions, int calculationRange = 3,
                            int kdTreeLeafSize = 10, int lookupTableSize_ = 100)
            : dataAdaptor_(data),
              bandwidths_(bandwidths),
              bandDimensions_(bandDimensions),
              CALCULATION_RANGE_(calculationRange),
              KDTREE_LEAF_SIZE_(kdTreeLeafSize),
              LOOKUP_TABLE_SIZE_(lookupTableSize_) {
        weights_.resize(data.size());
        std::fill(std::begin(weights_), std::end(weights_), 1.0);
        init();
    };

    KernelDensityEstimation(const std::vector<Point>& data, const std::vector<float>& weights,
                            const std::vector<double>& bandwidths,
                            const std::vector<int>& bandDimensions, int calculationRange = 3,
                            int kdTreeLeafSize = 10, int lookupTableSize_ = 100)
            : dataAdaptor_(data),
              weights_(weights),
              bandwidths_(bandwidths),
              bandDimensions_(bandDimensions),
              CALCULATION_RANGE_(calculationRange),
              KDTREE_LEAF_SIZE_(kdTreeLeafSize),
              LOOKUP_TABLE_SIZE_(lookupTableSize_) {
        init();
    };

    KernelDensityEstimation(const std::vector<double>& bandwidths,
                            const std::vector<int>& bandDimensions, int calculationRange = 3,
                            int kdTreeLeafSize = 10, int lookupTableSize_ = 100)
            : bandwidths_(bandwidths),
              bandDimensions_(bandDimensions),
              CALCULATION_RANGE_(calculationRange),
              KDTREE_LEAF_SIZE_(kdTreeLeafSize),
              LOOKUP_TABLE_SIZE_(lookupTableSize_) {
        init();
    };

    void addNewData(const Point& newData, float weight = 1.0) {
        dataAdaptor_.addNewData(newData);
        weights_.push_back(weight);
    }

    std::vector<Point> getData() const { return dataAdaptor_.getData(); }

    void buildTree() {
        kdTree_ = std::make_unique<KDTree>(
                DIM, dataAdaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(KDTREE_LEAF_SIZE_));
        kdTree_->buildIndex();

        sumWeights_ = std::accumulate(std::begin(weights_), std::end(weights_), 0.0);
    }

    std::vector<std::pair<std::uint32_t, T>> findNeighborPoints(const Point& point) const {
        std::vector<std::pair<std::uint32_t, T>> matches;
        nanoflann::SearchParams params(32, (0.0F), false);
        int matchSize = kdTree_->radiusSearch(point.data(), radius_, matches, params);

        return matches;
    }

    /**
        * 正規化(N=sumWeights_で割る)していない密度を推定
        */
    double estimateDensity(const Point& estimationPoint) const {
        std::vector<std::pair<std::uint32_t, T>> matches = findNeighborPoints(estimationPoint);
        return estimateDensity(estimationPoint, matches);
    }

    double estimateDensity(const Point& estimationPoint,
                           const std::vector<std::pair<std::uint32_t, T>>& matches) const {
        double density = 0.0;
        for (const auto& match : matches) {
            int index = match.first;
            density += calculateKernel(estimationPoint, dataAdaptor_.getData(index)) *
                       weights_.at(index);
        }

        return density;
    }

    Point calculateWeightedMean(const Point& point) const {
        std::vector<std::pair<std::uint32_t, T>> matches = findNeighborPoints(point);

        double totalWeight = 0.0;
        return calculateWeightedMean(point, totalWeight);
    }

    Point calculateWeightedMean(const Point& point, double& density) const {
        Point mean = calculateWeightedSum(point, density);
        if (density >= std::numeric_limits<double>::epsilon()) {
            for (int i = 0; i < DIM; ++i) {
                mean[i] /= density;
            }
        }

        return mean;
    }

    Point calculateWeightedSum(const Point& point, double& density) const {
        std::vector<std::pair<std::uint32_t, T>> matches = findNeighborPoints(point);

        density = 0.0;
        Point sum;
        std::fill(std::begin(sum), std::end(sum), 0.0);

        for (const auto& match : matches) {
            int index = match.first;
            Point matchPoint = dataAdaptor_.getData(index);
            double kernelValue = calculateKernel(point, matchPoint);
            double weight = kernelValue * weights_.at(index);
            density += weight;

            for (int i = 0; i < DIM; ++i) {
                sum[i] += weight * matchPoint[i];
            }
        }

        if (density == 0.0) {
            Point zeroPoint;
            std::fill(std::begin(zeroPoint), std::end(zeroPoint), 0.0);
            return zeroPoint;
        } else {
            return sum;
        }
    }

    double calculateDistance(const Point& point1, const Point& point2, int bandIndex) const {
        int bandStartIndex = 0;
        for (int i = 0; i < bandIndex; ++i) {
            bandStartIndex += bandDimensions_.at(i);
        }

        double distance = 0;
        for (int i = 0; i < bandDimensions_.at(bandIndex); ++i) {
            double diff = point1.at(bandStartIndex + i) - point2.at(bandStartIndex + i);
            distance += diff * diff;
        }

        return distance;
    }

    double calculateKernel(const Point& point1, const Point& point2) const {
        double kernelValue = bandConstant_;

        for (int i = 0; i < bandwidths_.size(); ++i) {
            double distance = calculateDistance(point1, point2, i);
            kernelValue *= calculateKernel(distance, i);
        }

        return kernelValue;
    }

    double calculateKernel(double distance, int bandIndex) const {
        double x = distance / bandIncrements_.at(bandIndex);
        int x0 = std::floor(x);
        int x1 = x0 + 1;
        if (x1 >= bandKernelLookupTable_.at(bandIndex).size()) {
            return 0.0;
        }
        double y0 = bandKernelLookupTable_.at(bandIndex).at(x0);
        double y1 = bandKernelLookupTable_.at(bandIndex).at(x1);
        double alpha = x - x0;
        double y = (1.0 - alpha) * y0 + alpha * y1;

        return y;
    }

    int getDataSize() const { return dataAdaptor_.kdtree_get_point_count(); }

    void setBandwidths(const std::vector<double>& bandwidths) {
        bandwidths_ = bandwidths;
        init();
    }

   private:
    void init() {
        bandConstant_ = 1.0;
        // for (int i = 0; i < bandwidths_.size(); ++i) {
        //    bandConstant_ /=
        //        (std::pow(2 * PI, bandDimensions_.at(i) / 2.0) *
        //         std::pow(bandwidths_.at(i), bandDimensions_.at(i)));
        //}

        bandKernelLookupTable_.resize(bandwidths_.size());
        bandIncrements_.resize(bandwidths_.size());
        for (int i = 0; i < bandKernelLookupTable_.size(); ++i) {
            bandIncrements_.at(i) = CALCULATION_RANGE_ * CALCULATION_RANGE_ * bandwidths_.at(i) *
                                    bandwidths_.at(i) / LOOKUP_TABLE_SIZE_;
            bandKernelLookupTable_.at(i).resize(LOOKUP_TABLE_SIZE_);
            for (int j = 0; j < bandKernelLookupTable_.at(i).size(); ++j) {
                bandKernelLookupTable_.at(i).at(j) =
                        calculateKernel(j * bandIncrements_.at(i), bandwidths_.at(i));
            }
        }

        double maxBandwidth = *std::max_element(std::begin(bandwidths_), std::end(bandwidths_));
        radius_ = maxBandwidth * CALCULATION_RANGE_;
        radius_ *= radius_;  // FLANNのL2計算がルートを取らないため
                             // radiusを2乗して合わせる
    }

    double calculateKernel(double x, double bandwidth) const {
        return std::exp(-x / (2.0 * bandwidth * bandwidth));
    }
};

template <typename T, std::uint32_t DIM>
const double KernelDensityEstimation<T, DIM>::PI = 3.141592653589793238463;
}

#endif