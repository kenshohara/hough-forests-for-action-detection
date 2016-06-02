#ifndef STIP_FEATURE
#define STIP_FEATURE

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include <memory>
#include <vector>

namespace nuisken {
namespace storage {

/**
 * 時空間局所特徴のパッチ
 * Spatio temporal local feature
 */
class STIPFeature {
   private:
    /**
     * パッチ内の特徴
     */
    std::vector<Eigen::MatrixXf> featureVectors;

    /**
     * パッチの中心座標
     */
    cv::Vec3i centerPoint;

    /**
     * 重心へのベクトル
     */
    cv::Vec3i displacementVector;

    /**
     * 特徴点の空間的なスケール
     */
    double spatialScale;

    /**
     * 特徴点の時間的なスケール
     */
    double temporalScale;

    /**
     * クラスのラベル
     */
    int classLabel;

    int viewLabel;

    int index;

   public:
    STIPFeature(const std::vector<Eigen::MatrixXf>& featureVectors, const cv::Vec3i& centerPoint,
                const cv::Vec3i& displacementVector, const std::pair<double, double>& scales,
                int classLabel, int viewLabel = 0)
            : featureVectors(featureVectors),
              centerPoint(centerPoint),
              displacementVector(displacementVector),
              spatialScale(scales.first),
              temporalScale(scales.second),
              classLabel(classLabel),
              viewLabel(viewLabel){};

    double getFeatureValue(int index, int featureChannel) const {
        return featureVectors.at(featureChannel).coeff(0, index);
    }

    std::vector<Eigen::MatrixXf> getFeatureVectors() const {
        auto tempFeatureVectors = this->featureVectors;
        return tempFeatureVectors;
    }

    cv::Vec3i getCenterPoint() const { return centerPoint; }

    cv::Vec3i getDisplacementVector() const { return displacementVector; }

    double getSpatialScale() const { return spatialScale; }

    double getTemporalScale() const { return temporalScale; }

    int getClassLabel() const { return classLabel; }

    int getNumberOfFeatureChannels() const { return featureVectors.size(); }

    int getNumberOfFeatureDimensions(int featureChannel) const {
        return featureVectors.at(featureChannel).cols();
    }

    int getIndex() const { return index; }

    int getViewLabel() const { return viewLabel; }

    void setFeatureVectors(const std::vector<Eigen::MatrixXf>& featureVectors) {
        this->featureVectors = featureVectors;
    }

    void setCenterPoint(const cv::Vec3i& centerPoint) { this->centerPoint = centerPoint; }

    void setDisplacementVector(const cv::Vec3i& displacementVector) {
        this->displacementVector = displacementVector;
    }

    void setSpatialScale(double spatialScale) { this->spatialScale = spatialScale; }

    void setTemporalScale(double temporalScale) { this->temporalScale = temporalScale; }

    void setClassLabel(int classLabel) { this->classLabel = classLabel; }

    void setIndex(int index) { this->index = index; }

    // void save(const std::string& filePath) const;
    // void load(const std::string& filePath);
};
}
}

#endif