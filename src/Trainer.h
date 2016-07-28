#ifndef TRAINER
#define TRAINER

#include "STIPFeature.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace nuisken {

class Trainer {
   private:
    using FeaturePtr = std::shared_ptr<storage::STIPFeature>;

   public:
    Trainer(){};
    ~Trainer(){};

    void extractTrainingFeatures(const std::string& positiveVideoDirectoryPath,
                                 const std::string& negativeVideoDirectoryPath,
                                 const std::string& labelFilePath,
                                 const std::string& dstDirectoryPath, int localWidth,
                                 int localHeight, int localDuration, int xBlockSize, int yBlockSize,
                                 int tBlockSize, int xStep, int yStep, int tStep,
                                 const std::vector<double>& negativeScales,
                                 int nPositiveSamplesPerStep, int nNegativeSamplesPerStep) const;

    void train(const std::string& featureDirectoryPath, const std::string& labelFilePath,
               const std::string& forestsDirectoryPath, const std::vector<int> trainingDataIndices,
               int nClasses, int baseScale, int nTrees, double bootstrapRatio, int maxDepth,
               int minData, int nSplits, int nThresholds);

   private:
    void extractPositiveFeatures(const std::string& videoDirectoryPath,
                                 const std::string& dstDirectoryPath, int localWidth,
                                 int localHeight, int localDuration, int xBlockSize, int yBlockSize,
                                 int tBlockSize, int xStep, int yStep, int tStep,
                                 int nSamplesPerStep) const;
    void extractNegativeFeatures(const std::string& videoDirectoryPath,
                                 const std::string& labelFilePath,
                                 const std::string& dstDirectoryPath, int localWidth,
                                 int localHeight, int localDuration, int xBlockSize, int yBlockSize,
                                 int tBlockSize, int xStep, int yStep, int tStep,
                                 const std::vector<double>& scales, int nSamplesPerStep) const;

    void readLabelsInfo(const std::string& labelFilePath, int dataIndex,
                        std::vector<int>& classLabels, std::vector<cv::Rect>& boxes,
                        std::vector<std::pair<int, int>>& temporalRanges) const;

    bool contains(const cv::Rect& box, const std::pair<int, int>& temporalRange,
                  const cv::Vec3i& point) const;
    bool contains(const std::vector<cv::Rect>& boxes,
                  const std::vector<std::pair<int, int>>& temporalRanges,
                  const cv::Vec3i& point) const;

    std::vector<FeaturePtr> readData(const std::string& directoryPath, int dataIndex,
                                     const std::vector<cv::Vec3i>& positiveActionPositions,
                                     int negativeLabel) const;
    std::vector<FeaturePtr> readPositiveData(const std::string& directoryPath, int dataIndex,
                                             int labelIndex, int classLabel,
                                             const cv::Vec3i& actionPosition) const;
    std::vector<FeaturePtr> readNegativeData(const std::string& directoryPath, int dataIndex,
                                             int negativeLabel) const;
    std::vector<FeaturePtr> readLocalFeatures(const std::string& pointFilePath,
                                              const std::string& descriptorFilePath, int classLabel,
                                              const cv::Vec3i& actionPosition) const;
    std::vector<FeaturePtr> readLocalFeatures(const std::string& pointFilePath,
                                              const std::string& descriptorFilePath,
                                              const std::string& foregroundFilePath, int classLabel,
                                              const cv::Vec3i& actionPosition) const;
};
}

#endif