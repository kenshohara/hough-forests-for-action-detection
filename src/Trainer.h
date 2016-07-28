#ifndef TRAINER
#define TRAINER

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace nuisken {

class Trainer {
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

    void outputTrainingData(const std::vector<std::string>&);

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
};
}

#endif