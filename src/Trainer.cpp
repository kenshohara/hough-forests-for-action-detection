#include "Trainer.h"
#include "LocalFeatureExtractor.h"

#include <numpy.hpp>

#include <boost/format.hpp>
#include <boost/tokenizer.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>

namespace nuisken {

void Trainer::extractTrainingFeatures(const std::string& positiveVideoDirectoryPath,
                                      const std::string& negativeVideoDirectoryPath,
                                      const std::string& labelFilePath,
                                      const std::string& dstDirectoryPath, int localWidth,
                                      int localHeight, int localDuration, int xBlockSize,
                                      int yBlockSize, int tBlockSize, int xStep, int yStep,
                                      int tStep, const std::vector<double>& negativeScales,
                                      int nPositiveSamplesPerStep,
                                      int nNegativeSamplesPerStep) const {
    extractPositiveFeatures(positiveVideoDirectoryPath, dstDirectoryPath, localWidth, localHeight,
                            localDuration, xBlockSize, yBlockSize, tBlockSize, xStep, yStep, tStep,
                            nPositiveSamplesPerStep);
	extractNegativeFeatures(negativeVideoDirectoryPath, labelFilePath, dstDirectoryPath,
							localWidth, localHeight, localDuration, xBlockSize, yBlockSize, tBlockSize,
							xStep, yStep, tStep, negativeScales, nNegativeSamplesPerStep);
}

void Trainer::extractPositiveFeatures(const std::string& videoDirectoryPath,
                                      const std::string& dstDirectoryPath, int localWidth,
                                      int localHeight, int localDuration, int xBlockSize,
                                      int yBlockSize, int tBlockSize, int xStep, int yStep,
                                      int tStep, int nSamplesPerStep) const {
    std::vector<double> scales = {1.0};
    int randomSeed = 1;
    std::mt19937 randomEngine(randomSeed);

    std::tr2::sys::path directory(videoDirectoryPath);
    std::tr2::sys::directory_iterator end;
    for (std::tr2::sys::directory_iterator itr(directory); itr != end; ++itr) {
        std::string filePath = itr->path().string();

        std::cout << "extract" << std::endl;
        houghforests::LocalFeatureExtractor extractor(filePath, scales, localWidth, localHeight,
                                                      localDuration, xBlockSize, yBlockSize,
                                                      tBlockSize, xStep, yStep, tStep);
        std::vector<cv::Vec3i> selectedPoints;
        std::vector<std::vector<float>> selectedDescriptors;
        while (true) {
            std::cout << "frame: " << extractor.getStoredFeatureBeginT() << std::endl;
            std::vector<std::vector<cv::Vec3i>> points;
            std::vector<std::vector<std::vector<float>>> descriptors;
            extractor.extractLocalFeatures(points, descriptors);
            if (extractor.isEnded()) {
                break;
            }

            std::vector<size_t> indices(points[0].size());
            std::iota(std::begin(indices), std::end(indices), 0);
            std::shuffle(std::begin(indices), std::end(indices), randomEngine);

            int n = 0;
            for (auto index : indices) {
                if (n++ >= nSamplesPerStep) {
                    break;
                }

                selectedPoints.push_back(points[0][index]);
                selectedDescriptors.push_back(descriptors[0][index]);
            }
        }

        std::cout << "output: " << selectedPoints.size() << std::endl;
        std::string dstFileName = itr->path().filename().stem().string();
        std::string dstPointsFilePath = dstDirectoryPath + dstFileName + "_pt.npy";
        std::vector<int> outputPoints;
        for (const auto& point : selectedPoints) {
            for (int i = 0; i < point.rows; ++i) {
                outputPoints.push_back(point(i));
            }
        }
        aoba::SaveArrayAsNumpy<int>(dstPointsFilePath, selectedPoints.size(),
                                    selectedPoints.front().rows, outputPoints.data());

        std::string dstDescriptorsFilePath = dstDirectoryPath + dstFileName + "_desc.npy";
        std::vector<float> outputDescriptors;
        for (const auto& desc : selectedDescriptors) {
            for (int i = 0; i < desc.size(); ++i) {
                outputDescriptors.push_back(desc[i]);
            }
        }
        aoba::SaveArrayAsNumpy<float>(dstDescriptorsFilePath, selectedDescriptors.size(),
                                      selectedDescriptors.front().size(), outputDescriptors.data());
    }
}

void Trainer::extractNegativeFeatures(const std::string& videoDirectoryPath,
                                      const std::string& labelFilePath,
                                      const std::string& dstDirectoryPath, int localWidth,
                                      int localHeight, int localDuration, int xBlockSize,
                                      int yBlockSize, int tBlockSize, int xStep, int yStep,
                                      int tStep, const std::vector<double>& scales,
                                      int nSamplesPerStep) const {
    int randomSeed = 1;
    std::mt19937 randomEngine(randomSeed);

    std::tr2::sys::path directory(videoDirectoryPath);
    std::tr2::sys::directory_iterator end;
    for (std::tr2::sys::directory_iterator itr(directory); itr != end; ++itr) {
        std::string filePath = itr->path().string();
        std::string fileName = itr->path().filename().string();
        int dataIndex = std::stoi(std::string{fileName[0]});

        std::vector<cv::Rect> boxes;
        std::vector<std::pair<int, int>> temporalRanges;
        std::cout << "read labels" << std::endl;
        readLabelsInfo(labelFilePath, dataIndex, std::vector<int>(), boxes, temporalRanges);

        std::cout << "extract" << std::endl;
        houghforests::LocalFeatureExtractor extractor(filePath, scales, localWidth, localHeight,
                                                      localDuration, xBlockSize, yBlockSize,
                                                      tBlockSize, xStep, yStep, tStep);
        std::vector<cv::Vec3i> selectedPoints;
        std::vector<std::vector<float>> selectedDescriptors;
        while (true) {
            std::cout << "frame: " << extractor.getStoredFeatureBeginT() << std::endl;
            std::vector<std::vector<cv::Vec3i>> points;
            std::vector<std::vector<std::vector<float>>> descriptors;
            extractor.extractLocalFeatures(points, descriptors);
            if (extractor.isEnded()) {
                break;
            }

            std::cout << "select" << std::endl;
            for (int scaleIndex = 0; scaleIndex < points.size(); ++scaleIndex) {
                std::vector<size_t> indices;
                int index = 0;
                for (const auto& point : points[scaleIndex]) {
                    cv::Vec3i scaledPoint(point);
                    scaledPoint(1) /= scales[scaleIndex];
                    scaledPoint(2) /= scales[scaleIndex];
                    if (!contains(boxes, temporalRanges, point)) {
                        indices.push_back(index);
                    }
                    ++index;
                }

                std::shuffle(std::begin(indices), std::end(indices), randomEngine);

                int n = 0;
                for (auto index : indices) {
                    if (n++ >= nSamplesPerStep) {
                        break;
                    }

                    selectedPoints.push_back(points[scaleIndex][index]);
                    selectedDescriptors.push_back(descriptors[scaleIndex][index]);
                }
            }
        }

        std::cout << "output: " << selectedPoints.size() << std::endl;
        std::string dstFileName = itr->path().filename().stem().string();
        std::string dstPointsFilePath = dstDirectoryPath + dstFileName + "_pt.npy";
        std::vector<int> outputPoints;
        for (const auto& point : selectedPoints) {
            for (int i = 0; i < point.rows; ++i) {
                outputPoints.push_back(point(i));
            }
        }
        aoba::SaveArrayAsNumpy<int>(dstPointsFilePath, selectedPoints.size(),
                                    selectedPoints.front().rows, outputPoints.data());

        std::string dstDescriptorsFilePath = dstDirectoryPath + dstFileName + "_desc.npy";
        std::vector<float> outputDescriptors;
        for (const auto& desc : selectedDescriptors) {
            for (int i = 0; i < desc.size(); ++i) {
                outputDescriptors.push_back(desc[i]);
            }
        }
        aoba::SaveArrayAsNumpy<float>(dstDescriptorsFilePath, selectedDescriptors.size(),
                                      selectedDescriptors.front().size(), outputDescriptors.data());
    }
}

void Trainer::readLabelsInfo(const std::string& labelFilePath, int dataIndex,
                             std::vector<int>& classLabels, std::vector<cv::Rect>& boxes,
                             std::vector<std::pair<int, int>>& temporalRanges) const {
    std::ifstream inputStream(labelFilePath);
    std::string line;
    while (std::getline(inputStream, line)) {
        boost::char_separator<char> commaSeparator(",");
        boost::tokenizer<boost::char_separator<char>> commaTokenizer(line, commaSeparator);
        std::vector<std::string> tokens;
        std::copy(std::begin(commaTokenizer), std::end(commaTokenizer), std::back_inserter(tokens));

        if (std::stoi(tokens.at(0)) == dataIndex) {
            auto label = std::stoi(tokens.at(1));
            classLabels.push_back(label);

            auto beginFrame = std::stoi(tokens.at(2));
            auto endFrame = std::stoi(tokens.at(3));
            auto topLeftX = std::stoi(tokens.at(4));
            auto topLeftY = std::stoi(tokens.at(5));
            auto bottomRightX = std::stoi(tokens.at(6));
            auto bottomRightY = std::stoi(tokens.at(7));
            boxes.emplace_back(cv::Point(topLeftX, topLeftY),
                               cv::Point(bottomRightX, bottomRightY));
            temporalRanges.emplace_back(beginFrame, endFrame);
        }
    }
}

bool Trainer::contains(const cv::Rect& box, const std::pair<int, int>& temporalRange,
                       const cv::Vec3i& point) const {
    bool space = box.contains(cv::Point(point(2), point(1)));
    bool time = (temporalRange.first <= point(0)) && (temporalRange.second < point(0));
    return space && time;
}

bool Trainer::contains(const std::vector<cv::Rect>& boxes,
                       const std::vector<std::pair<int, int>>& temporalRanges,
                       const cv::Vec3i& point) const {
    for (int i = 0; i < boxes.size(); ++i) {
        if (contains(boxes.at(i), temporalRanges.at(i), point)) {
            return true;
        }
    }
    return false;
}
}