#include "Trainer.h"
#include "HoughForests.h"
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
    extractNegativeFeatures(negativeVideoDirectoryPath, labelFilePath, dstDirectoryPath, localWidth,
                            localHeight, localDuration, xBlockSize, yBlockSize, tBlockSize, xStep,
                            yStep, tStep, negativeScales, nNegativeSamplesPerStep);
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

void Trainer::train(const std::string& featureDirectoryPath, const std::string& labelFilePath,
                    const std::string& forestsDirectoryPath,
                    const std::vector<int> trainingDataIndices, int nClasses, int baseScale,
                    int nTrees, double bootstrapRatio, int maxDepth, int minData, int nSplits,
                    int nThresholds) {
    using namespace nuisken;
    using namespace nuisken::houghforests;
    using namespace nuisken::randomforests;
    using namespace nuisken::storage;

    const int N_CHANNELS = 4;

    std::vector<std::shared_ptr<STIPFeature>> trainingData;
    for (int dataIndex : trainingDataIndices) {
        std::cout << "read " << dataIndex << std::endl;

        std::vector<cv::Rect> boxes;
        std::vector<std::pair<int, int>> ranges;
        readLabelsInfo(labelFilePath, dataIndex, std::vector<int>(), boxes, ranges);

        std::vector<cv::Vec3i> positiveActionPositions(boxes.size());
        for (int labelIndex = 0; labelIndex < boxes.size(); ++labelIndex) {
            cv::Vec3i position;
            position(T) = (ranges.at(labelIndex).second - ranges.at(labelIndex).first) / 2;
            int width = boxes.at(labelIndex).width;
            int height = boxes.at(labelIndex).height;
            double aspectRatio = static_cast<double>(width) / height;
            position(Y) = baseScale / 2;
            position(X) = (baseScale * aspectRatio) / 2;
            positiveActionPositions.at(labelIndex) = position;
        }

        int negativeLabel = nClasses - 1;

        auto tmpData =
                readData(featureDirectoryPath, dataIndex, positiveActionPositions, negativeLabel);
        std::copy(std::begin(tmpData), std::end(tmpData), std::back_inserter(trainingData));
    }

    auto type = TreeParameters::ALL_RATIO;
    bool hasNegatieClass = true;
    TreeParameters treeParameters(nClasses, nTrees, bootstrapRatio, maxDepth, minData, nSplits,
                                  nThresholds, type, hasNegatieClass);
    std::vector<int> numberOfFeatureDimensions(N_CHANNELS);
    for (auto i = 0; i < N_CHANNELS; ++i) {
        numberOfFeatureDimensions.at(i) = trainingData.front()->getNumberOfFeatureDimensions(i);
    }
    STIPNode stipNode(nClasses, N_CHANNELS, numberOfFeatureDimensions);
    HoughForestsParameters houghParameters;
    houghParameters.setTreeParameters(treeParameters);
    int nThreads = 6;
    HoughForests houghForests(stipNode, houghParameters, nThreads);
    houghForests.train(trainingData);

    std::tr2::sys::path directory(forestsDirectoryPath);
    if (!std::tr2::sys::exists(directory)) {
        std::tr2::sys::create_directory(directory);
    }

    houghForests.save(forestsDirectoryPath);
}

std::vector<Trainer::FeaturePtr> Trainer::readData(
        const std::string& directoryPath, int dataIndex,
        const std::vector<cv::Vec3i>& positiveActionPositions, int negativeLabel) const {
    std::tr2::sys::path directory(directoryPath);
    std::tr2::sys::directory_iterator end;
    std::vector<int> usedLabelIndices;
    bool isNegativeRead = false;
    std::vector<FeaturePtr> trainingData;
    for (std::tr2::sys::directory_iterator itr(directory); itr != end; ++itr) {
        std::string filePath = itr->path().string();
        std::string fileName = itr->path().filename().string();

        boost::char_separator<char> separator("_");
        boost::tokenizer<boost::char_separator<char>> tokenizer(fileName, separator);
        std::vector<std::string> tokens;
        std::copy(std::begin(tokenizer), std::end(tokenizer), std::back_inserter(tokens));
        int fileDataIndex = std::stoi(tokens.at(0));
        if (fileDataIndex != dataIndex) {
            continue;
        }

        if (tokens.size() == 2) {
            if (isNegativeRead) {
                continue;
            }
            auto negatives = readNegativeData(directoryPath, dataIndex, negativeLabel);
            std::copy(std::begin(negatives), std::end(negatives), std::back_inserter(trainingData));

            isNegativeRead = true;
        } else if (tokens.size() == 4) {
            int labelIndex = std::stoi(tokens.at(1));
            if (std::find(std::begin(usedLabelIndices), std::end(usedLabelIndices), labelIndex) !=
                std::end(usedLabelIndices)) {
                continue;
            }

            int classLabel = std::stoi(tokens.at(2));
            auto positives = readPositiveData(directoryPath, dataIndex, labelIndex, classLabel,
                                              positiveActionPositions.at(labelIndex));
            std::copy(std::begin(positives), std::end(positives), std::back_inserter(trainingData));

            usedLabelIndices.push_back(labelIndex);
        }
    }

    return trainingData;
}

std::vector<Trainer::FeaturePtr> Trainer::readPositiveData(const std::string& directoryPath,
                                                           int dataIndex, int labelIndex,
                                                           int classLabel,
                                                           const cv::Vec3i& actionPosition) const {
    std::string pointFilePath = (boost::format("%s%d_%d_%d_pt.npy") % directoryPath % dataIndex %
                                 labelIndex % classLabel)
                                        .str();
    std::string descriptorFilePath = (boost::format("%s%d_%d_%d_desc.npy") % directoryPath %
                                      dataIndex % labelIndex % classLabel)
                                             .str();
    std::string foregroundFilePath = (boost::format("%s%d_%d_%d_fgd.npy") % directoryPath %
                                      dataIndex % labelIndex % classLabel)
                                             .str();
    auto tmpFeatures =
            readLocalFeatures(pointFilePath, descriptorFilePath, foregroundFilePath, classLabel, actionPosition);

    std::string flippedPointFilePath = (boost::format("%s%d_%d_%d_flip_pt.npy") % directoryPath %
                                        dataIndex % labelIndex % classLabel)
                                               .str();
    std::string flippedDescriptorFilePath = (boost::format("%s%d_%d_%d_flip_desc.npy") %
                                             directoryPath % dataIndex % labelIndex % classLabel)
                                                    .str();
    std::string flippedForegroundFilePath = (boost::format("%s%d_%d_%d_flip_fgd.npy") %
                                             directoryPath % dataIndex % labelIndex % classLabel)
                                                    .str();
    auto tmpFlippedFeatures = readLocalFeatures(flippedPointFilePath, flippedDescriptorFilePath,
                                                flippedForegroundFilePath, classLabel, actionPosition);

    std::vector<FeaturePtr> trainingData;
    std::copy(std::begin(tmpFeatures), std::end(tmpFeatures), std::back_inserter(trainingData));
    std::copy(std::begin(tmpFlippedFeatures), std::end(tmpFlippedFeatures),
              std::back_inserter(trainingData));
    return trainingData;
}

std::vector<Trainer::FeaturePtr> Trainer::readNegativeData(const std::string& directoryPath,
                                                           int dataIndex, int negativeLabel) const {
    std::string pointFilePath = (boost::format("%s%d_pt.npy") % directoryPath % dataIndex).str();
    std::string descriptorFilePath =
            (boost::format("%s%d_desc.npy") % directoryPath % dataIndex).str();
    return readLocalFeatures(pointFilePath, descriptorFilePath, negativeLabel, cv::Vec3i());
}

std::vector<Trainer::FeaturePtr> Trainer::readLocalFeatures(const std::string& pointFilePath,
                                                            const std::string& descriptorFilePath,
                                                            int classLabel,
                                                            const cv::Vec3i& actionPosition) const {
    using namespace storage;
    using namespace houghforests;

    const int N_CHANNELS = LocalFeatureExtractor::N_CHANNELS_;

    std::vector<int> pointShape;
    std::vector<int> points;
    aoba::LoadArrayFromNumpy<int>(pointFilePath, pointShape, points);

    std::vector<int> descShape;
    std::vector<float> descriptors;
    aoba::LoadArrayFromNumpy<float>(descriptorFilePath, descShape, descriptors);

    int nChannelFeatures = descShape[1] / N_CHANNELS;
    std::vector<FeaturePtr> localFeatures;
    for (int localIndex = 0; localIndex < pointShape[0]; ++localIndex) {
        int pointIndex = localIndex * 3;
        cv::Vec3i point(points[pointIndex], points[pointIndex + 1], points[pointIndex + 2]);
        std::vector<Eigen::MatrixXf> features(N_CHANNELS);
        for (int channelIndex = 0; channelIndex < N_CHANNELS; ++channelIndex) {
            Eigen::MatrixXf feature(1, nChannelFeatures);
            for (int featureIndex = 0; featureIndex < nChannelFeatures; ++featureIndex) {
                int index =
                        localIndex * descShape[1] + channelIndex * nChannelFeatures + featureIndex;
                feature.coeffRef(0, featureIndex) = descriptors[index];
            }
            features.at(channelIndex) = feature;
        }

        cv::Vec3i offset = actionPosition - point;
        auto data = std::make_shared<STIPFeature>(features, point, offset, std::make_pair(0.0, 0.0),
                                                  classLabel);
        data->setIndex(-1);
        localFeatures.push_back(data);
    }
    return localFeatures;
}

std::vector<Trainer::FeaturePtr> Trainer::readLocalFeatures(const std::string& pointFilePath,
                                                            const std::string& descriptorFilePath,
                                                            const std::string& foregroundFilePath,
                                                            int classLabel,
                                                            const cv::Vec3i& actionPosition) const {
    using namespace storage;
    using namespace houghforests;

    const int N_CHANNELS = LocalFeatureExtractor::N_CHANNELS_;

    std::vector<int> pointShape;
    std::vector<int> points;
    aoba::LoadArrayFromNumpy<int>(pointFilePath, pointShape, points);

    std::vector<int> descShape;
    std::vector<float> descriptors;
    aoba::LoadArrayFromNumpy<float>(descriptorFilePath, descShape, descriptors);

    std::vector<int> fgdShape;
    std::vector<unsigned char> foregrounds;
    aoba::LoadArrayFromNumpy<unsigned char>(foregroundFilePath, fgdShape, foregrounds);
    cv::Mat1b foregroundMat(fgdShape.size(), fgdShape.data());
    foregroundMat.data = foregrounds.data();

    int nChannelFeatures = descShape[1] / N_CHANNELS;
    std::vector<FeaturePtr> localFeatures;
    for (int localIndex = 0; localIndex < pointShape[0]; ++localIndex) {
        int pointIndex = localIndex * 3;
        cv::Vec3i point(points[pointIndex], points[pointIndex + 1], points[pointIndex + 2]);
        if (foregroundMat(point) == 0) {
            continue;
        }

        std::vector<Eigen::MatrixXf> features(N_CHANNELS);
        for (int channelIndex = 0; channelIndex < N_CHANNELS; ++channelIndex) {
            Eigen::MatrixXf feature(1, nChannelFeatures);
            for (int featureIndex = 0; featureIndex < nChannelFeatures; ++featureIndex) {
                int index =
                        localIndex * descShape[1] + channelIndex * nChannelFeatures + featureIndex;
                feature.coeffRef(0, featureIndex) = descriptors[index];
            }
            features.at(channelIndex) = feature;
        }

        cv::Vec3i offset = actionPosition - point;
        auto data = std::make_shared<STIPFeature>(features, point, offset, std::make_pair(0.0, 0.0),
                                                  classLabel);
        data->setIndex(-1);
        localFeatures.push_back(data);
    }
    return localFeatures;
}
}