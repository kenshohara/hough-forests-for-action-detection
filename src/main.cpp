#include "HoughForests.h"
#include "LocalFeatureExtractor.h"
#include "STIPFeature.h"
#include "Utils.h"

#include <omp.h>

#include <numpy.hpp>

#include <boost/format.hpp>

#include <Eigen/Core>

#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

void extractPositiveFeatures() {
    using namespace nuisken::houghforests;

    int localWidth = 21;
    int localHeight = localWidth;
    int localDuration = 9;
    int xBlockSize = 7;
    int yBlockSize = 7;
    int tBlockSize = 3;
    int xStep = 10;
    int yStep = xStep;
    int tStep = 5;
    std::vector<double> scales = {1.0};

    int nSamplesPerStep = 30;
    int randomSeed = 1;
    std::mt19937 randomEngine(randomSeed);

    // std::string rootDirectoryPath = "D:/UT-Interaction/";
    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    std::string videoDirectoryPath = rootDirectoryPath + "segmented_fixed_scale_100/";
    std::string outputDirectoryPath = rootDirectoryPath + "feature_hf_pooling_half/";
    std::tr2::sys::path directory(videoDirectoryPath);
    std::tr2::sys::directory_iterator end;
    for (std::tr2::sys::directory_iterator itr(directory); itr != end; ++itr) {
        std::string filePath = itr->path().string();

        std::cout << "extract" << std::endl;
        LocalFeatureExtractor extractor(filePath, scales, localWidth, localHeight, localDuration,
                                        xBlockSize, yBlockSize, tBlockSize, xStep, yStep, tStep);
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
        std::string outputFileName = itr->path().filename().stem().string();
        std::string outputPointsFilePath = outputDirectoryPath + outputFileName + "_pt.npy";
        std::vector<int> outputPoints;
        for (const auto& point : selectedPoints) {
            for (int i = 0; i < point.rows; ++i) {
                outputPoints.push_back(point(i));
            }
        }
        aoba::SaveArrayAsNumpy<int>(outputPointsFilePath, selectedPoints.size(),
                                    selectedPoints.front().rows, outputPoints.data());

        std::string outputDescriptorsFilePath = outputDirectoryPath + outputFileName + "_desc.npy";
        std::vector<float> outputDescriptors;
        for (const auto& desc : selectedDescriptors) {
            for (int i = 0; i < desc.size(); ++i) {
                outputDescriptors.push_back(desc[i]);
            }
        }
        aoba::SaveArrayAsNumpy<float>(outputDescriptorsFilePath, selectedDescriptors.size(),
                                      selectedDescriptors.front().size(), outputDescriptors.data());
    }
}

void readLabelsInfo(const std::string& labelFilePath, int sequenceIndex,
                    std::vector<int>& classLabels, std::vector<cv::Rect>& boxes,
                    std::vector<std::pair<int, int>>& temporalRanges) {
    std::ifstream inputStream(labelFilePath);
    std::string line;
    while (std::getline(inputStream, line)) {
        boost::char_separator<char> commaSeparator(",");
        boost::tokenizer<boost::char_separator<char>> commaTokenizer(line, commaSeparator);
        std::vector<std::string> tokens;
        std::copy(std::begin(commaTokenizer), std::end(commaTokenizer), std::back_inserter(tokens));

        if (tokens.at(0) == "seq" + std::to_string(sequenceIndex)) {
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

void readLabelsInfo(const std::string& labelFilePath, int sequenceIndex,
                    std::vector<cv::Rect>& boxes,
                    std::vector<std::pair<int, int>>& temporalRanges) {
    std::ifstream inputStream(labelFilePath);
    std::string line;
    while (std::getline(inputStream, line)) {
        boost::char_separator<char> commaSeparator(",");
        boost::tokenizer<boost::char_separator<char>> commaTokenizer(line, commaSeparator);
        std::vector<std::string> tokens;
        std::copy(std::begin(commaTokenizer), std::end(commaTokenizer), std::back_inserter(tokens));

        if (tokens.at(0) == "seq" + std::to_string(sequenceIndex)) {
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

bool contains(const cv::Rect& box, const std::pair<int, int>& temporalRange,
              const cv::Vec3i& point) {
    bool space = box.contains(cv::Point(point(2), point(1)));
    bool time = (temporalRange.first <= point(0)) && (temporalRange.second < point(0));
    return space && time;
}

bool contains(const std::vector<cv::Rect>& boxes,
              const std::vector<std::pair<int, int>>& temporalRanges, const cv::Vec3i& point) {
    for (int i = 0; i < boxes.size(); ++i) {
        if (contains(boxes.at(i), temporalRanges.at(i), point)) {
            return true;
        }
    }
    return false;
}

void extractNegativeFeatures() {
    using namespace nuisken::houghforests;

    int localWidth = 21;
    int localHeight = localWidth;
    int localDuration = 9;
    int xBlockSize = 7;
    int yBlockSize = 7;
    int tBlockSize = 3;
    int xStep = 10;
    int yStep = xStep;
    int tStep = 5;
    std::vector<double> scales = {1.0, 0.707, 0.5};

    int nSamplesPerStep = 3;
    int randomSeed = 1;
    std::mt19937 randomEngine(randomSeed);

    std::vector<std::string> filePaths;
    std::string videoDirectoryPath = "E:/Hara/UT-Interaction/unsegmented_half/";
    std::string labelFilePath = "E:/Hara/UT-Interaction/labels.csv";
    std::string outputDirectoryPath = "E:/Hara/UT-Interaction/feature_hf_pooling_half/";
    for (int sequenceIndex = 1; sequenceIndex <= 5; ++sequenceIndex) {
        std::string filePath =
                (boost::format("%sseq%d.avi") % videoDirectoryPath % sequenceIndex).str();
        std::vector<cv::Rect> boxes;
        std::vector<std::pair<int, int>> temporalRanges;
        std::cout << "read labels" << std::endl;
        readLabelsInfo(labelFilePath, sequenceIndex, boxes, temporalRanges);

        std::cout << "extract" << std::endl;
        LocalFeatureExtractor extractor(filePath, scales, localWidth, localHeight, localDuration,
                                        xBlockSize, yBlockSize, tBlockSize, xStep, yStep, tStep);
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
        std::string outputFileName = (boost::format("seq%d") % sequenceIndex).str();
        std::string outputPointsFilePath = outputDirectoryPath + outputFileName + "_pt.npy";
        std::vector<int> outputPoints;
        for (const auto& point : selectedPoints) {
            for (int i = 0; i < point.rows; ++i) {
                outputPoints.push_back(point(i));
            }
        }
        aoba::SaveArrayAsNumpy<int>(outputPointsFilePath, selectedPoints.size(),
                                    selectedPoints.front().rows, outputPoints.data());

        std::string outputDescriptorsFilePath = outputDirectoryPath + outputFileName + "_desc.npy";
        std::vector<float> outputDescriptors;
        for (const auto& desc : selectedDescriptors) {
            for (int i = 0; i < desc.size(); ++i) {
                outputDescriptors.push_back(desc[i]);
            }
        }
        aoba::SaveArrayAsNumpy<float>(outputDescriptorsFilePath, selectedDescriptors.size(),
                                      selectedDescriptors.front().size(), outputDescriptors.data());
    }
}

void train() {
    using namespace nuisken::storage;
    using namespace nuisken::houghforests;
    using namespace nuisken::randomforests;

    const int N_CHANNELS = 4;
    const int N_CLASSES = 7;

    int baseScale = 100;

    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    // std::string rootDirectoryPath = "D:/UT-Interaction/";
    std::string featureDirectoryPath = rootDirectoryPath + "feature_hf_pooling_half/";
    std::string labelFilePath = rootDirectoryPath + "labels_half.csv";
    std::string forestsDirectoryPath = rootDirectoryPath + "data_hf/forests_hf_pooling_half/";
    std::vector<std::vector<int>> validationCombinations = {{19, 5}, {12, 6}, {4, 7},   {16, 17},
                                                            {9, 13}, {11, 8}, {10, 14}, {18, 15},
                                                            {3, 20}, {2, 1}};

    for (int validationIndex = 5; validationIndex < 10; ++validationIndex) {
        std::cout << "validation: " << validationIndex << std::endl;
        std::vector<std::shared_ptr<STIPFeature>> trainingData;
        for (int i = 0; i < validationCombinations.size(); ++i) {
            if (i == validationIndex) {
                continue;
            }
            for (int sequenceIndex : validationCombinations.at(i)) {
                std::cout << "read seq" << sequenceIndex << std::endl;
                std::vector<int> classLabels;
                std::vector<cv::Rect> boxes;
                std::vector<std::pair<int, int>> ranges;
                readLabelsInfo(labelFilePath, sequenceIndex, classLabels, boxes, ranges);
                for (int labelIndex = 0; labelIndex < classLabels.size(); ++labelIndex) {
                    std::string baseFilePath =
                            (boost::format("%sseq%d_%d_%d") % featureDirectoryPath % sequenceIndex %
                             labelIndex % classLabels[labelIndex])
                                    .str();
                    std::string pointFilePath = baseFilePath + "_pt.npy";
                    std::vector<int> pointShape;
                    std::vector<int> points;
                    aoba::LoadArrayFromNumpy<int>(pointFilePath, pointShape, points);

                    std::string descriptorFilePath = baseFilePath + "_desc.npy";
                    std::vector<int> descShape;
                    std::vector<float> descriptors;
                    aoba::LoadArrayFromNumpy<float>(descriptorFilePath, descShape, descriptors);

                    int nChannelFeatures = descShape[1] / N_CHANNELS;
                    for (int localIndex = 0; localIndex < pointShape[0]; ++localIndex) {
                        int pointIndex = localIndex * 3;
                        cv::Vec3i point(points[pointIndex], points[pointIndex + 1],
                                        points[pointIndex + 2]);
                        std::vector<Eigen::MatrixXf> features(N_CHANNELS);
                        for (int channelIndex = 0; channelIndex < N_CHANNELS; ++channelIndex) {
                            Eigen::MatrixXf feature(1, nChannelFeatures);
                            for (int featureIndex = 0; featureIndex < nChannelFeatures;
                                 ++featureIndex) {
                                int index = localIndex * descShape[1] +
                                            channelIndex * nChannelFeatures + featureIndex;
                                feature.coeffRef(0, featureIndex) = descriptors[index];
                            }
                            features.at(channelIndex) = feature;
                        }
                        cv::Vec3i actionPosition;
                        actionPosition(0) =
                                (ranges[labelIndex].second - ranges[labelIndex].first) / 2;
                        double aspectRatio = static_cast<double>(boxes[labelIndex].width) /
                                             boxes[labelIndex].height;
                        actionPosition(1) = baseScale / 2;
                        actionPosition(2) = baseScale * aspectRatio / 2;
                        // std::cout << actionPosition << std::endl;
                        cv::Vec3i offset = actionPosition - point;
                        auto data = std::make_shared<STIPFeature>(features, point, offset,
                                                                  std::make_pair(0.0, 0.0),
                                                                  classLabels[labelIndex]);
                        data->setIndex(-1);
                        trainingData.push_back(data);
                    }
                }

                std::string baseFilePath =
                        (boost::format("%sseq%d") % featureDirectoryPath % sequenceIndex).str();
                std::string pointFilePath = baseFilePath + "_pt.npy";
                std::vector<int> pointShape;
                std::vector<int> points;
                aoba::LoadArrayFromNumpy<int>(pointFilePath, pointShape, points);

                std::string descriptorFilePath = baseFilePath + "_desc.npy";
                std::vector<int> descShape;
                std::vector<float> descriptors;
                aoba::LoadArrayFromNumpy<float>(descriptorFilePath, descShape, descriptors);

                int nChannelFeatures = descShape[0] / N_CHANNELS;

                for (int localIndex = 0; localIndex < pointShape[1]; ++localIndex) {
                    int pointIndex = localIndex * 3;
                    cv::Vec3i point(points[pointIndex], points[pointIndex + 1],
                                    points[pointIndex + 2]);
                    std::vector<Eigen::MatrixXf> features(N_CHANNELS);
                    for (int channelIndex = 0; channelIndex < N_CHANNELS; ++channelIndex) {
                        Eigen::MatrixXf feature(1, nChannelFeatures);
                        for (int featureIndex = 0; featureIndex < nChannelFeatures;
                             ++featureIndex) {
                            int index = localIndex * descShape[0] +
                                        channelIndex * nChannelFeatures + featureIndex;
                            feature.coeffRef(0, featureIndex) = descriptors[index];
                        }
                        features.at(channelIndex) = feature;
                    }
                    auto data = std::make_shared<STIPFeature>(
                            features, point, cv::Vec3i(), std::make_pair(0.0, 0.0), N_CLASSES - 1);
                    data->setIndex(-1);
                    trainingData.push_back(data);
                }
            }
        }
        std::cout << "data size: " << trainingData.size() << std::endl;

        int nTrees = 15;
        double bootstrapRatio = 1.0;
        int maxDepth = 30;
        int minData = 5;
        int nSplits = 30;
        int nThresholds = 10;
        auto type = TreeParameters::ALL_RATIO;
        bool hasNegatieClass = true;
        TreeParameters treeParameters(N_CLASSES, nTrees, bootstrapRatio, maxDepth, minData, nSplits,
                                      nThresholds, type, hasNegatieClass);
        std::vector<int> numberOfFeatureDimensions(N_CHANNELS);
        for (auto i = 0; i < N_CHANNELS; ++i) {
            numberOfFeatureDimensions.at(i) = trainingData.front()->getNumberOfFeatureDimensions(i);
        }
        STIPNode stipNode(N_CLASSES, N_CHANNELS, numberOfFeatureDimensions);
        HoughForestsParameters houghParameters;
        houghParameters.setTreeParameters(treeParameters);
        int nThreads = 6;
        HoughForests houghForests(stipNode, houghParameters, nThreads);
        houghForests.train(trainingData);

        std::string outputDirectoryPath =
                (boost::format("%s%d/") % forestsDirectoryPath % validationIndex).str();
        std::tr2::sys::path directory(outputDirectoryPath);
        if (!std::tr2::sys::exists(directory)) {
            std::tr2::sys::create_directory(directory);
        }

        houghForests.save(outputDirectoryPath);
    }
}

void train1data() {
    using namespace nuisken::storage;
    using namespace nuisken::houghforests;
    using namespace nuisken::randomforests;

    const int N_CHANNELS = 4;
    const int N_USED_CHANNELS = 4;
    const int N_CLASSES = 7;

    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    // std::string rootDirectoryPath = "D:/UT-Interaction/";
    std::string featureDirectoryPath = rootDirectoryPath + "feature_hf2/";
    std::string labelFilePath = rootDirectoryPath + "labels.csv";
    std::string forestsDirectoryPath = rootDirectoryPath + "data_hf/forests_data1_7/";
    std::vector<std::vector<int>> validationCombinations = {{19, 5}, {12, 6}, {4, 7},   {16, 17},
                                                            {9, 13}, {11, 8}, {10, 14}, {18, 15},
                                                            {3, 20}, {2, 1}};

    for (int validationIndex = 0; validationIndex < 1; ++validationIndex) {
        std::cout << "validation: " << validationIndex << std::endl;
        std::vector<std::shared_ptr<STIPFeature>> trainingData;
        for (int i = 0; i < validationCombinations.size(); ++i) {
            if (i == validationIndex) {
                continue;
            }
            std::cout << i << std::endl;
            for (int sequenceIndex : validationCombinations.at(i)) {
                if (sequenceIndex != 1) {
                    continue;
                }
                std::cout << "read seq" << sequenceIndex << std::endl;
                std::vector<int> classLabels;
                std::vector<cv::Rect> boxes;
                std::vector<std::pair<int, int>> ranges;
                readLabelsInfo(labelFilePath, sequenceIndex, classLabels, boxes, ranges);
                for (int labelIndex = 0; labelIndex < classLabels.size(); ++labelIndex) {
                    std::string baseFilePath =
                            (boost::format("%sseq%d_%d_%d") % featureDirectoryPath % sequenceIndex %
                             labelIndex % classLabels[labelIndex])
                                    .str();
                    std::string pointFilePath = baseFilePath + "_pt.npy";
                    std::vector<int> pointShape;
                    std::vector<int> points;
                    aoba::LoadArrayFromNumpy<int>(pointFilePath, pointShape, points);

                    std::string descriptorFilePath = baseFilePath + "_desc.npy";
                    std::vector<int> descShape;
                    std::vector<float> descriptors;
                    aoba::LoadArrayFromNumpy<float>(descriptorFilePath, descShape, descriptors);

                    int nChannelFeatures = descShape[0] / N_CHANNELS;

                    for (int localIndex = 0; localIndex < pointShape[1]; ++localIndex) {
                        int pointIndex = localIndex * 3;
                        cv::Vec3i point(points[pointIndex], points[pointIndex + 1],
                                        points[pointIndex + 2]);
                        std::vector<Eigen::MatrixXf> features(N_USED_CHANNELS);
                        for (int channelIndex = 0; channelIndex < N_USED_CHANNELS; ++channelIndex) {
                            Eigen::MatrixXf feature(1, nChannelFeatures);
                            for (int featureIndex = 0; featureIndex < nChannelFeatures;
                                 ++featureIndex) {
                                int index = localIndex * descShape[0] +
                                            channelIndex * nChannelFeatures + featureIndex;
                                feature.coeffRef(0, featureIndex) = descriptors[index];
                            }
                            features.at(channelIndex) = feature;
                        }
                        cv::Vec3i actionPosition;
                        actionPosition(0) =
                                (ranges[labelIndex].second - ranges[labelIndex].first) / 2;
                        actionPosition(1) = boxes[labelIndex].height / 2;
                        actionPosition(2) = boxes[labelIndex].width / 2;
                        cv::Vec3i offset = actionPosition - point;
                        auto data = std::make_shared<STIPFeature>(features, point, offset,
                                                                  std::make_pair(0.0, 0.0),
                                                                  classLabels[labelIndex]);
                        trainingData.push_back(data);
                    }
                }
            }
        }

        int nTrees = 15;
        double bootstrapRatio = 1.0;
        int maxDepth = 80;
        int minData = 5;
        int nSplits = 100;
        int nThresholds = 10;
        auto type = TreeParameters::ALL_RATIO;
        bool hasNegatieClass = true;
        TreeParameters treeParameters(N_CLASSES, nTrees, bootstrapRatio, maxDepth, minData, nSplits,
                                      nThresholds, type, hasNegatieClass);
        std::vector<int> numberOfFeatureDimensions(N_USED_CHANNELS);
        for (auto i = 0; i < N_USED_CHANNELS; ++i) {
            numberOfFeatureDimensions.at(i) = trainingData.front()->getNumberOfFeatureDimensions(i);
        }
        STIPNode stipNode(N_CLASSES, N_USED_CHANNELS, numberOfFeatureDimensions);
        HoughForestsParameters houghParameters;
        houghParameters.setTreeParameters(treeParameters);
        int nThreads = 6;
        HoughForests houghForests(stipNode, houghParameters, nThreads);
        houghForests.train(trainingData);

        std::string outputDirectoryPath =
                (boost::format("%s%d/") % forestsDirectoryPath % validationIndex).str();
        std::tr2::sys::path directory(outputDirectoryPath);
        if (!std::tr2::sys::exists(directory)) {
            std::tr2::sys::create_directory(directory);
        }

        houghForests.save(outputDirectoryPath);
    }
}

void detect() {
    using namespace nuisken;
    using namespace nuisken::houghforests;
    using namespace nuisken::randomforests;
    using namespace nuisken::storage;

    // std::string rootDirectoryPath = "D:/UT-Interaction/";
    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    std::string forestsDirectoryPath = rootDirectoryPath + "data_hf/forests_hf_pooling_half/2/";
    std::string outputDirectoryPath = rootDirectoryPath + "data_hf/voting/";

    int localWidth = 21;
    int localHeight = localWidth;
    int localDuration = 9;
    int xBlockSize = 7;
    int yBlockSize = 7;
    int tBlockSize = 3;
    int xStep = 10;
    int yStep = xStep;
    int tStep = 5;
    std::vector<double> scales = {1.0, 0.707, 0.5};
    // std::vector<double> scales = {1.0};
    // std::vector<double> scales = {0.707};

    std::string videoFilePath = rootDirectoryPath + "unsegmented_half/seq4.avi";
    LocalFeatureExtractor extractor(videoFilePath, scales, localWidth, localHeight, localDuration,
                                    xBlockSize, yBlockSize, tBlockSize, xStep, yStep, tStep);

    int nClasses = 7;
    int nThreads = 6;
    int width = 360;
    int height = 240;
    // int width = 720;
    // int height = 480;
    int baseScale = 100;
    std::vector<double> bandwidths = {10.0, 8.0, 0.5};
    std::vector<int> binSizes = {10, 20, 20};
    std::vector<int> steps = {20, 10};
    int votesDeleteStep = 50;
    int votesBufferLength = 200;
    double votingSpaceDiscretizeRatio = 0.5;
    // std::vector<double> scoreThresholds = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> scoreThresholds(6, 2.0);
    std::vector<double> aspectRatios = {1.23, 1.22, 1.42, 0.69, 1.46, 1.72};
    std::vector<std::size_t> durations = {100, 116, 66, 83, 62, 85};
    double iouThreshold = 0.1;
    // std::vector<double> scoreThresholds(6, 0.05);
    bool hasNegativeClass = true;
    bool isBackprojection = false;
    TreeParameters treeParameters(nClasses, 0, 0, 0, 0, 0, 0, TreeParameters::ALL_RATIO,
                                  hasNegativeClass);
    HoughForestsParameters parameters(width, height, scales, baseScale, nClasses, bandwidths.at(0),
                                      bandwidths.at(1), bandwidths.at(2), steps.at(0), steps.at(1),
                                      binSizes, votesDeleteStep, votesBufferLength, scoreThresholds,
                                      durations, aspectRatios, iouThreshold, hasNegativeClass,
                                      isBackprojection, treeParameters);
    HoughForests houghForests(nThreads);
    houghForests.setHoughForestsParameters(parameters);
    houghForests.load(forestsDirectoryPath);

    std::vector<std::vector<DetectionResult<4>>> detectionResults;
    houghForests.detect(extractor, detectionResults);

    // std::cout << "output" << std::endl;
    // for (auto classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
    //    std::string outputFilePath =
    //            outputDirectoryPath + std::to_string(classLabel) + "_detection.txt";

    //    std::ofstream outputStream(outputFilePath);
    //    for (const auto& detectionResult : detectionResults.at(classLabel)) {
    //        LocalMaximum localMaximum = detectionResult.getLocalMaximum();
    //        outputStream << "LocalMaximum," << localMaximum.getPoint()(T) << ","
    //                     << localMaximum.getPoint()(Y) << "," << localMaximum.getPoint()(X) << ","
    //                     << localMaximum.getValue() << "," << localMaximum.getPoint()(3)
    //                     << std::endl;

    //        auto contributionPoints = detectionResult.getContributionPoints();
    //        for (const auto& contributionPoint : contributionPoints) {
    //            outputStream << contributionPoint.getPoint()(T) << ","
    //                         << contributionPoint.getPoint()(Y) << ","
    //                         << contributionPoint.getPoint()(X) << ","
    //                         << contributionPoint.getValue() << std::endl;
    //        }
    //        outputStream << std::endl;
    //    }
    //}
}

std::vector<std::size_t> readDurations(const std::string& filePath) {
    std::ifstream inputStream(filePath);
    std::string line;

    std::vector<std::size_t> durations;
    while (std::getline(inputStream, line)) {
        boost::char_separator<char> commaSeparator(",");
        boost::tokenizer<boost::char_separator<char>> commaTokenizer(line, commaSeparator);
        std::vector<std::string> tokens;
        std::copy(std::begin(commaTokenizer), std::end(commaTokenizer), std::back_inserter(tokens));

        durations.push_back(std::stoi(tokens.at(1)));
    }
    return durations;
}

std::vector<double> readAspectRatios(const std::string& filePath) {
    std::ifstream inputStream(filePath);
    std::string line;

    std::vector<double> aspectRatios;
    while (std::getline(inputStream, line)) {
        boost::char_separator<char> commaSeparator(",");
        boost::tokenizer<boost::char_separator<char>> commaTokenizer(line, commaSeparator);
        std::vector<std::string> tokens;
        std::copy(std::begin(commaTokenizer), std::end(commaTokenizer), std::back_inserter(tokens));

        aspectRatios.push_back(std::stod(tokens.at(1)));
    }
    return aspectRatios;
}

void detectAll() {
    using namespace nuisken;
    using namespace nuisken::houghforests;
    using namespace nuisken::randomforests;
    using namespace nuisken::storage;

    // std::string rootDirectoryPath = "D:/UT-Interaction/";
    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    std::string forestsDirectoryPath = rootDirectoryPath + "data_hf/forests_hf_pooling_half/";
    std::string outputDirectoryPath = rootDirectoryPath + "data_hf/voting_hist_bin_20_10_fix_old/";
    std::string durationDirectoryPath = rootDirectoryPath + "average_durations/";
    std::string aspectDirectoryPath = rootDirectoryPath + "average_aspect_ratios/";

    int localWidth = 21;
    int localHeight = localWidth;
    int localDuration = 9;
    int xBlockSize = 7;
    int yBlockSize = 7;
    int tBlockSize = 3;
    int xStep = 10;
    int yStep = xStep;
    int tStep = 5;
    // std::vector<double> scales = {1.0, 0.707, 0.5};
    // std::vector<double> scales = {1.0};
    std::vector<double> scales = {1.0, 0.707, 0.5};

    int nClasses = 7;
    int nThreads = 6;
    int width = 360;
    int height = 240;
    // int width = 720;
    // int height = 480;
    int baseScale = 100;
    std::vector<double> bandwidths = {10.0, 8.0, 0.5};
    std::vector<int> steps = {20, 10};
    // std::vector<int> steps = {10, 5};
    // std::vector<int> steps = {5, 1};
    std::vector<int> binSizes = {10, 20, 20};
    // std::vector<int> binSizes = {5, 10, 10};
    // std::vector<int> binSizes = {2, 5, 5};
    int votesDeleteStep = 50;
    int votesBufferLength = 200;
    double votingSpaceDiscretizeRatio = 1.0;
    // std::vector<double> scoreThresholds = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> scoreThresholds(6, 0.05);
    double iouThreshold = 0.1;
    // std::vector<double> scoreThresholds(6, 0.05);
    bool hasNegativeClass = true;
    bool isBackprojection = false;
    TreeParameters treeParameters(nClasses, 0, 0, 0, 0, 0, 0, TreeParameters::ALL_RATIO,
                                  hasNegativeClass);

    std::vector<std::vector<int>> validationCombinations = {{19, 5}, {12, 6}, {4, 7},   {16, 17},
                                                            {9, 13}, {11, 8}, {10, 14}, {18, 15},
                                                            {3, 20}, {2, 1}};

    for (int validationIndex = 0; validationIndex < 10; ++validationIndex) {
        std::vector<double> aspectRatios =
                readAspectRatios(aspectDirectoryPath + std::to_string(validationIndex) + ".csv");
        std::vector<std::size_t> durations =
                readDurations(durationDirectoryPath + std::to_string(validationIndex) + ".csv");
        HoughForestsParameters parameters(
                width, height, scales, baseScale, nClasses, bandwidths.at(0), bandwidths.at(1),
                bandwidths.at(2), steps.at(0), steps.at(1), binSizes, votesDeleteStep,
                votesBufferLength, scoreThresholds, durations, aspectRatios, iouThreshold,
                hasNegativeClass, isBackprojection, treeParameters);

        std::cout << "validation: " << validationIndex << std::endl;
        HoughForests houghForests(nThreads);
        houghForests.setHoughForestsParameters(parameters);
        std::string forestsDir = forestsDirectoryPath + std::to_string(validationIndex) + "/";
        houghForests.load(forestsDir);
        for (int sequenceIndex : validationCombinations.at(validationIndex)) {
            std::string videoFilePath = (boost::format("%s/unsegmented_half/seq%d.avi") %
                                         rootDirectoryPath % sequenceIndex)
                                                .str();
            LocalFeatureExtractor extractor(videoFilePath, scales, localWidth, localHeight,
                                            localDuration, xBlockSize, yBlockSize, tBlockSize,
                                            xStep, yStep, tStep);

            std::vector<std::vector<DetectionResult<4>>> detectionResults;
            houghForests.detect(extractor, detectionResults);

            std::cout << "output" << std::endl;
            for (auto classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
                std::string outputFilePath = (boost::format("%s%d_%d_detection.txt") %
                                              outputDirectoryPath % sequenceIndex % classLabel)
                                                     .str();

                std::ofstream outputStream(outputFilePath);
                for (const auto& detectionResult : detectionResults.at(classLabel)) {
                    LocalMaximum localMaximum = detectionResult.getLocalMaximum();
                    outputStream << "LocalMaximum," << localMaximum.getPoint()(T) << ","
                                 << localMaximum.getPoint()(Y) << "," << localMaximum.getPoint()(X)
                                 << "," << localMaximum.getValue() << ","
                                 << localMaximum.getPoint()(3) << std::endl;

                    auto contributionPoints = detectionResult.getContributionPoints();
                    for (const auto& contributionPoint : contributionPoints) {
                        outputStream << contributionPoint.getPoint()(T) << ","
                                     << contributionPoint.getPoint()(Y) << ","
                                     << contributionPoint.getPoint()(X) << ","
                                     << contributionPoint.getValue() << std::endl;
                    }
                    outputStream << std::endl;
                }
            }
        }
    }
}

void detectAllSTIP() {
    using namespace nuisken;
    using namespace nuisken::houghforests;
    using namespace nuisken::randomforests;
    using namespace nuisken::storage;

    // std::string rootDirectoryPath = "D:/UT-Interaction/";
    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    std::string forestsDirectoryPath =
            rootDirectoryPath + "data_fixed_scale/forests_first_layer_allratio/";
    std::string outputDirectoryPath = rootDirectoryPath + "data_hf/voting_stip/";
    std::string durationDirectoryPath = rootDirectoryPath + "average_durations/";
    std::string aspectDirectoryPath = rootDirectoryPath + "average_aspect_ratios/";

    std::string featureDirectoryPath = rootDirectoryPath + "feature_stip2_multiscale_first_layer/";

    // std::vector<double> scales = {1.0, 0.707, 0.5};
    // std::vector<double> scales = {1.0};
    std::vector<double> scales = {1.0, 0.707, 0.5};

    int nClasses = 7;
    int nThreads = 1;
    int width = 360;
    int height = 240;
    // int width = 720;
    // int height = 480;
    int baseScale = 100;
    std::vector<double> bandwidths = {10.0, 8.0, 0.5};
    std::vector<int> binSizes = {5, 10, 10};
    std::vector<int> steps = {20, 10};
    int votesDeleteStep = 50;
    int votesBufferLength = 200;
    double votingSpaceDiscretizeRatio = 0.5;
    // std::vector<double> scoreThresholds = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> scoreThresholds(6, 0.1);
    double iouThreshold = 0.1;
    // std::vector<double> scoreThresholds(6, 0.05);
    bool hasNegativeClass = true;
    bool isBackprojection = false;
    TreeParameters treeParameters(nClasses, 0, 0, 0, 0, 0, 0, TreeParameters::ALL_RATIO,
                                  hasNegativeClass);

    std::vector<std::vector<int>> validationCombinations = {{19, 5}, {12, 6}, {4, 7},   {16, 17},
                                                            {9, 13}, {11, 8}, {10, 14}, {18, 15},
                                                            {3, 20}, {2, 1}};

    for (int validationIndex = 0; validationIndex < 5; ++validationIndex) {
        std::vector<double> aspectRatios =
                readAspectRatios(aspectDirectoryPath + std::to_string(validationIndex) + ".csv");
        std::vector<std::size_t> durations =
                readDurations(durationDirectoryPath + std::to_string(validationIndex) + ".csv");
        HoughForestsParameters parameters(
                width, height, scales, baseScale, nClasses, bandwidths.at(0), bandwidths.at(1),
                bandwidths.at(2), steps.at(0), steps.at(1), binSizes, votesDeleteStep,
                votesBufferLength, scoreThresholds, durations, aspectRatios, iouThreshold,
                hasNegativeClass, isBackprojection, treeParameters);

        std::cout << "validation: " << validationIndex << std::endl;
        HoughForests houghForests(nThreads);
        houghForests.setHoughForestsParameters(parameters);
        std::string forestsDir = forestsDirectoryPath + std::to_string(validationIndex) + "/";
        houghForests.load(forestsDir);
        for (int sequenceIndex : validationCombinations.at(validationIndex)) {
            std::vector<std::string> featureFilePaths;
            for (int s = 0; s < 3; ++s) {
                featureFilePaths.push_back(
                        (boost::format("%sseq%d_s0.txt") % featureDirectoryPath % sequenceIndex)
                                .str());
            }

            std::vector<std::vector<DetectionResult<4>>> detectionResults;
            houghForests.detect(featureFilePaths, detectionResults);

            std::cout << "output" << std::endl;
            for (auto classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
                std::string outputFilePath = (boost::format("%s%d_%d_detection.txt") %
                                              outputDirectoryPath % sequenceIndex % classLabel)
                                                     .str();

                std::ofstream outputStream(outputFilePath);
                for (const auto& detectionResult : detectionResults.at(classLabel)) {
                    LocalMaximum localMaximum = detectionResult.getLocalMaximum();
                    outputStream << "LocalMaximum," << localMaximum.getPoint()(T) << ","
                                 << localMaximum.getPoint()(Y) << "," << localMaximum.getPoint()(X)
                                 << "," << localMaximum.getValue() << ","
                                 << localMaximum.getPoint()(3) << std::endl;

                    auto contributionPoints = detectionResult.getContributionPoints();
                    for (const auto& contributionPoint : contributionPoints) {
                        outputStream << contributionPoint.getPoint()(T) << ","
                                     << contributionPoint.getPoint()(Y) << ","
                                     << contributionPoint.getPoint()(X) << ","
                                     << contributionPoint.getValue() << std::endl;
                    }
                    outputStream << std::endl;
                }
            }
        }
    }
}

void classify() {
    // using namespace nuisken;
    // using namespace nuisken::houghforests;
    // using namespace nuisken::randomforests;
    // using namespace nuisken::storage;

    // std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    // std::string forestsDirectoryPath = rootDirectoryPath + "data_hf/forests_data1_7/0/";
    // std::string outputDirectoryPath = rootDirectoryPath + "data_hf/voting/";

    // int localWidth = 21;
    // int localHeight = localWidth;
    // int localDuration = 9;
    // int xBlockSize = 7;
    // int yBlockSize = 7;
    // int tBlockSize = 3;
    // int xStep = 10;
    // int yStep = xStep;
    // int tStep = 5;
    //// std::vector<double> scales = {1.0, 0.707, 0.5};
    // std::vector<double> scales = {1.0};

    //// std::string videoFilePath = rootDirectoryPath + "test.avi";  // "unsegmented/seq5.avi";
    // std::string videoFilePath = rootDirectoryPath + "segmented_fixed_scale/seq5_0_0.avi";
    // LocalFeatureExtractor extractor(videoFilePath, scales, localWidth, localHeight,
    // localDuration,
    //                                xBlockSize, yBlockSize, tBlockSize, xStep, yStep, tStep);

    // int nClasses = 7;
    // int nThreads = 1;
    // int width = 300;
    // int height = 200;
    // double initialScale = 1.0;
    // double scalingRate = 0.707;
    // int baseScale = 200;
    // std::vector<double> bandwidths = {10.0, 8.0, 0.5};
    // std::vector<int> steps = {20, 10};
    // int votesDeleteStep = 50;
    // int votesBufferLength = 200;
    // double votingSpaceDiscretizeRatio = 0.5;
    //// std::vector<double> scoreThresholds = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    // std::vector<double> scoreThresholds(6, 2.0);
    //// std::vector<double> scoreThresholds(6, 0.05);
    // bool hasNegativeClass = true;
    // bool isBackprojection = false;
    // TreeParameters treeParameters(nClasses, 0, 0, 0, 0, 0, 0, TreeParameters::ALL_RATIO,
    //                              hasNegativeClass);
    // HoughForestsParameters parameters(width, height, scales, baseScale, nClasses,
    // bandwidths.at(0),
    //                                  bandwidths.at(1), bandwidths.at(2), steps.at(0),
    //                                  steps.at(1),
    //                                  votesDeleteStep, votesBufferLength,
    //                                  votingSpaceDiscretizeRatio, scoreThresholds,
    //                                  hasNegativeClass,
    //                                  isBackprojection, treeParameters);
    // HoughForests houghForests(nThreads);
    // houghForests.setHoughForestsParameters(parameters);
    // houghForests.load(forestsDirectoryPath);

    // houghForests.classify(extractor);
}

int main() {
    // extractPositiveFeatures();
    // extractNegativeFeatures();
    // train();
    // detect();
    detectAll();
    // detectAllSTIP();
    // train1data();
    // classify();

    // using namespace nuisken;
    // using namespace nuisken::houghforests;
    // using namespace nuisken::storage;

    // int localWidth = 51;
    // int localHeight = localWidth;
    // int localDuration = 31;
    // int xStep = 25;
    // int yStep = xStep;
    // int tStep = 15;
    // std::vector<double> scales = {1.0, 0.5};
    // LocalFeatureExtractor lfe("D:/TestData/test_half.avi", localWidth, localHeight,
    // localDuration,
    //                          xStep, yStep, tStep, scales);
    // std::vector<std::vector<cv::Vec3i>> points;
    // std::vector<std::vector<std::vector<float>>> descs;
    // lfe.extractLocalFeatures(points, descs);
    // points.clear();
    // descs.clear();
    // lfe.extractLocalFeatures(points, descs);
    // points.clear();
    // descs.clear();
    // lfe.extractLocalFeatures(points, descs);

    // int validationIndex = 0;
    // int sequenceIndex = 5;
    // std::string forestsRootDirectoryPath =
    // "E:/Hara/UT-Interaction/data_fixed_scale/forests_first_layer_allratio/";
    // std::string featureDirectoryPath =
    // "E:/Hara/UT-Interaction/feature_stip2_multiscale_first_layer/";
    // std::vector<std::string> featureFilePaths;
    // for (int s = 0; s < 3; ++s) {
    //    featureFilePaths.push_back((boost::format("%sseq%d_s0.txt") % featureDirectoryPath %
    //    sequenceIndex).str());
    //}
    // std::string outputDirectoryPath = "E:/Hara/UT-Interaction/online_results/voting/";

    // std::size_t width = 720;
    // std::size_t height = 480;
    // std::vector<double> scales = {1.0, 0.707, 0.5};
    // int baseScale = 200;
    // int nClasses = 7;
    // double sigma = 10.0;
    // double tau = 8.0;
    // double scaleBandwidth = 0.5;
    // int spatialStep = 20;
    // int temporalStep = 10;
    // int votesDeleteStep = 50;
    // int votesBufferLength = 200;
    // std::vector<double> scoreThresholds = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    // bool hasNegativeClass = true;
    // bool isBackprojection = false;
    // randomforests::TreeParameters treeParameters(nClasses, 0, 0, 0, 0, 0, 0,
    // randomforests::TreeParameters::ALL_RATIO, hasNegativeClass);
    // HoughForestsParameters parameters(width, height, scales, baseScale, nClasses, sigma, tau,
    // scaleBandwidth, spatialStep, temporalStep,
    //                                  votesDeleteStep, votesBufferLength, scoreThresholds,
    //                                  hasNegativeClass, isBackprojection, treeParameters);

    // int nThreads = 1;
    // HoughForests houghForests(nThreads);
    // houghForests.setHoughForestsParameters(parameters);
    // houghForests.load(forestsRootDirectoryPath + std::to_string(validationIndex) + "/");

    // std::vector<std::vector<DetectionResult<4>>> detectionResults;
    // houghForests.detect(featureFilePaths, detectionResults);

    // std::cout << "output" << std::endl;
    // for (auto classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
    //    std::string outputFilePath =
    //        outputDirectoryPath + std::to_string(sequenceIndex) + "_" + std::to_string(classLabel)
    //        + "_detection.txt";

    //    std::ofstream outputStream(outputFilePath);
    //    for (const auto& detectionResult : detectionResults.at(classLabel)) {
    //        LocalMaximum localMaximum = detectionResult.getLocalMaximum();
    //        outputStream << "LocalMaximum," << localMaximum.getPoint()(T) << ","
    //            << localMaximum.getPoint()(Y) << "," << localMaximum.getPoint()(X) << ","
    //            << localMaximum.getValue() << "," << localMaximum.getPoint()(3) << std::endl;

    //        auto contributionPoints = detectionResult.getContributionPoints();
    //        for (const auto& contributionPoint : contributionPoints) {
    //            outputStream << contributionPoint.getPoint()(T) << ","
    //                << contributionPoint.getPoint()(Y) << ","
    //                << contributionPoint.getPoint()(X) << "," << contributionPoint.getValue()
    //                << std::endl;
    //        }
    //        outputStream << std::endl;
    //    }
    //}
}