#include "HoughForests.h"
#include "LocalFeatureExtractor.h"
#include "STIPFeature.h"
#include "Utils.h"

#include <numpy.hpp>

#include <boost/format.hpp>

#include <Eigen/Core>

#include <filesystem>
#include <string>
#include <vector>

void extractPositiveFeatures() {
    using namespace nuisken::houghforests;

    int localWidth = 21;
    int localHeight = localWidth;
    int localDuration = 9;
    int xStep = 10;
    int yStep = xStep;
    int tStep = 5;
    std::vector<double> scales = {1.0};

    int nSamplesPerStep = 30;
    int randomSeed = 1;
    std::mt19937 randomEngine(randomSeed);

    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    std::string videoDirectoryPath = rootDirectoryPath + "segmented_fixed_scale/";
    std::string outputDirectoryPath = rootDirectoryPath + "feature_hf/";
    std::tr2::sys::path directory(videoDirectoryPath);
    std::tr2::sys::directory_iterator end;
    for (std::tr2::sys::directory_iterator itr(directory); itr != end; ++itr) {
        std::string filePath = itr->path().string();

        std::cout << "extract" << std::endl;
        LocalFeatureExtractor extractor(filePath, scales, localWidth, localHeight, localDuration,
                                        xStep, yStep, tStep);
        std::vector<cv::Vec3i> selectedPoints;
        std::vector<std::vector<float>> selectedDescriptors;
        while (true) {
            std::cout << "frame: " << extractor.getStoredStartT() << std::endl;
            std::vector<std::vector<cv::Vec3i>> points;
            std::vector<std::vector<std::vector<float>>> descriptors;
            extractor.extractLocalFeatures(points, descriptors);
            if (extractor.isEnd()) {
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
        aoba::SaveArrayAsNumpy<int>(outputPointsFilePath, selectedPoints.front().rows,
                                    selectedPoints.size(), outputPoints.data());

        std::string outputDescriptorsFilePath = outputDirectoryPath + outputFileName + "_desc.npy";
        std::vector<float> outputDescriptors;
        for (const auto& desc : selectedDescriptors) {
            for (int i = 0; i < desc.size(); ++i) {
                outputDescriptors.push_back(desc[i]);
            }
        }
        aoba::SaveArrayAsNumpy<float>(outputDescriptorsFilePath, selectedDescriptors.front().size(),
                                      selectedDescriptors.size(), outputDescriptors.data());
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

            auto startFrame = std::stoi(tokens.at(2));
            auto endFrame = std::stoi(tokens.at(3));
            auto topLeftX = std::stoi(tokens.at(4));
            auto topLeftY = std::stoi(tokens.at(5));
            auto bottomRightX = std::stoi(tokens.at(6));
            auto bottomRightY = std::stoi(tokens.at(7));
            boxes.emplace_back(cv::Point(topLeftX, topLeftY),
                               cv::Point(bottomRightX, bottomRightY));
            temporalRanges.emplace_back(startFrame, endFrame);
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
            auto startFrame = std::stoi(tokens.at(2));
            auto endFrame = std::stoi(tokens.at(3));
            auto topLeftX = std::stoi(tokens.at(4));
            auto topLeftY = std::stoi(tokens.at(5));
            auto bottomRightX = std::stoi(tokens.at(6));
            auto bottomRightY = std::stoi(tokens.at(7));

            boxes.emplace_back(cv::Point(topLeftX, topLeftY),
                               cv::Point(bottomRightX, bottomRightY));
            temporalRanges.emplace_back(startFrame, endFrame);
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
    int xStep = 10;
    int yStep = xStep;
    int tStep = 5;
    std::vector<double> scales = {1.0, 0.707, 0.5};

    int nSamplesPerStep = 3;
    int randomSeed = 1;
    std::mt19937 randomEngine(randomSeed);

    std::vector<std::string> filePaths;
    std::string videoDirectoryPath = "E:/Hara/UT-Interaction/unsegmented/";
    std::string labelFilePath = "E:/Hara/UT-Interaction/labels.csv";
    std::string outputDirectoryPath = "E:/Hara/UT-Interaction/feature_hf/";
    for (int sequenceIndex = 19; sequenceIndex <= 20; ++sequenceIndex) {
        std::string filePath =
                (boost::format("%sseq%d.avi") % videoDirectoryPath % sequenceIndex).str();
        std::vector<cv::Rect> boxes;
        std::vector<std::pair<int, int>> temporalRanges;
        std::cout << "read labels" << std::endl;
        readLabelsInfo(labelFilePath, sequenceIndex, boxes, temporalRanges);

        std::cout << "extract" << std::endl;
        LocalFeatureExtractor extractor(filePath, scales, localWidth, localHeight, localDuration,
                                        xStep, yStep, tStep);
        std::vector<cv::Vec3i> selectedPoints;
        std::vector<std::vector<float>> selectedDescriptors;
        while (true) {
            std::cout << "frame: " << extractor.getStoredStartT() << std::endl;
            std::vector<std::vector<cv::Vec3i>> points;
            std::vector<std::vector<std::vector<float>>> descriptors;
            extractor.extractLocalFeatures(points, descriptors);
            if (extractor.isEnd()) {
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
        aoba::SaveArrayAsNumpy<int>(outputPointsFilePath, selectedPoints.front().rows,
                                    selectedPoints.size(), outputPoints.data());

        std::string outputDescriptorsFilePath = outputDirectoryPath + outputFileName + "_desc.npy";
        std::vector<float> outputDescriptors;
        for (const auto& desc : selectedDescriptors) {
            for (int i = 0; i < desc.size(); ++i) {
                outputDescriptors.push_back(desc[i]);
            }
        }
        aoba::SaveArrayAsNumpy<float>(outputDescriptorsFilePath, selectedDescriptors.front().size(),
                                      selectedDescriptors.size(), outputDescriptors.data());
    }
}

void train() {
    using namespace nuisken::storage;
    using namespace nuisken::houghforests;
    using namespace nuisken::randomforests;

    const int N_CHANNELS = 6;
    const int N_CLASSES = 7;

    std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    // std::string rootDirectoryPath = "E:/Hara/UT-Interaction/";
    std::string featureDirectoryPath = rootDirectoryPath + "feature_hf/";
    std::string labelFilePath = rootDirectoryPath + "labels.csv";
    std::string forestsDirectoryPath = rootDirectoryPath + "data_hf/forests/";
    std::vector<std::vector<int>> validationCombinations = {{19, 5}, {12, 6}, {4, 7},   {16, 17},
                                                            {9, 13}, {11, 8}, {10, 14}, {18, 15},
                                                            {3, 20}, {2, 1}};

    for (int validationIndex = 0; validationIndex < validationCombinations.size();
         ++validationIndex) {
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
                        cv::Vec3i actionPosition;
                        actionPosition(0) = ranges[labelIndex].second - ranges[labelIndex].first;
                        actionPosition(1) = boxes[labelIndex].height / 2;
                        actionPosition(2) = boxes[labelIndex].width / 2;
                        cv::Vec3i offset = actionPosition - point;
                        auto data = std::make_shared<STIPFeature>(features, point, offset,
                                                                  std::make_pair(0.0, 0.0),
                                                                  classLabels[labelIndex]);
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
                    trainingData.push_back(data);
                }
            }
        }
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

void detect() {}

int main() {
    // extractPositiveFeatures();
    //extractNegativeFeatures();
     train();

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