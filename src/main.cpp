#include "HoughForests.h"
#include "Utils.h"

#include <boost/format.hpp>

#include <string>

int main() {
    using namespace nuisken;
    using namespace nuisken::houghforests;
    using namespace nuisken::storage;

    int validationIndex = 0;
    int sequenceIndex = 5;
    std::string forestsRootDirectoryPath = "E:/Hara/UT-Interaction/data_fixed_scale/forests_first_layer_allratio/";
    std::string featureDirectoryPath = "E:/Hara/UT-Interaction/feature_stip2_multiscale_first_layer/";
    std::vector<std::string> featureFilePaths;
    for (int s = 0; s < 3; ++s) {
        featureFilePaths.push_back((boost::format("%sseq%d_s0.txt") % featureDirectoryPath % sequenceIndex).str());
    }
    std::string outputDirectoryPath = "E:/Hara/UT-Interaction/online_results/voting/";

    std::size_t width = 720;
    std::size_t height = 480;
    std::vector<double> scales = {1.0, 0.707, 0.5};
    int baseScale = 200;
    int nClasses = 7;
    double sigma = 10.0;
    double tau = 8.0;
    double scaleBandwidth = 0.5;
    int spatialStep = 20;
    int temporalStep = 10;
    int votesDeleteStep = 50;
    int votesBufferLength = 200;
    std::vector<double> scoreThresholds = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    bool hasNegativeClass = true;
    bool isBackprojection = false;
    randomforests::TreeParameters treeParameters(nClasses, 0, 0, 0, 0, 0, 0, randomforests::TreeParameters::ALL_RATIO, hasNegativeClass);
    HoughForestsParameters parameters(width, height, scales, baseScale, nClasses, sigma, tau, scaleBandwidth, spatialStep, temporalStep, 
                                      votesDeleteStep, votesBufferLength, scoreThresholds, hasNegativeClass, isBackprojection, treeParameters);

    int nThreads = 1;
    HoughForests houghForests(nThreads);
    houghForests.setHoughForestsParameters(parameters);
    houghForests.load(forestsRootDirectoryPath + std::to_string(validationIndex) + "/");

    std::vector<std::vector<DetectionResult<4>>> detectionResults;
    houghForests.detect(featureFilePaths, detectionResults);

    std::cout << "output" << std::endl;
    for (auto classLabel = 0; classLabel < detectionResults.size(); ++classLabel) {
        std::string outputFilePath =
            outputDirectoryPath + std::to_string(sequenceIndex) + "_" + std::to_string(classLabel) + "_detection.txt";

        std::ofstream outputStream(outputFilePath);
        for (const auto& detectionResult : detectionResults.at(classLabel)) {
            LocalMaximum localMaximum = detectionResult.getLocalMaximum();
            outputStream << "LocalMaximum," << localMaximum.getPoint()(T) << ","
                << localMaximum.getPoint()(Y) << "," << localMaximum.getPoint()(X) << ","
                << localMaximum.getValue() << "," << localMaximum.getPoint()(3) << std::endl;

            auto contributionPoints = detectionResult.getContributionPoints();
            for (const auto& contributionPoint : contributionPoints) {
                outputStream << contributionPoint.getPoint()(T) << ","
                    << contributionPoint.getPoint()(Y) << ","
                    << contributionPoint.getPoint()(X) << "," << contributionPoint.getValue()
                    << std::endl;
            }
            outputStream << std::endl;
        }
    }
}