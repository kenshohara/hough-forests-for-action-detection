#ifndef RANDOM_FORESTS_INL
#define RANDOM_FORESTS_INL

#include "RandomForests.h"
#include "RandomGenerator.h"
#include "ThreadProcess.h"

#include <chrono>

namespace nuisken {
namespace randomforests {

template <class Type>
void RandomForests<Type>::train(const std::vector<FeaturePtr>& features, int maxNumberOfThreads) {
    using TrainTask = std::function<void()>;
    std::queue<TrainTask> tasks;
    for (auto i = 0; i < forests.size(); ++i) {
        tasks.push([&, this, i]() { trainOneTree(features, i); });
    }

    thread::threadProcess(tasks, maxNumberOfThreads);
}

template <class Type>
void RandomForests<Type>::trainOneTree(const std::vector<FeaturePtr>& features, int index) {
    std::cout << "tree : " << index << std::endl;

    std::vector<FeatureRawPtr> bootstrapData;
    selectBootstrapData(features, bootstrapData);

    auto begin = std::chrono::system_clock::now();
    forests.at(index).grow(bootstrapData);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << std::endl;
}

template <class Type>
void RandomForests<Type>::trainOneTree(const std::vector<FeaturePtr>& features,
                                       std::vector<FeatureRawPtr>& bootstrapData, int index) {
    std::cout << "tree : " << index << std::endl;
    selectBootstrapData(features, bootstrapData);

    auto begin = std::chrono::system_clock::now();
    forests.at(index).grow(bootstrapData);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << std::endl;
}

template <class Type>
void RandomForests<Type>::selectBootstrapData(const std::vector<FeaturePtr>& features,
                                              std::vector<FeatureRawPtr>& bootstrapData) {
    if (parameters.getBootstrapType() == TreeParameters::ALL_RATIO) {
        selectBootstrapDataAllRatio(features, bootstrapData);
    } else if (parameters.getBootstrapType() == TreeParameters::MAX_WITHOUT_NEGATIVE) {
        selectBootstrapDataMaxWithoutNegative(features, bootstrapData);
    }
}

template <class Type>
void RandomForests<Type>::selectBootstrapDataAllRatio(const std::vector<FeaturePtr>& features,
                                                      std::vector<FeatureRawPtr>& bootstrapData) {
    double bootstrapRatio = parameters.getBootstrapRatio();
    int numberOfBootstrapData = features.size() * bootstrapRatio;

    std::vector<std::vector<FeatureRawPtr>> classFeaturesVector(type.getNumberOfClasses());
    for (const auto& patch : features) {
        classFeaturesVector.at(patch->getClassLabel()).push_back(patch.get());
    }

    bootstrapData.reserve(numberOfBootstrapData);
    for (int i = 0; i < numberOfBootstrapData; ++i) {
        std::uniform_int_distribution<> classDistribution(0, type.getNumberOfClasses() - 1);
        int classIndex = classDistribution(RandomGenerator::getInstance().generator_);
        if (!classFeaturesVector.at(classIndex).empty()) {
            std::uniform_int_distribution<> patchDistribution(
                    0, classFeaturesVector.at(classIndex).size() - 1);
            int patchIndex = patchDistribution(RandomGenerator::getInstance().generator_);
            bootstrapData.push_back(classFeaturesVector.at(classIndex).at(patchIndex));
        }
    }
}

template <class Type>
void RandomForests<Type>::selectBootstrapDataMaxWithoutNegative(
        const std::vector<FeaturePtr>& features, std::vector<FeatureRawPtr>& bootstrapData) {
    if (!parameters.hasNegativeClass()) {
        std::cout << "data must have negative class" << std::endl;
        std::exit(0);
    }

    std::vector<std::vector<FeatureRawPtr>> classFeaturesVector(type.getNumberOfClasses());
    for (const auto& patch : features) {
        classFeaturesVector.at(patch->getClassLabel()).push_back(patch.get());
    }

    int maxSize = 0;
    for (int classLabel = 0; classLabel < parameters.getNumberOfClasses() - 1; ++classLabel) {
        if (classFeaturesVector.at(classLabel).size() > maxSize) {
            maxSize = classFeaturesVector.at(classLabel).size();
        }
    }

    int numberOfBootstrapData =
            maxSize * parameters.getNumberOfClasses() * parameters.getBootstrapRatio();
    bootstrapData.reserve(numberOfBootstrapData);
    for (int i = 0; i < numberOfBootstrapData; ++i) {
        std::uniform_int_distribution<> classDistribution(0, type.getNumberOfClasses() - 1);
        int classIndex = classDistribution(RandomGenerator::getInstance().generator_);
        if (!classFeaturesVector.at(classIndex).empty()) {
            std::uniform_int_distribution<> patchDistribution(
                    0, classFeaturesVector.at(classIndex).size() - 1);
            int patchIndex = patchDistribution(RandomGenerator::getInstance().generator_);
            bootstrapData.push_back(classFeaturesVector.at(classIndex).at(patchIndex));
        }
    }
}

template <class Type>
std::vector<typename RandomForests<Type>::LeafPtr> RandomForests<Type>::match(
        const FeaturePtr& feature) const {
    std::vector<LeafPtr> leafData;
    leafData.reserve(forests.size());
    for (const auto& tree : forests) {
        leafData.push_back(tree.match(feature.get()));
    }

    return leafData;
}

template <class Type>
void RandomForests<Type>::save(const std::string& directoryPath) const {
    std::string parametersFilePath = directoryPath + "TreeParameters.xml";
    parameters.save(parametersFilePath);

    for (int i = 0; i < forests.size(); ++i) {
        std::string filePath = directoryPath + "tree" + std::to_string(i) + ".csv";
        std::ofstream treeStream(filePath);
        forests.at(i).save(treeStream);
    }
}

template <class Type>
void RandomForests<Type>::load(const std::string& directoryPath) {
    std::string parametersFilePath = directoryPath + "TreeParameters.xml";
    parameters.load(parametersFilePath);

    std::vector<std::string> treeFilePaths;
    std::tr2::sys::path directory(directoryPath);
    std::tr2::sys::directory_iterator end;
    for (std::tr2::sys::directory_iterator itr(directory); itr != end; ++itr) {
        auto index = itr->path().filename().string().find("tree");
        if (index != std::string::npos) {
            treeFilePaths.push_back(directory.string() + itr->path().filename().string());
        }
    }

    forests.resize(treeFilePaths.size());
    for (int i = 0; i < forests.size(); ++i) {
        std::cout << "load tree " << i << std::endl;
        forests.at(i).setParameters(parameters);
        forests.at(i).setType(type);

        std::ifstream treeSteram(treeFilePaths.at(i));
        forests.at(i).load(treeSteram);
    }
}
}
}

#endif