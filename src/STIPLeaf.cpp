#include "STIPLeaf.h"
#include "Utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace nuisken {
namespace randomforests {

void STIPLeaf::save(std::ofstream& treeStream) const {
    for (const auto& featureInfo : featureInfo) {
        treeStream << featureInfo.getIndex() << ",";
        treeStream << featureInfo.getClassLabel() << ",";
        treeStream << featureInfo.getSpatialScale() << ",";
        treeStream << featureInfo.getTemporalScale() << ",";
        cv::Vec3i displacementVector = featureInfo.getDisplacementVector();
        treeStream << displacementVector[T] << "," << displacementVector[Y] << ","
                   << displacementVector[X] << ",";
    }
}

void STIPLeaf::load(std::queue<std::string>& nodeElements) {
    int numberOfLeafElements = 7;
    int numberOfFeatureInfo = nodeElements.size() / numberOfLeafElements;
    featureInfo.resize(numberOfFeatureInfo);
    for (int i = 0; i < numberOfFeatureInfo; ++i) {
        int index = std::stoi(nodeElements.front());
        nodeElements.pop();
        int classLabel = std::stoi(nodeElements.front());
        nodeElements.pop();
        double spatialScale = std::stod(nodeElements.front());
        nodeElements.pop();
        double temporalScale = std::stod(nodeElements.front());
        nodeElements.pop();

        int t = std::stoi(nodeElements.front());
        nodeElements.pop();
        int y = std::stoi(nodeElements.front());
        nodeElements.pop();
        int x = std::stoi(nodeElements.front());
        nodeElements.pop();
        cv::Vec3i displacementVector(t, y, x);

        FeatureInfo aFeatureInfo(index, classLabel, spatialScale, temporalScale,
                                 displacementVector);
        featureInfo.at(i) = aFeatureInfo;
    }
}
}
}