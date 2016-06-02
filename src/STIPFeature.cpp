#include "STIPFeature.h"
#include "Utils.h"

namespace nuisken {
namespace storage {

// void STIPFeature::save(const std::string& filePath) const {
// cv::FileStorage fileStorage(filePath, CV_STORAGE_WRITE);
// cv::write(fileStorage, "classLabel", getClassLabel());

// io::saveXYTPoint(fileStorage, "centerPoint", getCenterPoint());
// io::saveXYTPoint(fileStorage, "displacementVector", getDisplacementVector());

// cv::WriteStructContext writeStructContext(fileStorage, "featureVector", CV_NODE_SEQ);
// auto featureVectors = getFeatureVectors();
// for (const auto& feature : featureVectors) {
//    cv::write(fileStorage, "", feature);
//}
//}

// void STIPFeature::load(const std::string& filePath) {
//    cv::FileStorage fileStorage(filePath, CV_STORAGE_READ);
//    cv::FileNode topNode(fileStorage.fs, 0);
//
//    setClassLabel(topNode["classLabel"]);
//
//    setCenterPoint(io::loadXYTPoint(fileStorage, "centerPoint"));
//    setDisplacementVector(io::loadXYTPoint(fileStorage, "displacementVector"));
//
//    cv::FileNode featureNode = topNode["featureVectors"];
//
//    std::vector<cv::Mat1f> featureVectors(featureNode.size());
//    for (auto i = 0; i < featureNode.size(); ++i) {
//        cv::read(featureNode[i], featureVectors.at(i));
//    }
//    setFeatureVectors(featureVectors);
//}
}
}