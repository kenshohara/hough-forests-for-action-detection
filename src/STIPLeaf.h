#ifndef STIP_LEAF
#define STIP_LEAF

#include "Storage.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <memory>
#include <queue>
#include <tuple>
#include <vector>

namespace nuisken {
namespace randomforests {

/**
 * 決定木で識別された時，返るデータのクラス
 */
class STIPLeaf {
   public:
    typedef storage::FeatureInfo FeatureInfo;

   private:
    /**
     * 葉ノードに対応付けられた各特徴の情報
     */
    std::vector<FeatureInfo> featureInfo;

   public:
    STIPLeaf(){};
    STIPLeaf(const std::vector<FeatureInfo>& featureInfo) : featureInfo(featureInfo) {}

    std::vector<FeatureInfo> getFeatureInfo() const { return featureInfo; }

    void setFeatureInfo(const std::vector<FeatureInfo>& featureInfo) {
        this->featureInfo = featureInfo;
    }

    void save(std::ofstream& treeStream) const;
    void load(std::queue<std::string>& nodeElements);
};
}
}

#endif