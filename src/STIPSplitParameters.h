#ifndef STIP_SPLIT_PARAMETERS
#define STIP_SPLIT_PARAMETERS

#include <opencv2/core/core.hpp>

#include <queue>
#include <fstream>

namespace nuisken {
namespace randomforests {

/**
 * 時空間局所特徴用の分割関数のパラメータ
 * 特徴の2点間の差分で分割する
 */
class STIPSplitParameters {
   private:
    int index1;
    int index2;
    int featureChannel;

   public:
    STIPSplitParameters() : index1(0), index2(0), featureChannel(0){};

    STIPSplitParameters(int index1, int index2, int featureChannel)
            : index1(index1), index2(index2), featureChannel(featureChannel){};

    int getIndex1() const { return index1; }

    int getIndex2() const { return index2; }

    int getFeatureChannel() const { return featureChannel; }

    void save(std::ofstream& treeStream) const;
    void load(std::queue<std::string>& nodeElements);

   private:
    void setIndex1(int index1) { this->index1 = index1; }

    void setIndex2(int index2) { this->index2 = index2; }

    void setFeatureChannel(int featureChannel) { this->featureChannel = featureChannel; }
};
}
}

#endif