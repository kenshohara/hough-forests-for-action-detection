#ifndef TREE_PARAMETERS
#define TREE_PARAMETERS

#include <opencv2/highgui/highgui.hpp>

#include <string>

namespace nuisken {
namespace randomforests {

/**
 * 決定木のパラメータのクラス
 */
class TreeParameters {
   public:
    enum BootstrapType { ALL_RATIO, MAX_WITHOUT_NEGATIVE };

   private:
    /**
     * 学習するクラス数
     */
    int numberOfClasses_;

    /**
     * 決定木の数
     */
    int numberOfTrees_;

    /**
     * 各決定木に渡すデータ数
     */
    double bootstrapRatio_;

    /**
     * 最大の深さ
     */
    int maxDepth_;

    /**
     * 最小のデータ数
     */
    int minNumberOfData_;

    /**
     * ノードでのパラメータ学習時の繰り返し回数
     *（τ以外）
     */
    int numberOfTrainIteration_;

    /**
     * τの決定時の繰り返し回数
     */
    int numberOfTauIteration_;

    BootstrapType type_;
    bool hasNegativeClass_;

   public:
    TreeParameters(){};
    TreeParameters(int numberOfClasses, int numberOfTrees, double bootstrapRatio, int maxDepth,
                   int minNumberOfData, int numberOfTrainIteration, int numberOfTauIteration,
                   BootstrapType type, bool hasNegativeClass)
            : numberOfClasses_(numberOfClasses),
              numberOfTrees_(numberOfTrees),
              bootstrapRatio_(bootstrapRatio),
              maxDepth_(maxDepth),
              minNumberOfData_(minNumberOfData),
              numberOfTrainIteration_(numberOfTrainIteration),
              numberOfTauIteration_(numberOfTauIteration),
              type_(type),
              hasNegativeClass_(hasNegativeClass){};

    int getNumberOfClasses() const { return numberOfClasses_; }

    int getNumberOfTrees() const { return numberOfTrees_; }

    double getBootstrapRatio() const { return bootstrapRatio_; }

    int getMaxDepth() const { return maxDepth_; }

    int getMinNumberOfData() const { return minNumberOfData_; }

    int getNumberOfTrainIteration() const { return numberOfTrainIteration_; }

    int getNumberOfTauIteration() const { return numberOfTauIteration_; }

    BootstrapType getBootstrapType() const { return type_; }

    bool hasNegativeClass() const { return hasNegativeClass_; }

    void setNumberOfClasses(int numberOfClasses) { this->numberOfClasses_ = numberOfClasses; }

    void setNumberOfTrees(int numberOfTrees) { this->numberOfTrees_ = numberOfTrees; }

    void setBootstrapRatio(double bootstrapRatio) { this->bootstrapRatio_ = bootstrapRatio; }

    void setMaxDepth(int maxDepth) { this->maxDepth_ = maxDepth; }

    void setMinNumberOfData(int minNumberOfData) { this->minNumberOfData_ = minNumberOfData; }

    void setNumberOfTrainIteration(int numberOfTrainIteration) {
        this->numberOfTrainIteration_ = numberOfTrainIteration;
    }

    void setNumberOfTauIteration(int numberOfTauIteration) {
        this->numberOfTauIteration_ = numberOfTauIteration;
    }

    void save(const std::string& filePath) const {
        cv::FileStorage fileStorage(filePath, CV_STORAGE_WRITE);
        cv::write(fileStorage, "numberOfClasses", numberOfClasses_);
        cv::write(fileStorage, "numberOfTrees", numberOfTrees_);
        cv::write(fileStorage, "bootstrapRatio", bootstrapRatio_);
        cv::write(fileStorage, "maxDepth", maxDepth_);
        cv::write(fileStorage, "minNumberOfData", minNumberOfData_);
        cv::write(fileStorage, "numberOfTrainIteration", numberOfTrainIteration_);
        cv::write(fileStorage, "numberOfTauIteration", numberOfTauIteration_);
        if (type_ == ALL_RATIO) {
            cv::write(fileStorage, "bootstrapType", "ALL_RATIO");
        } else if (type_ == MAX_WITHOUT_NEGATIVE) {
            cv::write(fileStorage, "bootstrapType", "MAX_WITHOUT_NEGATIVE");
        }
        cv::write(fileStorage, "hasNegativeClass", hasNegativeClass_);
    }

    void load(const std::string& filePath) {
        cv::FileStorage fileStorage(filePath, CV_STORAGE_READ);
        cv::FileNode topNode(fileStorage.fs, 0);

        numberOfClasses_ = topNode["numberOfClasses"];
        numberOfTrees_ = topNode["numberOfTrees"];
        bootstrapRatio_ = topNode["bootstrapRatio"];
        maxDepth_ = topNode["maxDepth"];
        minNumberOfData_ = topNode["minNumberOfData"];
        numberOfTrainIteration_ = topNode["numberOfTrainIteration"];
        numberOfTauIteration_ = topNode["numberOfTauIteration"];

        int tmpType = topNode["bootstrapType"];
        if (tmpType == 0) {
            type_ = ALL_RATIO;
        } else if (tmpType == 1) {
            type_ = MAX_WITHOUT_NEGATIVE;
        }

        int tmpHasNegativeClass = topNode["hasNegativeClass"];
        if (tmpHasNegativeClass == 1) {
            hasNegativeClass_ = true;
        } else {
            hasNegativeClass_ = false;
        }
    }
};
}
}

#endif