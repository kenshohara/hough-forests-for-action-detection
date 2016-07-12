#ifndef TREE_NODE
#define TREE_NODE

#include "TreeParameters.h"

#include <opencv2/core/core.hpp>

#include <boost/tokenizer.hpp>

#include <algorithm>
#include <memory>
#include <queue>
#include <random>

namespace nuisken {
namespace randomforests {

/**
 * 決定木のノードのクラス
 */
template <class Type>
class TreeNode {
   private:
    typedef typename Type::FeatureType* FeatureRawPtr;
    typedef std::shared_ptr<typename Type::LeafType> LeafPtr;
    typedef typename Type::SplitParametersType SplitParameters;
    typedef std::pair<double, FeatureRawPtr> Container;

   private:
    Type type;

    /**
     * ノードの深さ
     */
    int depth;

    /**
     * ノードのインデックス
     */
    int nodeIndex;

    /**
     * 葉ノードかどうか
     */
    bool leaf;

    /**
     * マッチした時に返す値
     * （葉ノードのみ）
     */
    LeafPtr leafData;

    /**
     * 分岐のパラメータ
     */
    SplitParameters splitParameter;

    /**
     * 分岐のパラメータ
     */
    double tau;

    /**
     * 子ノードのポインタ
     */
    std::unique_ptr<TreeNode<Type>> leftChild;
    std::unique_ptr<TreeNode<Type>> rightChild;

   public:
    TreeNode(){};
    TreeNode(const Type& type, int depth, int nodeIndex, bool leaf = false)
            : type(type),
              depth(depth),
              nodeIndex(nodeIndex),
              leaf(leaf),
              tau(0.0),
              rightChild(nullptr),
              leftChild(nullptr){};

    bool isLeaf() const { return leaf; }

    int getDepth() const { return depth; }

    std::unique_ptr<TreeNode<Type>> getLeftChild() const { return leftChild; }

    std::unique_ptr<TreeNode<Type>> getRightChild() const { return rightChild; }

    int getNodeIndex() const { return nodeIndex; }

    void setNodeIndex(int nodeIndex) { this->nodeIndex = nodeIndex; }

    void setLeafData(const LeafPtr& leafData) { this->leafData = leafData; }

    void setLeftChild(std::unique_ptr<TreeNode>&& leftChild) {
        this->leftChild = std::move(leftChild);
    }

    void setRightChild(std::unique_ptr<TreeNode>&& rightChild) {
        this->rightChild = std::move(rightChild);
    }

    void setType(const Type& type) { this->type = type; }

    /**
     * パラメータを学習する
     * 葉ノードであればtrue，それ以外はfalseを返す
     */
    bool train(const std::vector<FeatureRawPtr>& features, const TreeParameters& treeParameters,
               std::vector<FeatureRawPtr>& leftFeatures, std::vector<FeatureRawPtr>& rightFeatures);

    /**
     * どの葉ノードに対応するパッチか判断する
     * 葉ノード以外では学習したパラメータでどちらの子に投げるか決める
     * 葉ノードではそこに対応したデータを返す
     */
    LeafPtr match(const FeatureRawPtr& feature) const;

    /**
     * 現在のノード番号を返す
     */
    void save(std::ofstream& treeStream) const;
    void load(std::ifstream& treeStream);

   private:
    /**
     * データを2つに分割
     */
    void split(const std::vector<Container>& splitValues, double tau,
               std::vector<FeatureRawPtr>& leftFeatures,
               std::vector<FeatureRawPtr>& rightFeatures) const;

    void saveNode(std::ofstream& treeStream) const;
    void loadNode(std::queue<std::string>& nodeElements);
};
}
}

#endif