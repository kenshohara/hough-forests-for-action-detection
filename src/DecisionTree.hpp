#ifndef DECISION_TREE_INL
#define DECISION_TREE_INL

#include "DecisionTree.h"

namespace nuisken {
namespace randomforests {

template <class Type>
void DecisionTree<Type>::grow(const std::vector<FeatureRawPtr>& features) {
    //根ノードを追加
    auto rootDepth = 1;
    auto nodeIndex = 0;
    root = std::make_unique<TreeNode<Type>>(type, rootDepth, nodeIndex++);

    //根ノードから学習
    trainNode(root, features, parameters, nodeIndex);
}

template <class Type>
void DecisionTree<Type>::trainNode(std::unique_ptr<TreeNode<Type>>& node,
                                   const std::vector<FeatureRawPtr>& features,
                                   const TreeParameters& parameters, int& nodeIndex) {
    std::vector<FeatureRawPtr> leftFeatures;
    std::vector<FeatureRawPtr> rightFeatures;
    bool isLeaf = node->train(features, parameters, leftFeatures, rightFeatures);

    if (!isLeaf) {
        auto leftChild = std::make_unique<TreeNode<Type>>(type, node->getDepth() + 1, nodeIndex++);
        auto rightChild = std::make_unique<TreeNode<Type>>(type, node->getDepth() + 1, nodeIndex++);

        trainNode(leftChild, leftFeatures, parameters, nodeIndex);
        trainNode(rightChild, rightFeatures, parameters, nodeIndex);

        node->setLeftChild(std::move(leftChild));
        node->setRightChild(std::move(rightChild));
    } else {
        node->setLeafData(type.calculateLeafData(features));
    }
}

template <class Type>
int DecisionTree<Type>::getNumberOfLeaves() const {
    auto numberofLeaves = 0;
    return getNumberOfLeaves(root, numberofLeaves);
}

template <class Type>
int DecisionTree<Type>::getNumberOfLeaves(const std::unique_ptr<TreeNode<Type>>& node,
                                          int numberOfLeaves) const {
    if (node == nullptr) {
        return numberOfLeaves;
    } else if (node->isLeaf()) {
        return numberOfLeaves + 1;
    } else {
        return getNumberOfLeaves(node->getLeftChild(), numberOfLeaves) +
               getNumberOfLeaves(node->getRightChild(), numberOfLeaves);
    }
}

template <class Type>
void DecisionTree<Type>::numberNodes() {
    auto nodeIndex = 0;
    numberNodes(root, nodeIndex);
}

template <class Type>
void DecisionTree<Type>::numberNodes(std::unique_ptr<TreeNode<Type>>& node, int& nodeIndex) {
    if (node == nullptr) {
        return;
    } else {
        node->setNodeIndex(nodeIndex++);
        numberNodes(node->getLeftChild(), nodeIndex);
        numberNodes(node->getRightChild(), nodeIndex);
    }
}

template <class Type>
void DecisionTree<Type>::mapLeafIndices() {
    auto leafIndex = 0;
    mapLeafIndices(root, leafIndex);
}

template <class Type>
void DecisionTree<Type>::mapLeafIndices(const std::unique_ptr<TreeNode<Type>>& node,
                                        int& leafIndex) {
    if (node == nullptr) {
        return;
    } else if (node->isLeaf()) {
        leafIndices.insert(std::make_pair(node->getNodeIndex(), leafIndex++));
    } else {
        mapLeafIndices(node->getLeftChild(), leafIndex);
        mapLeafIndices(node->getRightChild(), leafIndex);
    }
}

template <class Type>
void DecisionTree<Type>::save(std::ofstream& treeStream) const {
    root->save(treeStream);
}

template <class Type>
void DecisionTree<Type>::load(std::ifstream& treeStream) {
    root = std::make_unique<TreeNode<Type>>();
    root->setType(type);
    root->load(treeStream);
}
}
}

#endif