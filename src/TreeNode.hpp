#ifndef TREE_NODE_INL
#define TREE_NODE_INL

#include "TreeNode.h"

#include <boost/spirit/include/qi.hpp>

#include <limits>

namespace nuisken {
namespace randomforests {

template <class Type>
bool TreeNode<Type>::train(const std::vector<FeatureRawPtr>& features,
                           const TreeParameters& treeParameters,
                           std::vector<FeatureRawPtr>& leftFeatures,
                           std::vector<FeatureRawPtr>& rightFeatures) {
    //葉ノードであれば学習は行わない
    if (isLeaf() || depth >= treeParameters.getMaxDepth() ||
        features.size() <= treeParameters.getMinNumberOfData()) {
        leaf = true;
        return true;
    }

    //最適な結果の値を保持
    auto bestValue = -std::numeric_limits<double>::max();

    for (int i = 0; i < treeParameters.getNumberOfTrainIteration(); ++i) {
        //ランダムにパラメータを選択
        SplitParameters tempParameter = type.generateRandomParameter();

        //選択したパラメータで2点の特徴の差を計算
        std::vector<Container> splitValues;
        splitValues.reserve(features.size());
        for (const auto& feature : features) {
            splitValues.emplace_back(type.calculateSplitValue(feature, tempParameter), feature);
        }
        std::sort(std::begin(splitValues), std::end(splitValues),
                  [](const Container& x, const Container& y) { return x.first < y.first; });

        //τの範囲を決定（evaluateで計算した値の最小値～最大値の範囲）
        auto minValue = splitValues.front().first;
        auto maxValue = splitValues.back().first;

        if (0 == (maxValue - minValue)) {
            continue;
        }

        for (int j = 0; j < treeParameters.getNumberOfTauIteration(); ++j) {
            auto tempTau = type.generateTau(minValue, maxValue);
            std::vector<FeatureRawPtr> tempLeftFeatures;
            std::vector<FeatureRawPtr> tempRightFeatures;
            split(splitValues, tempTau, tempLeftFeatures, tempRightFeatures);

            //分割した結果を評価

            auto tempValue = type.evaluateSplit(tempLeftFeatures, tempRightFeatures);

            //よりよい結果なら結果を更新
            if (tempValue > bestValue) {
                bestValue = tempValue;
                leftFeatures = tempLeftFeatures;
                rightFeatures = tempRightFeatures;
                splitParameter = tempParameter;
                tau = tempTau;
            }
        }
    }

    if (0 == leftFeatures.size() || 0 == rightFeatures.size()) {
        leaf = true;
        return true;
    } else {
        return false;
    }
}

template <class Type>
void TreeNode<Type>::split(const std::vector<Container>& splitValues, double tau,
                           std::vector<FeatureRawPtr>& leftFeatures,
                           std::vector<FeatureRawPtr>& rightFeatures) const {
    leftFeatures.reserve(splitValues.size());
    rightFeatures.reserve(splitValues.size());

    auto splitItr = std::begin(splitValues);
    while ((*splitItr).first < tau) {
        leftFeatures.push_back((*splitItr).second);
        ++splitItr;
    }

    auto end = std::end(splitValues);
    // std::transform(splitItr, end, std::back_inserter(rightFeatures),
    //               [](const Container& a) {return a.second; });
    for (auto itr = splitItr; itr != end; ++itr) {
        rightFeatures.push_back((*itr).second);
    }
}

template <class Type>
typename TreeNode<Type>::LeafPtr TreeNode<Type>::match(const FeatureRawPtr& feature) const {
    if (isLeaf()) {
        return leafData;
    } else {
        if (type.decision(feature, splitParameter, tau)) {
            return leftChild->match(feature);
        } else {
            return rightChild->match(feature);
        }
    }
}

template <class Type>
void TreeNode<Type>::save(std::ofstream& treeStream) const {
    saveNode(treeStream);
    treeStream << std::endl;

    if (leftChild != 0) {
        leftChild->save(treeStream);
    }
    if (rightChild != 0) {
        rightChild->save(treeStream);
    }
}

template <class Type>
void TreeNode<Type>::saveNode(std::ofstream& treeStream) const {
    treeStream << depth << "," << leaf << "," << tau << ",";
    splitParameter.save(treeStream);

    if (leaf) {
        leafData->save(treeStream);
    }
}

template <class Type>
void TreeNode<Type>::load(std::ifstream& treeStream) {
    std::string line;
    std::getline(treeStream, line);
    boost::tokenizer<boost::escaped_list_separator<char>> tokenizer(line);
    std::queue<std::string> nodeElements;
    for (auto it = std::begin(tokenizer); it != std::end(tokenizer); ++it) {
        nodeElements.push(*it);
    }

    loadNode(nodeElements);

    if (leaf) {
        leafData = type.loadLeafData(nodeElements);
    } else {
        leftChild = std::make_unique<TreeNode<Type>>();
        leftChild->setType(type);
        leftChild->load(treeStream);

        rightChild = std::make_unique<TreeNode<Type>>();
        rightChild->setType(type);
        rightChild->load(treeStream);
    }
}

template <class Type>
void TreeNode<Type>::loadNode(std::queue<std::string>& nodeElements) {
    boost::spirit::qi::parse(std::begin(nodeElements.front()), std::end(nodeElements.front()),
                             boost::spirit::qi::int_, depth);
    nodeElements.pop();

    int leafInt;
    boost::spirit::qi::parse(std::begin(nodeElements.front()), std::end(nodeElements.front()),
                             boost::spirit::qi::int_, leafInt);
    leaf = static_cast<bool>(leafInt);
    nodeElements.pop();

    boost::spirit::qi::parse(std::begin(nodeElements.front()), std::end(nodeElements.front()),
                             boost::spirit::qi::double_, tau);
    nodeElements.pop();

    splitParameter.load(nodeElements);
}
}
}

#endif