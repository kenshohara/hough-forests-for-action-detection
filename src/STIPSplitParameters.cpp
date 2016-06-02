#include "STIPSplitParameters.h"
#include "STIPFeature.h"

#include <boost/spirit/include/qi.hpp>

namespace nuisken {
namespace randomforests {

void STIPSplitParameters::save(std::ofstream& treeStream) const {
    treeStream << getFeatureChannel() << "," << getIndex1() << "," << getIndex2() << ",";
}

void STIPSplitParameters::load(std::queue<std::string>& nodeElements) {
    int featureChannel;
    boost::spirit::qi::parse(std::begin(nodeElements.front()), std::end(nodeElements.front()),
                             boost::spirit::qi::int_, featureChannel);
    setFeatureChannel(featureChannel);
    nodeElements.pop();

    int index1;
    boost::spirit::qi::parse(std::begin(nodeElements.front()), std::end(nodeElements.front()),
                             boost::spirit::qi::int_, index1);
    setIndex1(index1);
    nodeElements.pop();

    int index2;
    boost::spirit::qi::parse(std::begin(nodeElements.front()), std::end(nodeElements.front()),
                             boost::spirit::qi::int_, index2);
    setIndex2(index2);
    nodeElements.pop();
}
}
}