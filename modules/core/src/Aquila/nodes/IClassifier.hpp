#pragma once

#include "Node.hpp"

namespace aq{
    namespace nodes{
        class IClassifier: public Node{
        public:
            MO_DERIVE(IClassifier, Node)
                PARAM(mo::ReadFile, label_file, {})
                PARAM_UPDATE_SLOT(label_file)
                OUTPUT(std::vector<std::string>, labels, {})
                APPEND_FLAGS(labels, mo::Unstamped_e)
                OUTPUT(std::vector<cv::Scalar>, colormap, {})
                APPEND_FLAGS(colormap, mo::Unstamped_e)
            MO_END;
        };
    }
}