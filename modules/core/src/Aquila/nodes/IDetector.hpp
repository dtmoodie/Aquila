#include "IClassifier.hpp"
#include <Aquila/types/ObjectDetection.hpp>
namespace aq{
namespace nodes{
    class IImageDetector: IClassifier{
    public:
        MO_DERIVE(IDetector, IClassifier)
            OUTPUT(std::vector<DetectedObject2d>)
        MO_END;
    };
}
}