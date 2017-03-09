#pragma once
#include "Node.h"
namespace aq
{
    namespace Nodes
    {
    
    class AQUILA_EXPORTS CpuExtraction : public Node
    {
    public:
        virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void process(const cv::Mat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream) = 0;
    };

    class AQUILA_EXPORTS GpuExtraction : public Node
    {
    public:
        virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual void process(const cv::cuda::GpuMat& mat, double timestamp, int frame_number, cv::cuda::Stream& stream) = 0;
    };
    } // namespace Nodes
} // namespace aq