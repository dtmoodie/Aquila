#pragma once

#include "Node.h"

namespace aq
{
    namespace Nodes
    {
        /*class AQUILA_EXPORTS Sink: public Node
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream) = 0;
        };
        class AQUILA_EXPORTS CpuSink: public Sink
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual void doProcess(const cv::Mat& mat, mo::time_t timestamp, size_t frame_number, cv::cuda::Stream& stream) = 0;
        };
        class AQUILA_EXPORTS GpuSink : public Sink
        {
        public:
            virtual TS<SyncedMemory> process(TS<SyncedMemory> input, cv::cuda::Stream& stream);
            virtual void doProcess(const cv::cuda::GpuMat& mat, mo::time_t timestamp, size_t frame_number, cv::cuda::Stream& stream) = 0;
        };*/
    } // namespace Nodes    
} // namespace aq