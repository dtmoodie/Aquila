#include "Aquila/Nodes/Sink.h"

using namespace aq;
using namespace aq::Nodes;



/*TS<SyncedMemory> CpuSink::process(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    doProcess(input.GetMat(stream), input.timestamp, input.frame_number, stream);
    return input;
}

TS<SyncedMemory> GpuSink::process(TS<SyncedMemory> input, cv::cuda::Stream& stream)
{
    doProcess(input.GetGpuMat(stream), input.timestamp, input.frame_number, stream);
    return input;
}*/

