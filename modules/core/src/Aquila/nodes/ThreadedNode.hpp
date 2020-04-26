#pragma once
#include "Aquila/nodes/Node.hpp"
#include <boost/thread.hpp>

namespace aq
{
    namespace nodes
    {
        class AQUILA_EXPORTS ThreadedNode : public Node
        {
          public:
            ThreadedNode();
            ~ThreadedNode();

            bool process();

            
            void addChild(Ptr child);

            MO_DERIVE(ThreadedNode, Node);
            MO_SLOT(void, stopThread);
            MO_SLOT(void, pauseThread);
            MO_SLOT(void, resumeThread);
            MO_SLOT(void, startThread);
            MO_END;

          protected:
            bool processImpl() { return true; }
          private:
            void processingFunction();
            mo::IAsyncStreamPtr_t _thread_ctx;
            boost::thread _processing_thread;
            bool _run;
        };
    }
}
