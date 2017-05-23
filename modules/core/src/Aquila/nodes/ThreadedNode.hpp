#pragma once
#include "Aquila/nodes/Node.hpp"
#include <boost/thread.hpp>

namespace aq
{
    namespace Nodes
    {
        class AQUILA_EXPORTS ThreadedNode: public Node
        {
        public:
            ThreadedNode();
            ~ThreadedNode();

            bool process();

            Node::Ptr addChild(Node* child);
            Node::Ptr addChild(Node::Ptr child);

            MO_DERIVE(ThreadedNode, Node);
                MO_SLOT(void, stopThread);
                MO_SLOT(void, pauseThread);
                MO_SLOT(void, resumeThread);
                MO_SLOT(void, startThread);
            MO_END;
        protected:
            bool processImpl(){ return true; }
        private:
            void processingFunction();
            std::unique_ptr<mo::Context> _thread_ctx;
            boost::thread _processing_thread;
            bool _run;
        };
    }
}
