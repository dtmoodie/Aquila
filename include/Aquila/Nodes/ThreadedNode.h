#pragma once
#include "Aquila/Nodes/Node.h"
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
            
            bool Process();
            
            Node::Ptr AddChild(Node* child);
            Node::Ptr AddChild(Node::Ptr child);

            MO_DERIVE(ThreadedNode, Node);
                MO_SLOT(void, StopThread);
                MO_SLOT(void, PauseThread);
                MO_SLOT(void, ResumeThread);
                MO_SLOT(void, StartThread);
            MO_END;
        protected:
            bool ProcessImpl(){ return true; }
        private:
            void processingFunction();
            mo::Context _thread_context;
            boost::thread _processing_thread;
            bool _run;
        };
    }
}
