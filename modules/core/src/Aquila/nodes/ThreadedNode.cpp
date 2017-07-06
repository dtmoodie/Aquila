#include "Aquila/nodes/ThreadedNode.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include <MetaObject/thread/InterThread.hpp>
#include <MetaObject/thread/boost_thread.hpp>

using namespace aq;
using namespace aq::nodes;

ThreadedNode::ThreadedNode()
{
    _run = false;
    startThread();
    //_thread_context = _processing_thread;
    _run = true;
}

ThreadedNode::~ThreadedNode()
{
    stopThread();
}

void ThreadedNode::stopThread()
{
    _processing_thread.interrupt();
    _processing_thread.join();
}

void ThreadedNode::pauseThread()
{
    _run = false;
}

void ThreadedNode::resumeThread()
{
    _run = true;
}

void ThreadedNode::startThread()
{
    _processing_thread = boost::thread(boost::bind(&ThreadedNode::processingFunction, this));
}

bool ThreadedNode::process()
{
    return true;
}

Node::Ptr ThreadedNode::addChild(Node* child)
{
    auto ptr = Node::addChild(child);
    child->setContext(_thread_ctx, true);
    return ptr;
}

Node::Ptr ThreadedNode::addChild(Node::Ptr child)
{
    auto ptr = Node::addChild(child);
    child->setContext(_thread_ctx);
    return ptr;
}

void ThreadedNode::processingFunction()
{
    _thread_ctx = mo::Context::create("ThreadedNodeContext");
    while(!boost::this_thread::interruption_requested())
    {
        if(_run)
        {
            mo::ThreadSpecificQueue::run(mo::getThisThread());
            mo::Mutex_t::scoped_lock lock(*_mtx);
            for(auto& child : _children)
            {
                child->process();
            }
        }else
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        }
    }
}

MO_REGISTER_OBJECT(ThreadedNode);
