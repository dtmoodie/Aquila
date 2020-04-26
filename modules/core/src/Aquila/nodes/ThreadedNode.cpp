#include "Aquila/nodes/ThreadedNode.hpp"
#include "Aquila/nodes/NodeImpl.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include <MetaObject/core/AsyncStreamFactory.hpp>
#include <MetaObject/thread/ThreadInfo.hpp>

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

void ThreadedNode::addChild(Ptr child)
{
    Node::addChild(child);
    child->setStream(_thread_ctx);
}

void ThreadedNode::processingFunction()
{

    _thread_ctx = mo::AsyncStreamFactory::instance()->create("ThreadedNodeStream");
    while (!boost::this_thread::interruption_requested())
    {
        if (_run)
        {
            // TODO
            // mo::ThreadSpecificQueue::run(mo::getThisThread());
            // mo::Mutex_t::scoped_lock lock(getMutex());
            auto children = getChildren();
            for (auto& child : children)
            {
                child->process();
            }
        }
        else
        {
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        }
    }
}

MO_REGISTER_OBJECT(ThreadedNode);
