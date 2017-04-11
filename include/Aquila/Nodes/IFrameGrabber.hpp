#pragma once
#include "Aquila/SyncedMemory.h"
#include "Aquila/Nodes/Node.h"
#include "Aquila/Nodes/NodeInfo.hpp"
#include "IObject.h"
#include "IObjectInfo.h"


#include <MetaObject/Signals/detail/SlotMacros.hpp>
#include <MetaObject/Parameters/ParameterMacros.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Signals/detail/SignalMacros.hpp>
#include <MetaObject/Context.hpp>
#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/Thread/ThreadHandle.hpp>
#include <MetaObject/Thread/ThreadPool.hpp>

#include <RuntimeInclude.h>
#include <RuntimeSourceDependency.h>
#include <shared_ptr.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>


#include <atomic>
#include <string>

RUNTIME_MODIFIABLE_INCLUDE;
RUNTIME_COMPILER_SOURCEDEPENDENCY_FILE("../../../src/Aquila/Nodes/IFrameGrabber", ".cpp");
namespace aq
{
    namespace Nodes
    {
        class IFrameGrabber;
        class FrameGrabberInfo;
        class Grabber;
        class GrabberInfo;
    }
}

namespace aq
{
    class IDataStream;
    class ICoordinateManager;
    namespace Nodes
    {
    
    class AQUILA_EXPORTS GrabberInfo : virtual public mo::IMetaObjectInfo
    {
    public:
        virtual int CanLoad(const std::string& path) const;
        virtual void ListPaths(std::vector<std::string>& paths) const;
        virtual int Timeout() const;
    };
    class AQUILA_EXPORTS IGrabber : public TInterface<ctcrc32("aq::Nodes::IGrabber"), Algorithm>
    {
    public:
        typedef GrabberInfo InterfaceInfo;
        typedef IGrabber Interface;
        typedef rcc::shared_ptr<IGrabber> Ptr;
        MO_BEGIN(IGrabber)
            PARAM(std::string, loaded_document, "")
        MO_END;

        virtual bool Load(const std::string& path) = 0;
        virtual bool Grab() = 0;
    protected:
        bool ProcessImpl();
    };

    class AQUILA_EXPORTS FrameGrabberInfo: virtual public NodeInfo
    {
    public:
        /*!
         * \brief CanLoadDocument determines if the frame grabber associated with this info object can load an input document
         * \param document is a string descibing a file / path / URI to load
         * \return 0 if the document cannot be loaded, priority of the frame grabber otherwise.  Higher value means higher compatibility with this document
         */
        virtual int CanLoadPath(const std::string& document) const;
        /*!
         * \brief LoadTimeout returns the ms that should be allowed for the frame grabber's LoadFile function before a timeout condition
         * \return timeout in ms
         */
        virtual int LoadTimeout() const;

        // Function used for listing what documents are available for loading, used in cases of connected devices to list what
        // devices have been enumerated
        virtual std::vector<std::string> ListLoadablePaths() const;

        std::string Print() const;
    };

    // Interface class for the base level of features frame grabber
    class AQUILA_EXPORTS IFrameGrabber: virtual public TInterface<ctcrc32("aq::Nodes::FrameGrabber"), Node>
    {
    public:
        typedef FrameGrabberInfo InterfaceInfo;
        typedef IFrameGrabber Interface;

        static rcc::shared_ptr<IFrameGrabber> Create(const std::string& doc, const std::string& preferred_loader = "");
        // Returns all data sources that can be loaded with the name of the loader that can load it
        static std::vector<std::pair<std::string, std::string>> ListAllLoadableDocuments();
        
        MO_DERIVE(IFrameGrabber, Node)
            MO_SIGNAL(void, update)
            MO_SLOT(void, Restart)
            PARAM(std::vector<std::string>, loaded_document, {})
            PARAM_UPDATE_SLOT(loaded_document)
            MO_SLOT(bool, Load, std::string)
            MO_SLOT(bool, Load, std::vector<std::string>)
        MO_END;
    };
    } // namespace nodes
} // namespace aq
