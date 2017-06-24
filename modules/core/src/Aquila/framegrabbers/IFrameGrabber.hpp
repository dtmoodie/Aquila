#pragma once
#include "IObject.h"
#include "IObjectInfo.h"

#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/NodeInfo.hpp"

#include <MetaObject/core/Context.hpp>
#include <MetaObject/object/IMetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>
#include <MetaObject/signals/detail/SlotMacros.hpp>
#include <MetaObject/thread/ThreadHandle.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/thread.hpp>

#include <atomic>
#include <string>

RUNTIME_MODIFIABLE_INCLUDE
RUNTIME_COMPILER_SOURCEDEPENDENCY

namespace aq {
namespace nodes {
    class IFrameGrabber;
    class FrameGrabberInfo;
    class Grabber;
    class GrabberInfo;
}
}

namespace aq {
class IDataStream;
namespace nodes {

    class AQUILA_EXPORTS GrabberInfo : virtual public mo::IMetaObjectInfo {
    public:
        virtual int canLoad(const std::string& path) const;
        virtual void listPaths(std::vector<std::string>& paths) const;
        virtual int timeout() const;
    };
    class AQUILA_EXPORTS IGrabber : public TInterface<IGrabber, Algorithm> {
    public:
        typedef GrabberInfo InterfaceInfo;
        typedef IGrabber Interface;
        typedef rcc::shared_ptr<IGrabber> Ptr;
        MO_BEGIN(IGrabber)
        PARAM(std::string, loaded_document, "")
        MO_END;

        virtual bool loadData(const std::string& path) = 0;
        virtual bool grab() = 0;

    protected:
        bool processImpl();
    };

    class AQUILA_EXPORTS FrameGrabberInfo : virtual public NodeInfo {
    public:
        /*!
         * \brief canLoadPath determines if the frame grabber associated with this info object can load an input document
         * \param document is a string descibing a file / path / URI to load
         * \return 0 if the document cannot be loaded, priority of the frame grabber otherwise.  Higher value means higher compatibility with this document
         */
        virtual int canLoadPath(const std::string& document) const;
        /*!
         * \brief loadTimeout returns the ms that should be allowed for the frame grabber's LoadFile function before a timeout condition
         * \return timeout in ms
         */
        virtual int loadTimeout() const;

        // Function used for listing what documents are available for loading, used in cases of connected devices to list what
        // devices have been enumerated
        virtual std::vector<std::string> listLoadablePaths() const;

        std::string Print(IObjectInfo::Verbosity verbosity = IObjectInfo::INFO) const;
    };

    // Interface class for the base level of features frame grabber
    class AQUILA_EXPORTS IFrameGrabber : virtual public TInterface<IFrameGrabber, Node> {
    public:
        typedef FrameGrabberInfo InterfaceInfo;
        typedef IFrameGrabber Interface;

        static rcc::shared_ptr<IFrameGrabber> create(const std::string& doc, const std::string& preferred_loader = "");
        // Returns all data sources that can be loaded with the name of the loader that can load it
        static std::vector<std::pair<std::string, std::string> > listAllLoadableDocuments();

        MO_DERIVE(IFrameGrabber, Node)
        MO_SIGNAL(void, update)
        MO_SLOT(void, Restart)
        PARAM(std::vector<std::string>, loaded_document, {})
        PARAM_UPDATE_SLOT(loaded_document)
        MO_SLOT(bool, loadData, std::string)
        MO_SLOT(bool, loadData, std::vector<std::string>)
        MO_END;
    };
} // namespace nodes
} // namespace aq
