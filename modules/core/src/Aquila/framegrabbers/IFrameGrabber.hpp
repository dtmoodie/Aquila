#pragma once
#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/NodeInfo.hpp"

#include <MetaObject/core/AsyncStream.hpp>
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

namespace aq
{
    namespace nodes
    {
        class IFrameGrabber;
        class FrameGrabberInfo;
        class Grabber;
        class GrabberInfo;
    } // namespace nodes
} // namespace aq

namespace aq
{
    class IGraph;

    namespace nodes
    {
        // timeout used by frame grabbers in ms
        // TODO change to std::chrono::milliseconds
        using Timeout_t = int32_t;

        // Priority frame grabber for loading a data source
        using Priority_t = int32_t;

        class AQUILA_EXPORTS GrabberInfo : virtual public mo::IMetaObjectInfo
        {
          public:
            /*!
             * \brief canLoad used to determine if a certain object can load a given path
             * \param path to the data that should be loaded
             * \return the priority of this frame grabber based on the provided path
             */
            virtual Priority_t canLoad(const std::string& path) const;
            /*!
             * \brief listPaths lists all possible paths that this frame grabber is aware that it can load
             *        this is useful for enumerating devices on a system
             * \param paths that can be loaded with this frame grabber
             */
            virtual void listPaths(std::vector<std::string>& paths) const;
            /*!
             * \brief timeout for trying to load from this frame grabber in milliseconds
             * \return
             */
            virtual Timeout_t timeout() const;
        };

        class AQUILA_EXPORTS IGrabber : virtual public TInterface<IGrabber, Algorithm>
        {
          public:
            typedef GrabberInfo InterfaceInfo;
            typedef IGrabber Interface;
            typedef rcc::shared_ptr<IGrabber> Ptr;
            MO_DERIVE(IGrabber, Algorithm)
                PARAM(std::string, loaded_document, "")
            MO_END;

            virtual bool loadData(const std::string& path) = 0;
            virtual bool grab() = 0;

          protected:
            bool processImpl() override;
        };

        class AQUILA_EXPORTS FrameGrabberInfo : virtual public NodeInfo
        {
          public:
            /*!
             * \brief canLoadPath determines if the frame grabber associated with this info object can load an input
             * document
             * \param document is a string descibing a file / path / URI to load
             * \return 0 if the document cannot be loaded, priority of the frame grabber otherwise.  Higher value means
             * higher compatibility with this document
             */
            virtual Priority_t canLoadPath(const std::string& document) const;
            /*!
             * \brief loadTimeout returns the ms that should be allowed for the frame grabber's LoadFile function before
             * a timeout condition
             * \return timeout in ms
             */
            virtual Timeout_t loadTimeout() const;

            // Function used for listing what documents are available for loading, used in cases of connected devices to
            // list what
            // devices have been enumerated
            virtual std::vector<std::string> listLoadablePaths() const;

            std::string Print(IObjectInfo::Verbosity verbosity = IObjectInfo::INFO) const;
        };

        // Interface class for the base level of features frame grabber
        class AQUILA_EXPORTS IFrameGrabber : virtual public TInterface<IFrameGrabber, Node>
        {
          public:
            typedef FrameGrabberInfo InterfaceInfo;
            typedef IFrameGrabber Interface;

            static rcc::shared_ptr<IFrameGrabber> create(const std::string& doc,
                                                         const std::string& preferred_loader = "");
            // Returns all data sources that can be loaded with the name of the loader that can load it
            static std::vector<std::pair<std::string, std::string>> listAllLoadableDocuments();

            MO_DERIVE(IFrameGrabber, Node)
                MO_SIGNAL(void, update)
                MO_SLOT(void, restart)
                PARAM(std::vector<std::string>, loaded_document, {})
                PARAM_UPDATE_SLOT(loaded_document)
                MO_SLOT(bool, loadData, std::string)
                MO_SLOT(bool, loadData, std::vector<std::string>)
            MO_END;
        };

        class FrameGrabber : virtual public IFrameGrabber
        {
          public:
            static std::vector<std::string> listLoadablePaths();
            static Timeout_t loadTimeout();
            static Priority_t canLoadPath(const std::string& path);
            MO_DERIVE(FrameGrabber, IFrameGrabber)
                MO_SLOT(bool, loadData, std::string)
            MO_END;
            bool processImpl() override;
            void addComponent(const rcc::weak_ptr<IAlgorithm>& component) override;

          protected:
            std::vector<IGrabber::Ptr> _grabbers;
        };
    } // namespace nodes
} // namespace aq
