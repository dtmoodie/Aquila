#pragma once
#include "Aquila/core/detail/Export.hpp"

#include <RuntimeObjectSystem/ObjectInterface.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <vector>

struct IObjectConstructor;

namespace aq
{
    class IGraph;
    namespace python
    {

        template <class T>
        struct GraphOwnerWrapper
        {
            typedef T element_type;
            GraphOwnerWrapper(const rcc::shared_ptr<T>& obj_ = rcc::shared_ptr<T>())
                : obj(obj_)
            {
            }
            ~GraphOwnerWrapper()
            {
                obj.reset();
                graph.reset();
            }
            operator rcc::shared_ptr<T>&()
            {
                return obj;
            }
            operator const rcc::shared_ptr<T>&() const
            {
                return obj;
            }

            rcc::shared_ptr<T> obj;
            rcc::shared_ptr<aq::IGraph> graph;
        };

        template <class T>
        T* get_pointer(const GraphOwnerWrapper<T>& wrapper)
        {
            return wrapper.obj.get();
        }

        template <class T>
        GraphOwnerWrapper<T> constructWrappedObject(IObjectConstructor* ctr)
        {
            rcc::shared_ptr<T> output;
            auto obj = ctr->Construct();
            if (obj)
            {
                output = obj;
                output->Init(true);
            }
            return output;
        }

        AQUILA_EXPORTS void setupNodeInterface();
        AQUILA_EXPORTS void setupNodeObjects(std::vector<IObjectConstructor*>& ctrs);
    } // namespace python
} // namespace aq
