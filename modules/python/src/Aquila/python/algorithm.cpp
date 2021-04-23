#include "algorithm.hpp"
#include <Aquila/core/Algorithm.hpp>
#include <MetaObject/python/rcc_ptr.hpp>
#include <boost/python.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/signature.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace aq
{
    namespace python
    {
        std::vector<rcc::shared_ptr<aq::IAlgorithm>> getComponents(aq::IAlgorithm& alg)
        {
            auto cmps = alg.getComponents();
            std::vector<rcc::shared_ptr<aq::IAlgorithm>> ret;
            for (const auto& cmp : cmps)
            {
                auto shared = cmp.lock();
                if (shared)
                {
                    ret.push_back(shared);
                }
            }

            return ret;
        }

        void setupAlgorithmInterface()
        {
            static bool setup = false;
            if (setup)
            {
                return;
            }
            MO_LOG(info, "Registering IAlgorithm to python");
            boost::python::class_<aq::IAlgorithm,
                                  rcc::shared_ptr<aq::IAlgorithm>,
                                  boost::python::bases<mo::IMetaObject>,
                                  boost::noncopyable>
                bpobj("IAlgorithm", boost::python::no_init);
            bpobj.def("process", static_cast<bool (aq::IAlgorithm::*)()>(&aq::IAlgorithm::process));
            bpobj.def("process", static_cast<bool (aq::IAlgorithm::*)(mo::IAsyncStream&)>(&aq::IAlgorithm::process));
            bpobj.def("addComponent", &aq::IAlgorithm::addComponent);
            bpobj.def("getComponents", &getComponents);
            bpobj.def("setSyncParam", &aq::IAlgorithm::setSyncInput);

            boost::python::class_<std::vector<rcc::shared_ptr<aq::IAlgorithm>>> algvec("IAlgorithmVec",
                                                                                       boost::python::no_init);
            algvec.def(boost::python::vector_indexing_suite<std::vector<rcc::shared_ptr<aq::IAlgorithm>>>());
            setup = true;
        }

        void setupAlgorithmObjects(std::vector<IObjectConstructor*>&)
        {
            // auto info = dynamic_cast<aq::Algorithm>((*itr)->GetObjectInfo());
        }
    } // namespace python
} // namespace aq
