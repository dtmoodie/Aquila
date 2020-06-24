#include <ct/types/opencv.hpp>

#include <Aquila/types/DetectionDescription.hpp>
#include <Aquila/types/ObjectDetection.hpp>
#include <Aquila/types/SyncedMemory.hpp>

#include <MetaObject/core/metaobject_config.hpp>
#include <MetaObject/params/detail/MetaParamImpl.hpp>
#include <MetaObject/python/PythonAllocator.hpp>

#include <cereal/types/string.hpp>

#ifdef MO_HAVE_PYTHON
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace mo
{
    namespace python
    {

        template <class T, size_t N>
        inline void convertFromPython(const boost::python::object& obj, ct::ArrayAdapter<T, N> result)
        {
            if (result.ptr)
            {
                for (size_t i = 0; i < N; ++i)
                {
                    boost::python::extract<T> extractor(obj[i]);
                    result.ptr[i] = extractor();
                }
            }
        }
    } // namespace python
} // namespace mo
#endif

#include <MetaObject/python/PythonAllocator.hpp>

namespace cereal
{
    template <class AR>
    void serialize(AR& ar, aq::Category const*& cat)
    {
    }
} // namespace cereal

#if MO_HAVE_PYTHON == 1

namespace ct
{
    template <>
    inline auto convertToPython(const aq::SyncedMemory& obj) -> boost::python::object
    {
        auto host = obj.host();
        return convertToPython(host);
    }

    template <>
    inline auto convertToPython(aq::Category* const& obj) -> boost::python::object
    {
        if (obj)
        {
            return boost::python::object(obj->name);
        }
        else
        {
            return boost::python::object();
        }
    }
} // namespace ct

namespace mo
{
    namespace python
    {

        template <class T, class Enable>
        struct ToPythonDataConverter;

        void registerSetupFunction(std::function<void(void)>&& func);

        template <>
        struct ToPythonDataConverter<aq::SyncedMemory, void>
        {
            ToPythonDataConverter(SystemTable* table, const char* name)
            {
                python::registerSetupFunction(std::bind(&ToPythonDataConverter<aq::SyncedMemory, void>::setup));
            }

            static void setup()
            {
                boost::python::class_<aq::SyncedMemory> bpobj("SyncedMemory");
                bpobj.def("asnumpy",
                          static_cast<boost::python::object (*)(const aq::SyncedMemory&)>(&ct::convertToPython));
            }
        };

        template <class T>
        struct ToPythonDataConverter<aq::TDetectedObjectSet<T>, void>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(
                    std::bind(&ToPythonDataConverter<aq::TDetectedObjectSet<T>, void>::setup));
            }

            static void setup()
            {
                std::string name = "DetectedObjectSet<" + TypeTable::instance()->typeToName(TypeInfo(typeid(T))) + ">";
                boost::python::class_<aq::TDetectedObjectSet<T>> bpobj(name.c_str());
                bpobj.def(boost::python::vector_indexing_suite<aq::TDetectedObjectSet<T>>());
            }
        };

        template <>
        struct ToPythonDataConverter<aq::CategorySet, void>
        {
            ToPythonDataConverter(SystemTable* /*table*/, const char* name)
            {
                python::registerSetupFunction(std::bind(&ToPythonDataConverter<aq::CategorySet, void>::setup));
            }

            static void setup()
            {
                boost::python::class_<aq::CategorySet> bpobj("CategorySet");
                bpobj.def(boost::python::vector_indexing_suite<aq::CategorySet>());
            }
        };
    } // namespace python
} // namespace mo
#endif

namespace aq
{

    namespace serialization
    {
        template <class T>
        using vector = std::vector<T>;

        void initModule(SystemTable* table)
        {
            INSTANTIATE_META_PARAM(SyncedMemory, table);
            INSTANTIATE_META_PARAM(vector<SyncedMemory>, table);
            INSTANTIATE_META_PARAM(cv::Mat, table);
            INSTANTIATE_META_PARAM(Category, table);
            INSTANTIATE_META_PARAM(Classification, table);
            INSTANTIATE_META_PARAM(DetectedObject, table);
            INSTANTIATE_META_PARAM(DetectedObjectSet, table);
            INSTANTIATE_META_PARAM(CategorySet, table);
        }
    } // namespace serialization
} // namespace aq
