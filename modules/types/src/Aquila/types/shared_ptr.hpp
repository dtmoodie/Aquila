#ifndef AQ_TYPES_SHARED_PTR_HPP
#define AQ_TYPES_SHARED_PTR_HPP
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/runtime_reflection/IDynamicVisitor.hpp>
#include <MetaObject/runtime_reflection/StructTraits.hpp>
#include <MetaObject/runtime_reflection/TraitInterface.hpp>
#include <MetaObject/runtime_reflection/visitor_traits/memory.hpp>

#include <ce/shared_ptr.hpp>

namespace mo
{
    template <class T>
    struct TTraits<ce::shared_ptr<T>, 5> : StructBase<ce::shared_ptr<T>>
    {

        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            size_t id = 0;
            auto val = this->ptr(inst);
            visitor(&id, "id");
            if (id != 0)
            {
                auto ptr = visitor.getPointer<T>(id);
                if (!ptr)
                {
                    PolymorphicSerializationHelper<T>::load(visitor, *val);
                    //*val = ce::shared_ptr<T>::create();
                    visitor(val->get(), "data");
                    visitor.setSerializedPointer(val->get(), id);
                    auto cache_ptr = *val;
                    visitor.pushCach(std::move(cache_ptr), std::string("shared_ptr ") + typeid(T).name(), id);
                }
                else
                {
                    auto cache_ptr =
                        visitor.popCache<std::shared_ptr<T>>(std::string("shared_ptr ") + typeid(T).name(), id);
                    if (cache_ptr)
                    {
                        *val = cache_ptr;
                    }
                }
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto val = this->ptr(inst);
            size_t id = 0;
            id = size_t(val->get());
            auto ptr = visitor.getPointer<T>(id);
            visitor(&id, "id");
            if (*val && ptr == nullptr)
            {
                visitor(val->get(), "data");
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("data");
        }
    };

    template <class T>
    struct TTraits<ce::shared_ptr<const T>, 5> : StructBase<ce::shared_ptr<const T>>
    {

        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            size_t id = 0;
            auto val = this->ptr(inst);
            visitor(&id, "id");
            if (id != 0)
            {
                auto ptr = visitor.getPointer<T>(id);
                if (ptr)
                {
                    auto cache_ptr =
                        visitor.popCache<std::shared_ptr<T>>(std::string("shared_ptr ") + typeid(T).name(), id);
                    if (cache_ptr)
                    {
                        *val = cache_ptr;
                    }
                }
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t cnt) const override
        {
            MO_ASSERT_EQ(cnt, 1);
            auto val = this->ptr(inst);
            size_t id = 0;
            id = size_t(val->get());
            auto ptr = visitor.getPointer<T>(id);
            visitor(&id, "id");
            if (*val && ptr == nullptr)
            {
                visitor(val->get(), "data");
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<T>("data");
        }
    };
} // namespace mo

namespace ce
{
    template <class AR, class T>
    void save(AR& ar, const ce::shared_ptr<T>& val)
    {
        std::shared_ptr<T> ptr = val;
        ar(ptr);
    }

    template <class AR, class T>
    void load(AR& ar, ce::shared_ptr<T>& val)
    {
        std::shared_ptr<T> ptr;
        ar(ptr);
        val = ptr;
    }
} // namespace ce

#endif // AQ_TYPES_SHARED_PTR_HPP
