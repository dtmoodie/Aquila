#ifndef AQ_CORE_IPARAMETER_SYNCHRONIZER_HPP
#define AQ_CORE_IPARAMETER_SYNCHRONIZER_HPP

namespace mo
{
    class IPublisher;
}
namespace aq
{
    class IParameterSynchronizer
    {
      public:
        static std::unique_ptr<IParameterSynchronizer> create();
        using PublisherVec_t = mo::SmallVec<mo::IPublisher*, 20>;
        using Callback_s = void(const mo::Header&, const PublisherVec_t);

        virtual ~IParameterSynchronizer();
        virtual void setInputs(std::vector<mo::IPublisher*>) = 0;
        virtual void setCallback(std::function<Callback_s>) = 0;
    };
} // namespace aq

#endif // AQ_CORE_IPARAMETER_SYNCHRONIZER_HPP
