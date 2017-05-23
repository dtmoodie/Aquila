#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "Aquila/rcc/external_includes/cv_core.hpp"
#include <MetaObject/params/traits/TypeTraits.hpp>
#include <opencv2/core/cuda.hpp>
#include <boost/optional.hpp>
#include <memory>

namespace aq{ class SyncedMemory; }
namespace mo{
    class Context;
    class SyncedMemoryTrait;

    template<> struct TraitSelector<aq::SyncedMemory, 2>{
        typedef SyncedMemoryTrait TraitType;
    };
}

namespace aq
{
    class AQUILA_EXPORTS SyncedMemory
    {
    public:
        enum SYNC_STATE
        {
            SYNCED = 0,
            HOST_UPDATED,
            DEVICE_UPDATED,
            DO_NOT_SYNC
        };
        SyncedMemory();
        SyncedMemory(const cv::Mat& h_mat, const cv::cuda::GpuMat& d_mat);
        SyncedMemory(const cv::Mat& h_mat);
        SyncedMemory(const cv::cuda::GpuMat& d_mat);
        SyncedMemory(const std::vector<cv::cuda::GpuMat> & d_mats);
        SyncedMemory(const std::vector<cv::Mat>& h_mats);
        SyncedMemory(cv::MatAllocator* cpu_allocator, cv::cuda::GpuMat::Allocator* gpu_allocator);
        SyncedMemory(const std::vector<cv::Mat>& h_mat, const std::vector<cv::cuda::GpuMat>& d_mat, SYNC_STATE state = SYNCED);
        SyncedMemory(const std::vector<cv::Mat>& h_mat, const std::vector<cv::cuda::GpuMat>& d_mat, const std::vector<SYNC_STATE> state);
        SyncedMemory clone(cv::cuda::Stream& stream);

        const cv::Mat&                         GetMat(cv::cuda::Stream& stream, int idx = 0) const;
        cv::Mat&                               GetMatMutable(cv::cuda::Stream& stream, int idx = 0);
        const cv::Mat&                         GetMatNoSync(int idx = 0) const;

        const cv::cuda::GpuMat&                GetGpuMat(cv::cuda::Stream& stream, int idx = 0) const;
        cv::cuda::GpuMat&                      GetGpuMatMutable(cv::cuda::Stream& stream, int idx = 0);
        const cv::cuda::GpuMat&                GetGpuMatNoSync(int idx = 0) const;

        const std::vector<cv::Mat>&            GetMatVec(cv::cuda::Stream& stream) const;
        std::vector<cv::Mat>&                  GetMatVecMutable(cv::cuda::Stream& stream);

        const std::vector<cv::cuda::GpuMat>&   GetGpuMatVec(cv::cuda::Stream& stream) const;
        std::vector<cv::cuda::GpuMat>&         GetGpuMatVecMutable(cv::cuda::Stream& stream);

        SYNC_STATE                             GetSyncState(int index = 0) const;

        mo::Context*                           getContext() const;
        void                                   setContext(mo::Context* ctx);

        bool Clone(cv::Mat& dest, cv::cuda::Stream& stream, int idx = 0) const;
        bool Clone(cv::cuda::GpuMat& dest, cv::cuda::Stream& stream, int idx = 0) const;

        void Synchronize(cv::cuda::Stream& stream = cv::cuda::Stream::Null()) const;
        void ResizeNumMats(int new_size = 1);
        void ReleaseGpu(cv::cuda::Stream& stream = cv::cuda::Stream::Null());

        int GetNumMats() const;
        bool empty() const;
        cv::Size GetSize() const;
        int GetChannels() const;
        std::vector<int> GetShape() const;
        int GetDim(int dim) const;
        int GetDepth() const;
        int GetType() const;
        int GetElemSize() const;
        template<typename A> void load(A& ar);
        template<typename A> void save(A & ar) const;
    private:
        struct impl
        {
            impl():
                _ctx(nullptr)
            {
            }
            std::vector<cv::Mat> h_data;
            std::vector<cv::cuda::GpuMat> d_data;
            std::vector<SyncedMemory::SYNC_STATE> sync_flags;
            mo::Context* _ctx;
        };

        std::shared_ptr<impl> _pimpl;
    };
}

namespace mo{
class SyncedMemoryTrait{
public:
    enum {
        REQUIRES_GPU_SYNC = 1,
        HAS_TRIVIAL_MOVE = 0
    };
    typedef aq::SyncedMemory Raw_t;
    typedef aq::SyncedMemory Storage_t; // Used in output wrapping parameters
    // Used by output parameters where the member is really just a reference to what
    // is owned as a Storage_t by the parameter
    typedef aq::SyncedMemory& TypeRef_t;
    typedef const aq::SyncedMemory& ConstTypeRef_t;

    // Pointer to typed stored by storage
    typedef aq::SyncedMemory* StoragePtr_t;
    typedef const aq::SyncedMemory* ConstStoragePtr_t;

    // Used when passing data around within a thread
    typedef const aq::SyncedMemory& ConstStorageRef_t;

    // Used for input parameters
    // Wrapping param storage
    typedef boost::optional<const aq::SyncedMemory> InputStorage_t;
    // User space input pointer, used in TInputParamPtr
    typedef const aq::SyncedMemory* Input_t;

    static inline Storage_t copy(const aq::SyncedMemory& value){
        return value;
    }
    static inline Storage_t clone(const aq::SyncedMemory& value){
        // TODO use built in stream or current threads stream
        return value.clone(cv::cuda::Stream());
    }

    template<class...Args>
    static aq::SyncedMemory& reset(Storage_t& input_storage, Args&&...args) {
        input_storage = aq::SyncedMemory(std::forward<Args>(args)...);
        return input_storage;
    }

    template<class...Args>
    static void reset(InputStorage_t& input_storage, Args&&...args) {
        input_storage = aq::SyncedMemory(std::forward<Args>(args)...);
    }
    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        input_storage.reset();
    }
    static inline aq::SyncedMemory& get(Storage_t& value){
        return value;
    }
    static inline const aq::SyncedMemory& get(const Storage_t& value){
        return value;
    }
    static inline aq::SyncedMemory* ptr(aq::SyncedMemory& value){
        return &value;
    }
    static inline const aq::SyncedMemory* ptr(const aq::SyncedMemory& value){
        return &value;
    }
};
}
