#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "Aquila/rcc/external_includes/cv_core.hpp"
#include <MetaObject/params/traits/TypeTraits.hpp>
#include <opencv2/core/cuda.hpp>
#include <memory>

namespace aq{ class SyncedMemory; }
namespace mo{
    class Context;
    class SyncedMemoryTrait;
    class CvMatTrait;
    template<> struct TraitSelector<aq::SyncedMemory, 2>{
        typedef SyncedMemoryTrait TraitType;
    };
    template<> struct TraitSelector<cv::Mat, 2>{
        typedef CvMatTrait TraitType;
    };
}

namespace aq{
    class AQUILA_EXPORTS SyncedMemory{
    public:
        enum SYNC_STATE{
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

        SyncedMemory clone(cv::cuda::Stream& stream) const;

        const cv::Mat&                         getMat(cv::cuda::Stream& stream, int idx = 0) const;
        cv::Mat&                               getMatMutable(cv::cuda::Stream& stream, int idx = 0);
        const cv::Mat&                         getMatNoSync(int idx = 0) const;

        const cv::cuda::GpuMat&                getGpuMat(cv::cuda::Stream& stream, int idx = 0) const;
        cv::cuda::GpuMat&                      getGpuMatMutable(cv::cuda::Stream& stream, int idx = 0);
        const cv::cuda::GpuMat&                getGpuMatNoSync(int idx = 0) const;

        const std::vector<cv::Mat>&            getMatVec(cv::cuda::Stream& stream) const;
        std::vector<cv::Mat>&                  getMatVecMutable(cv::cuda::Stream& stream);

        const std::vector<cv::cuda::GpuMat>&   getGpuMatVec(cv::cuda::Stream& stream) const;
        std::vector<cv::cuda::GpuMat>&         getGpuMatVecMutable(cv::cuda::Stream& stream);

        SYNC_STATE                             getSyncState(int index = 0) const;

        mo::Context*                           getContext() const;
        void                                   setContext(mo::Context* ctx);

        bool clone(cv::Mat& dest, cv::cuda::Stream& stream, int idx = 0) const;
        bool clone(cv::cuda::GpuMat& dest, cv::cuda::Stream& stream, int idx = 0) const;

        void synchronize(cv::cuda::Stream& stream = cv::cuda::Stream::Null()) const;
        void resizeNumMats(int new_size = 1);
        void releaseGpu(cv::cuda::Stream& stream = cv::cuda::Stream::Null());

        int getNumMats() const;
        bool empty() const;
        cv::Size getSize() const;
        int getChannels() const;
        std::vector<int> getShape() const;
        int getDim(int dim) const;
        int getDepth() const;
        int getType() const;
        int getElemSize() const;
        template<typename A> void load(A& ar);
        template<typename A> void save(A & ar) const;
    private:
        struct impl{
            impl():_ctx(nullptr){}
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
    typedef aq::SyncedMemory InputStorage_t;
    // User space input pointer, used in TInputParamPtr
    typedef const aq::SyncedMemory* Input_t;

    static inline Storage_t copy(const aq::SyncedMemory& value){
        return value;
    }
    static inline Storage_t clone(const aq::SyncedMemory& value){
        // TODO use built in stream or current threads stream
        return value.clone(cv::cuda::Stream::Null());
    }

    template<class...Args>
    static aq::SyncedMemory& reset(Storage_t& input_storage, Args&&...args) {
        input_storage = aq::SyncedMemory(std::forward<Args>(args)...);
        return input_storage;
    }

    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        // TODO
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
class CvMatTrait {
public:
    enum {
        REQUIRES_GPU_SYNC = 0,
        HAS_TRIVIAL_MOVE = 0
    };
    typedef cv::Mat Raw_t;
    typedef cv::Mat Storage_t; // Used in output wrapping parameters
                                // Used by output parameters where the member is really just a reference to what
                                // is owned as a Storage_t by the parameter
    typedef cv::Mat& TypeRef_t;
    typedef const cv::Mat& ConstTypeRef_t;

    // Pointer to typed stored by storage
    typedef cv::Mat* StoragePtr_t;
    typedef const cv::Mat* ConstStoragePtr_t;

    // Used when passing data around within a thread
    typedef const cv::Mat& ConstStorageRef_t;

    // Used for input parameters
    // Wrapping param storage
    typedef cv::Mat InputStorage_t;
    // User space input pointer, used in TInputParamPtr
    typedef const cv::Mat* Input_t;

    static inline cv::Mat copy(const cv::Mat& value) {
        return value;
    }
    static inline cv::Mat clone(const cv::Mat& value) {
        return value.clone();
    }

    template<class...Args>
    static cv::Mat& reset(Storage_t& input_storage, Args&&...args) {
        input_storage = cv::Mat(std::forward<Args>(args)...);
        return input_storage;
    }

    template<class...Args>
    static void nullify(InputStorage_t& input_storage) {
        // TODO
        input_storage.release();
    }
    static inline cv::Mat& get(Storage_t& value) {
        return value;
    }
    static inline const cv::Mat& get(const Storage_t& value) {
        return value;
    }
    static inline cv::Mat* ptr(cv::Mat& value) {
        return &value;
    }
    static inline const cv::Mat* ptr(const cv::Mat& value) {
        return &value;
    }
}; // class CvMatTrait
}
