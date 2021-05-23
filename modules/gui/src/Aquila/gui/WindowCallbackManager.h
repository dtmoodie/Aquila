#pragma once
#include "Aquila/detail/export.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

namespace aq
{
    /*class WindowCallbackHandler;
    class SignalManager;
    class AQUILA_EXPORTS WindowCallbackHandler:
            public TInterface<ctcrc32("aq::WindowCallbackHandler"), mo::IMetaObject>
    {
    public:
        MO_BEGIN(WindowCallbackHandler)
            MO_SIGNAL(void, on_keypress, std::string, int)
            MO_SIGNAL(void, on_left, std::string)
            MO_SIGNAL(void, on_rect, std::string, int, cv::Rect2f)
            MO_SIGNAL(void, on_points, std::string, int, std::vector<cv::Point2f>)
        MO_END;

        void imshow(const std::string& name, const cv::Mat& mat);
        void imshow(const std::string& name, cv::cuda::GpuMat& mat, cv::cuda::Stream& stream);
        void imshow(const std::string& name, const SyncedMemory& mat, cv::cuda::Stream& stream);
    protected:

        struct WindowHandler
        {
        public:
            WindowCallbackHandler* _parent;
            std::vector<cv::Point> _points;
            cv::Rect _rect;
            cv::Size _image_size;
            std::string _window_name;
            bool _dragging = false;
            int _flags = 0;
            int _event = 0;
        protected:
            static void mouseCallback(int ev, int x, int y, int flags, void* userData);
        };

        std::map<std::string, std::shared_ptr<WindowHandler>> _handlers;
    };*/
}
