#pragma once
#include "Aquila/Detail/Export.hpp"
#include <MetaObject/IMetaObject.hpp>
#include <MetaObject/Detail/MetaObjectMacros.hpp>
#include <MetaObject/Signals/detail/SignalMacros.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/opengl.hpp>

#include <map>
#include <set>
#include <memory>

namespace aq
{
    class IDataStream;
    // Single instance per stream
    class AQUILA_EXPORTS WindowCallbackHandler: public TInterface<WindowCallbackHandler,mo::IMetaObject>
    {
    public:
        enum 
        {
            PAUSE_DRAG = 1 << 31
        };
        WindowCallbackHandler();

        void imshow(const std::string& window_name, cv::Mat img, int flags =  1);
        void imshowd(const std::string& window_name, cv::cuda::GpuMat img, int flags = cv::WINDOW_OPENGL);
        void imshowb(const std::string& window_name, cv::ogl::Buffer buffer, int flags = cv::WINDOW_OPENGL);
        void Init(bool firstInit);
        MO_BEGIN(WindowCallbackHandler)
            MO_SIGNAL(void, click_right, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, click_left, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, click_middle, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, move_mouse, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, click, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, select_rect, std::string, cv::Rect, int, cv::Mat)
            MO_SIGNAL(void, select_points, std::string, std::vector<cv::Point>, int, cv::Mat)
            MO_SIGNAL(void, on_key, int)
        MO_END

        struct AQUILA_EXPORTS EventLoop
        {
        public:
            static EventLoop* Instance();
            void Register(WindowCallbackHandler*);
            void run();
        private:
            EventLoop();
            ~EventLoop();
            std::vector<rcc::weak_ptr<WindowCallbackHandler>> handlers;
            std::mutex mtx;
        };
    private:
        static void on_mouse_click(int event, int x, int y, int flags, void* callback_handler);
        struct AQUILA_EXPORTS WindowHandler
        {
            WindowCallbackHandler* parent;
            bool dragging;
            cv::Point drag_start;
            std::vector<cv::Point> dragged_points;
            std::string win_name;
            cv::Mat displayed_image;
            void on_mouse(int event, int x, int y, int flags);
        };
        std::map<std::string, std::shared_ptr<WindowHandler>> windows;
    };
} // namespace aq
