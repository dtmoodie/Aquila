#pragma once
#include <Aquila/detail/export.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

#include <map>
#include <memory>
#include <set>

#ifdef _WIN32
#ifdef _DEBUG
RUNTIME_COMPILER_LINKLIBRARY("aquila_guid.lib")
#else
RUNTIME_COMPILER_LINKLIBRARY("aquila_gui.lib")
#endif
#else
#ifdef NDEBUG
RUNTIME_COMPILER_LINKLIBRARY("-laquila_gui")
#else
RUNTIME_COMPILER_LINKLIBRARY("-laquila_guid")
#endif
#endif

namespace aq
{
    class IGraph;
    // Single instance per stream
    class AQUILA_EXPORTS WindowCallbackHandler : public TInterface<WindowCallbackHandler, mo::MetaObject>
    {
      public:
        enum
        {
            PAUSE_DRAG = 1 << 31
        };
        WindowCallbackHandler();

        void imshow(const std::string& window_name, cv::Mat img, int flags = 1);
        void imshowd(const std::string& window_name, cv::cuda::GpuMat img, int flags = cv::WINDOW_OPENGL);
        void imshowb(const std::string& window_name, cv::ogl::Buffer buffer, int flags = cv::WINDOW_OPENGL);
        void Init(bool firstInit);

        void setUiStream(std::shared_ptr<mo::IAsyncStream> stream);

        MO_BEGIN(WindowCallbackHandler)
            MO_SIGNAL(void, click_right, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, click_left, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, click_middle, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, move_mouse, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, click, std::string, cv::Point, int, cv::Mat)
            MO_SIGNAL(void, select_rect, std::string, cv::Rect, int, cv::Mat)
            MO_SIGNAL(void, select_points, std::string, std::vector<cv::Point>, int, cv::Mat)
            MO_SIGNAL(void, on_key, int)
            MO_SIGNAL(void, mouseDrag, std::string, cv::Point, cv::Point, int, cv::Mat)
        MO_END;

        struct AQUILA_EXPORTS EventLoop
        {
          public:
            static EventLoop* Instance();
            void Register(WindowCallbackHandler*);
            void run();

          private:
            EventLoop();
            ~EventLoop();
            std::vector<rcc::weak_ptr<WindowCallbackHandler>> m_handlers;
            std::mutex mtx;
        };

      protected:
        std::shared_ptr<mo::IAsyncStream> getUiStream();

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
        std::shared_ptr<mo::IAsyncStream> m_ui_stream;
    };
} // namespace aq
