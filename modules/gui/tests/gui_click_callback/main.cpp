#include <Aquila/core.hpp>
#include <Aquila/gui.hpp>
#include <Aquila/types/CompressedImage.hpp>
#include <Aquila/types/IImageCompressor.hpp>
#include <Aquila/types/SyncedImage.hpp>

int main(int argc, char** argv)
{

    auto table = SystemTable::instance();
    mo::MetaObjectFactory::loadStandardPlugins();
    auto stream = mo::IAsyncStream::create();
    mo::IAsyncStream::setCurrent(stream);
    auto logger = table->getDefaultLogger();

    if (argc != 2)
    {
        logger->critical("Must pass in path to image file for test");
        return -1;
    }

    std::shared_ptr<const aq::CompressedImage> compressed = aq::CompressedImage::load(argv[1]);

    if (compressed == nullptr)
    {
        logger->critical("Unable to load image from {}", argv[1]);
        return -1;
    }

    boost::thread gui_thread = aq::gui::createGuiThread();

    rcc::shared_ptr<aq::WindowCallbackHandler> window_manager = aq::WindowCallbackHandler::create();
    if (window_manager == nullptr)
    {
        logger->critical("Unable to create window manager");
    }
    mo::TSignal<void(std::string, cv::Point, int, cv::Mat)>* signal =
        dynamic_cast<mo::TSignal<void(std::string, cv::Point, int, cv::Mat)>*>(window_manager->getSignal(
            "click_right", mo::TypeInfo::create<void(std::string, cv::Point, int, cv::Mat)>()));

    if (signal == nullptr)
    {
        logger->critical("Unable to get signal");
        return -1;
    }

    std::atomic<bool> signal_received;
    signal_received = false;
    auto cb = [&signal_received](std::string, cv::Point, int, cv::Mat) { signal_received = true; };

    mo::TSlot<void(std::string, cv::Point, int, cv::Mat)> slot(std::move(cb));
    auto connection = signal->connect(slot);

    aq::SyncedImage image;
    compressed->image(image);
    if (image.empty())
    {
        logger->critical("Unable to decompress image");
        return -1;
    }

    const cv::Mat mat = image.mat();
    window_manager->imshow("test", mat);
    while (!signal_received)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    logger->info("Signal received! Success!");
}
