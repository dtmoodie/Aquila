#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <memory>
#ifdef HAVE_OPENCV
#include <opencv2/core/types.hpp>
#endif
namespace mo
{
    class IParameter;
    namespace UI
    {
        namespace qt
        {
            class IParameterProxy;
        }
    }
}

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    public slots:
    void on_btnSerialize_clicked();
private:
    Ui::MainWindow *ui;
    std::vector<std::shared_ptr<mo::UI::qt::IParameterProxy>> proxies;
    std::vector<std::shared_ptr<mo::IParameter>> parameters;
#ifdef HAVE_OPENCV
    std::vector<cv::Point2f> testRefVec;
    std::vector<cv::Scalar> testRefScalar;
#endif
};

#endif // MAINWINDOW_H
