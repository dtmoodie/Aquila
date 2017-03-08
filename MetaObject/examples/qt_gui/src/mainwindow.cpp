#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "MetaObject/Parameters/UI/WidgetFactory.hpp"
#include "MetaObject/Parameters/UI/Qt/IParameterProxy.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"

// The following lines are commented out to demonstrate user interface instantiation in a different translation unit
// Since the instantiation library is included, instantiations of several types are registered with the full user
// interface code for those types.  Thus the following are not needed for those types.  However, not all types are
// included, so a few of the parameters will use the default met parameter
//#include "MetaObject/Parameters/UI/Qt/POD.hpp"
#ifdef HAVE_OPENCV
//#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#endif
//#include "MetaObject/Parameters/UI/Qt/Containers.hpp"

#include "MetaObject/Parameters/TypedParameter.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
//#include "MetaObject/Parameters/RangedParameter.hpp"
#include "instantiate.hpp"
using namespace mo;
MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    mo::instantiations::initialize();
    ui->setupUi(this);
    {
        /*auto param = new mo::RangedParameter<std::vector<float>>(0.0,20,"vector float");
        param->GetDataPtr()->push_back(15.0);
        param->GetDataPtr()->push_back(14.0);
        param->GetDataPtr()->push_back(13.0);
        param->GetDataPtr()->push_back(12.0);
        param->GetDataPtr()->push_back(11.0);
        parameters.push_back(std::shared_ptr<IParameter>(param));*/
    }
    {
        auto param = new mo::TypedParameter<std::vector<int>>("vector int");
        param->GetDataPtr()->push_back(15);
        param->GetDataPtr()->push_back(14);
        param->GetDataPtr()->push_back(13);
        param->GetDataPtr()->push_back(12);
        param->GetDataPtr()->push_back(11);
        parameters.push_back(std::shared_ptr<IParameter>(param));
    }
    {
#ifdef HAVE_OPENCV
        auto param = new mo::TypedParameter<cv::Point3f>("Point3f");
        param->GetDataPtr()->z = 15;
        parameters.push_back(std::shared_ptr<IParameter>(param));
#endif
    }
    {
        auto param = new TypedParameter<std::vector<std::pair<std::string, std::string>>>("Vector std::pair<std::string, std::string>");
        param->GetDataPtr()->push_back(std::pair<std::string, std::string>("asdf", "1234"));
        parameters.push_back(std::shared_ptr<IParameter>(param));
    }
    {
#ifdef HAVE_OPENCV
        std::shared_ptr<IParameter> param(new TypedParameterPtr<std::vector<cv::Point2f>>("Vector cv::Point2f", &testRefVec));
        testRefVec.push_back(cv::Point2f(0, 1));
        testRefVec.push_back(cv::Point2f(2, 3));
        testRefVec.push_back(cv::Point2f(4, 5));
        testRefVec.push_back(cv::Point2f(6, 7));
        testRefVec.push_back(cv::Point2f(8, 1));
        testRefVec.push_back(cv::Point2f(9, 1));
        testRefVec.push_back(cv::Point2f(10, 1));
        param->OnUpdate(nullptr);
        parameters.push_back(param);
    }
    {

        std::shared_ptr<IParameter> param(new mo::TypedParameterPtr<std::vector<cv::Scalar>>("Vector cv::Scalar", &testRefScalar));
        testRefScalar.push_back(cv::Scalar(0));
        testRefScalar.push_back(cv::Scalar(1));
        testRefScalar.push_back(cv::Scalar(2));
        testRefScalar.push_back(cv::Scalar(3));
        testRefScalar.push_back(cv::Scalar(4));
        testRefScalar.push_back(cv::Scalar(5));
        testRefScalar.push_back(cv::Scalar(6));
        testRefScalar.push_back(cv::Scalar::all(7));
        testRefScalar.push_back(cv::Scalar(8));
        param->OnUpdate(nullptr);
        parameters.push_back(param);
#endif
    }
    {
        auto param = new TypedParameter<int>("int");
        *param->GetDataPtr() = 10;
        parameters.push_back(std::shared_ptr<IParameter>(param));
    }
#ifdef HAVE_OPENCV
    {
        auto param = new TypedParameter<cv::Scalar>("scalar");
        parameters.push_back(std::shared_ptr<IParameter>(param));
    }
    {

        auto param = new TypedParameter<cv::Matx<double,4,4>>("Mat4x4d");
        parameters.push_back(std::shared_ptr<IParameter>(param));
    }
    {
        auto param = new TypedParameter<cv::Vec<double, 6>>("Vec6d");
        parameters.push_back(std::shared_ptr<IParameter>(param));
    }
#endif
    for (int i = 0; i < parameters.size(); ++i)
    {
        auto proxy = mo::UI::qt::WidgetFactory::Instance()->CreateProxy(parameters[i].get());
        ui->widgetLayout->addWidget(proxy->GetParameterWidget(this));
        proxies.push_back(proxy);
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::on_btnSerialize_clicked()
{
    /*{
        cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);
        for (int i = 0; i < parameters.size(); ++i)
        {
            Parameters::Persistence::cv::Serialize(&fs, parameters[i].get());
        }
    }
    {
        cv::FileStorage fs("test.yml", cv::FileStorage::READ);
        cv::FileNode node = fs.root();
        std::cout << node.name().c_str() << std::endl;
        for (int i = 0; i < parameters.size(); ++i)
        {
            Parameters::Persistence::cv::DeSerialize(&node, parameters[i].get());
        }
    }*/
    
}

