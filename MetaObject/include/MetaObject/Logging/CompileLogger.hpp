#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "ICompilerLogger.h"
#include "IRuntimeObjectSystem.h"
#include <cstdarg>
namespace mo
{
    class MO_EXPORTS CompileLogger: public ICompilerLogger
    {
    public:    
        virtual void LogError(const char * format, ...);
        virtual void LogWarning(const char * format, ...);
        virtual void LogInfo(const char * format, ...);
        virtual void LogDebug(const char * format, ...);

    protected:
        void LogInternal(int severity, const char * format, va_list args);
        static const size_t LOGSYSTEM_MAX_BUFFER = 409600;
        char m_buff[LOGSYSTEM_MAX_BUFFER];
    };

    class MO_EXPORTS BuildCallback: public ITestBuildNotifier
    {
        virtual bool TestBuildCallback(const char* file, TestBuildResult type);
        virtual bool TestBuildWaitAndUpdate();
    };

}
