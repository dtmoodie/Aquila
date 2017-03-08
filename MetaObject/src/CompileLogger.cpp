#include "MetaObject/Logging/CompileLogger.hpp"
#include <MetaObject/Logging/Log.hpp>
#include <stdio.h>
using namespace mo;


void CompileLogger::LogError(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(3, format, args);
}
void CompileLogger::LogWarning(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(2, format, args);
}
void CompileLogger::LogInfo(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(1, format, args);
}
void CompileLogger::LogDebug(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    LogInternal(0, format, args);
}

void CompileLogger::LogInternal(int severity, const char * format, va_list args)
{
    vsnprintf(m_buff, LOGSYSTEM_MAX_BUFFER-1, format, args);
    // Make sure there's a limit to the amount of rubbish we can output
    m_buff[LOGSYSTEM_MAX_BUFFER-1] = '\0';
    for(int i = 0; i < LOGSYSTEM_MAX_BUFFER -1; ++i)
    {
        if(m_buff[i] == '\n')
            m_buff[i] = ' ';
    }
    if(severity == 0)
    {
        BOOST_LOG_TRIVIAL(debug) << m_buff;
    }
    if(severity == 1)
    {
        BOOST_LOG_TRIVIAL(info) << m_buff;
    }
    if(severity == 2)
    {
        BOOST_LOG_TRIVIAL(warning) << m_buff;
    }
    if(severity == 3)
    {
        BOOST_LOG_TRIVIAL(error) << m_buff;
    }
}

bool BuildCallback::TestBuildCallback(const char* file, TestBuildResult type)
{
    switch(type)
    {
    case TESTBUILDRRESULT_SUCCESS:
    {BOOST_LOG_TRIVIAL(info) << file << "TESTBUILDRRESULT_SUCCESS\n"; break;}
    case TESTBUILDRRESULT_NO_FILES_TO_BUILD:
    {BOOST_LOG_TRIVIAL(info) << file << "TESTBUILDRRESULT_NO_FILES_TO_BUILD\n"; break;}
    case TESTBUILDRRESULT_BUILD_FILE_GONE:
    {BOOST_LOG_TRIVIAL(info) << file << "TESTBUILDRRESULT_BUILD_FILE_GONE\n"; break;}
    case TESTBUILDRRESULT_BUILD_NOT_STARTED:
    {BOOST_LOG_TRIVIAL(info) << file << "TESTBUILDRRESULT_BUILD_NOT_STARTED\n"; break;}
    case TESTBUILDRRESULT_BUILD_FAILED:
    {BOOST_LOG_TRIVIAL(info) << file << "TESTBUILDRRESULT_BUILD_FAILED\n"; break;}
    case TESTBUILDRRESULT_OBJECT_SWAP_FAIL:
    {BOOST_LOG_TRIVIAL(info) << file << "TESTBUILDRRESULT_OBJECT_SWAP_FAIL\n"; break;}
    }
    return true;
}

bool BuildCallback::TestBuildWaitAndUpdate()
{
    return true;
}
