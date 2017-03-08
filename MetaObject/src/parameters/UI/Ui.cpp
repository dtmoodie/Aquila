/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/parameters
*/
#include "MetaObject/Parameters/UI/UI.hpp"

/*using namespace mo::UI;



void InvalidCallbacks::invalidate(void* sender)
{
    std::lock_guard<std::mutex> lock(mtx);
    invalid_senders.push_back(sender);
}
bool InvalidCallbacks::check_valid(void* sender)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (std::find(invalid_senders.begin(), invalid_senders.end(), sender) == invalid_senders.end())
        return true;
    return false;
}
void InvalidCallbacks::clear()
{
    std::lock_guard<std::mutex> lock(mtx);
    invalid_senders.clear();
}
        
std::list<void*> InvalidCallbacks::invalid_senders;
std::mutex InvalidCallbacks::mtx;

UiCallbackService * UiCallbackService::Instance()
{
    static UiCallbackService instance;
    return &instance;   
}
void UiCallbackService::post(std::function<void()> f, std::pair<void*, mo::TypeInfo> source)
{
    if(user_thread_callback_service)
    {
        user_thread_callback_service(f, source);
        return;
    }
    if (user_thread_callback_notifier)
        user_thread_callback_notifier();
    io_queue.push(std::make_pair(source, f));
}
size_t UiCallbackService::queue_size()
{
    return io_queue.size();
}

void UiCallbackService::setCallback(std::function<void(std::function<void()>, std::pair<void*, mo::TypeInfo>)> f)
{
    Instance()->user_thread_callback_service = f;
}
void UiCallbackService::setCallback(std::function<void(void)>& f)
{
    Instance()->user_thread_callback_notifier = f;
}

void UiCallbackService::run()
{
    std::pair<std::pair<void*, mo::TypeInfo>, std::function<void(void)>> data;
    auto inst = Instance();
    while (inst->io_queue.try_pop(data))
    {
        if (InvalidCallbacks::check_valid(data.first.first))
        {
            data.second();
        }
    }
}


ProcessingThreadCallbackService* ProcessingThreadCallbackService::Instance()
{
    static ProcessingThreadCallbackService instance;
    return &instance;
}

void ProcessingThreadCallbackService::setCallback(std::function<void(std::function<void(void)>, std::pair<void*, mo::TypeInfo>)> f)
{
    Instance()->user_processing_thread_callback_function = f;
}


void ProcessingThreadCallbackService::run()
{
    std::pair<std::pair<void*, mo::TypeInfo>, std::function<void(void)>> data;
    auto inst = Instance();
    while (inst->io_queue.try_pop(data))
    {
        if (InvalidCallbacks::check_valid(data.first.first))
        {
            data.second();
        }
    }
}

void ProcessingThreadCallbackService::post(std::function<void(void)> f, std::pair<void*, mo::TypeInfo> source)
{
    auto instance = Instance();
    if (instance->user_processing_thread_callback_function)
    {
        instance->user_processing_thread_callback_function(f, source);
        return;
    }
    instance->io_queue.push(std::make_pair(source, f));
}*/