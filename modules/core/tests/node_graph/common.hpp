#pragma once

#include <Aquila/core.hpp>
#include <Aquila/core/IGraph.hpp>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/nodes/ThreadedNode.hpp>

#include "MetaObject/object/MetaObjectFactory.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/params/TSubscriberPtr.hpp"
#include "MetaObject/thread/ThreadPool.hpp"
#include <MetaObject/core/AsyncStreamFactory.hpp>
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/params.hpp>

#include <MetaObject/core/detail/allocator_policies/opencv.hpp>
#include <spdlog/sinks/stdout_sinks.h>

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <type_traits>

static bool timestamp_mode = true;
