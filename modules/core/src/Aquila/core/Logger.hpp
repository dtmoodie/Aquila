#pragma once


#include <boost/log/sinks/basic_sink_backend.hpp>
#include "Aquila/core/detail/Export.hpp"
#include <boost/log/trivial.hpp>
#include "RuntimeObjectSystem/ObjectInterface.h"
#include <MetaObject/signals/TSignal.hpp>

namespace aq
{
    namespace Nodes
    {
        class Node;
    }
    class AQUILA_EXPORTS ui_collector : public boost::log::sinks::basic_formatted_sink_backend<char, boost::log::sinks::concurrent_feeding>
    {
        typedef std::function<void(boost::log::trivial::severity_level, std::string, std::string)> object_log_handler_t;
        typedef std::function<void(boost::log::trivial::severity_level, std::string)> log_handler_t;
        object_log_handler_t object_log_handler;
        log_handler_t log_handler;
    public:
        ui_collector(object_log_handler_t olh = object_log_handler_t(), log_handler_t lh = log_handler_t());

        void consume(boost::log::record_view const& rec, string_type const& command_line);

        static mo::TSignal<void(boost::log::trivial::severity_level, std::string)>&                 get_object_log_handler(std::string);
        static mo::TSignal<void(boost::log::trivial::severity_level, std::string, std::string)>&    get_object_log_handler();
        static mo::TSignal<void(boost::log::trivial::severity_level, std::string)>&                 get_log_handler();

        static void set_node_name(std::string name);
    };
}
