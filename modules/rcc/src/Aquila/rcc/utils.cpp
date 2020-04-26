#include "utils.hpp"


namespace aq
{
    namespace rcc
    {
        std::vector<::rcc::shared_ptr<IGraph>> SystemTable::getGraphs()
        {
            return m_graphs;
        }

        void SystemTable::addGraph(const ::rcc::shared_ptr<IGraph>& graph)
        {
            m_graphs.push_back(graph);
        }
    }
}