#pragma once
#include <Aquila/core/detail/Export.hpp>
#include <MetaObject/core/SystemTable.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

namespace aq
{
class IGraph;
namespace rcc
{
struct AQUILA_EXPORTS SystemTable : public ::SystemTable
{
    std::vector<::rcc::shared_ptr<IGraph>> getGraphs();
    void addGraph(const ::rcc::shared_ptr<IGraph>& graph);

  protected:
    std::vector<::rcc::shared_ptr<IGraph>> m_graphs;
};
}
}
