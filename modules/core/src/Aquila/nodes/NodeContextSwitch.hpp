#pragma once
#include <MetaObject/core/Context.hpp>
#include <MetaObject/core/Context.hpp>
#include <MetaObject/core/CvContext.hpp>
#include <MetaObject/params/TypeSelector.hpp>

namespace aq
{
namespace nodes
{

template <class Node>
struct NodeContextSwitch
{
    NodeContextSwitch(Node& node) : m_node(node) {}

    template <class CType, class... Args>
    void apply(mo::IAsyncStream* ctx_, bool* success, Args&&... args)
    {
        CType* ctx = dynamic_cast<CType*>(ctx_);
        *success = m_node.processImpl(ctx, std::forward<Args>(args)...);
    }

  private:
    Node& m_node;
};

template <class N, class... Args>
bool nodeContextSwitch(N* node, mo::IAsyncStream* ctx, Args&&... args)
{
    bool success = false;
    NodeContextSwitch<N> switcher(*node);
    mo::selectType<mo::IAsyncStreamypes>(switcher, ctx->context_type, ctx, &success, std::forward<Args>(args)...);
    return success;
}

} // namespace aq::nodes
} // namespace aq
