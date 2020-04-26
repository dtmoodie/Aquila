#pragma once

#include <algorithm>
#include <vector>

// Sort value and key based on values, key will be reorganized such that

template <typename T>
void indexSort(const std::vector<T>& value, std::vector<size_t>& indecies, const bool ascending = true)
{
    indecies.resize(value.size());
    for (size_t i = 0; i < value.size(); ++i)
    {
        indecies[i] = i;
    }
    if(ascending)
    {
        std::sort(indecies.begin(), indecies.end(), [&value](size_t i1, size_t i2) { return value[i1] < value[i2]; });
    }else
    {
        std::sort(indecies.begin(), indecies.end(), [&value](size_t i1, size_t i2) { return value[i1] > value[i2]; });
    }
}

template <typename T>
std::vector<size_t> indexSort(const std::vector<T>& value, const bool ascending = true)
{
    std::vector<size_t> indecies;
    indexSort(value, indecies, ascending);
    return indecies;
}
