#include "Category.hpp"
#include "Classification.hpp"
#include <MetaObject/logging/logging.hpp>
#include <fstream>
#include <opencv2/imgproc.hpp>

namespace aq
{
    Category::Category(const std::string& name_, cv::Vec3b color_, int32_t parent_, unsigned int idx)
        : parent(parent_)
        , name(name_)
        , index(idx)
        , color(color_)
    {
    }

    const std::string& Category::getName() const
    {
        return name;
    }

    Classification Category::operator()() const
    {
        return Classification(this, 1.0);
    }

    Classification Category::operator()(double conf) const
    {
        return Classification(this, conf);
    }

    bool operator==(const Category& lhs, const Category& rhs)
    {
        return lhs.name == rhs.name;
    }

    CategorySet::CategorySet(const std::vector<std::string>& cats_,
                             const std::map<std::string, cv::Vec3b>& colormap_,
                             const std::vector<int>& /*tree_*/)
    {
        resize(cats_.size());
        for (size_t i = 0; i < cats_.size(); ++i)
        {
            (*this)[i].name = cats_[i];
            (*this)[i].index = static_cast<unsigned int>(i);
        }
        colorize(colormap_);
    }

    CategorySet::CategorySet(const std::string& cat_file,
                             const std::map<std::string, cv::Vec3b>& colormap,
                             const std::vector<int>& /*tree*/)
    {
        std::ifstream ifs(cat_file);
        MO_ASSERT_FMT(ifs.is_open(), "Unable to open {}", cat_file);
        std::string line;
        unsigned int count = 0;
        while (std::getline(ifs, line))
        {
            emplace_back(std::move(line));
            back().index = count;
            ++count;
        }
        colorize(colormap);
    }

    Category& CategorySet::operator()(const std::string& name)
    {
        for (size_t i = 0; i < size(); ++i)
        {
            if ((*this)[i].name == name)
            {
                return (*this)[i];
            }
        }
        THROW(warn, "No category found with name {}", name);
        return *static_cast<Category*>(nullptr);
    }

    const Category& CategorySet::operator()(const std::string& name) const
    {
        for (size_t i = 0; i < size(); ++i)
        {
            if ((*this)[i].name == name)
            {
                return (*this)[i];
            }
        }
        THROW(warn, "No category found with name {}", name);
        return *static_cast<Category*>(nullptr);
    }

    void CategorySet::colorize(const std::map<std::string, cv::Vec3b>& colormap)
    {
        cv::Mat wrap_color(
            static_cast<int>(size()), 1, CV_8UC3, &((*this)[0].color), static_cast<int>(sizeof(Category)));
        for (size_t i = 0; i < size(); ++i)
        {
            (*this)[i].color = cv::Vec3b(static_cast<uchar>(i * 180 / size()), 200, 255);
        }
        cv::cvtColor(wrap_color, wrap_color, cv::COLOR_HSV2BGR);
        for (const auto& itr : colormap)
        {
            (*this)(itr.first).color = itr.second;
        }
    }

    void CategorySet::hierarchy(const std::vector<int>& tree)
    {
        for (size_t i = 0; i < tree.size(); ++i)
        {
            if (tree[i] >= 0)
            {
                (*this)[i].parent = i;
            }
        }
    }

} // namespace aq
