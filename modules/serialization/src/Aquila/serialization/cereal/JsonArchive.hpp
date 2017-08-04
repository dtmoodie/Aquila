#pragma once
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>

#include <Aquila/nodes/Node.hpp>
#include <Aquila/core/IDataStream.hpp>

#include <MetaObject/serialization/SerializationFactory.hpp>
#include <MetaObject/params/buffers/IBuffer.hpp>
#include <MetaObject/params/buffers/BufferFactory.hpp>
#include <MetaObject/params/ParamFactory.hpp>
#include <boost/lexical_cast.hpp>

namespace aq
{
    struct InputInfo
    {
        std::string name;
        std::string type;
        bool sync = false;
        int buffer_size = -1;
        boost::optional<mo::Time_t> buffer_time;

        template<class AR> void load(AR& ar)
        {
            ar(CEREAL_NVP(name));
            ar(CEREAL_OPTIONAL_NVP(sync, false));
            ar(CEREAL_OPTIONAL_NVP(buffer_size, -1));
            mo::Time_t buf_time;
            auto bt = cereal::make_optional_nvp("buffer_time", buf_time, buf_time);
            ar(bt);
            if(bt.success)
                buffer_time = bt.value;
            ar(CEREAL_OPTIONAL_NVP(type, "Direct"));
        }
        template<class AR> void save(AR& ar) const
        {
            ar(CEREAL_NVP(name));
            if(sync)
                ar(CEREAL_NVP(sync));
            if(buffer_size != -1)
                ar(CEREAL_NVP(buffer_size));
            if(type != "Direct")
                ar(CEREAL_NVP(type));
        }
    };
    class AQUILA_EXPORTS JSONOutputArchive : public cereal::JSONOutputArchive{
        const std::map<std::string, std::string>& _vm;
        const std::map<std::string, std::string>& _sm;
        bool _writing_defaults = false;
    public:
        std::string preset;
        JSONOutputArchive(std::ostream & stream,
                          Options const & options = Options::Default(),
                          const std::map<std::string, std::string>& vm = std::map<std::string, std::string>(),
                          const std::map<std::string, std::string>& sm = std::map<std::string, std::string>(),
                          const std::string preset_ = "Default") :
            cereal::JSONOutputArchive(stream, options),
            _vm(vm),
            _sm(sm),
            preset(preset_)
        {
            _writing_defaults = true;
            if(sm.size())
                (*this)(cereal::make_nvp("DefaultStrings", sm));
            if(vm.size())
                (*this)(cereal::make_nvp("DefaultVariables", vm));
            _writing_defaults = false;
        }

        //! Destructor, flushes the JSON
        ~JSONOutputArchive() CEREAL_NOEXCEPT
        {

        }

        //! Saves some binary data, encoded as a base64 string, with an optional name
        /*! This will create a new node, optionally named, and insert a value that consists of
        the data encoded as a base64 string */
        void saveBinaryValue(const void * data, size_t size, const char * name = nullptr)
        {
            setNextName(name);
            writeName();

            auto base64string = cereal::base64::encode(reinterpret_cast<const unsigned char *>(data), size);
            saveValue(base64string);
        }

        void startNode()
        {
            writeName();
            itsNodeStack.push(NodeType::StartObject);
            itsNameCounter.push(0);
        }

        //! Designates the most recently added node as finished
        void finishNode()
        {
            switch (itsNodeStack.top())
            {
            case NodeType::StartArray:
                itsWriter.StartArray();
            case NodeType::InArray:
                itsWriter.EndArray();
                break;
            case NodeType::StartObject:
                itsWriter.StartObject();
            case NodeType::InObject:
                itsWriter.EndObject();
                break;
            }

            itsNodeStack.pop();
            itsNameCounter.pop();
        }

        //! Sets the name for the next node created with startNode
        void setNextName(const char * name)
        {
            itsNextName = name;
        }

        //! Saves a bool to the current node
        void saveValue(bool b) { itsWriter.Bool(b); }
        //! Saves an int to the current node
        void saveValue(int i) { itsWriter.Int(i); }
        //! Saves a uint to the current node
        void saveValue(unsigned u) { itsWriter.Uint(u); }
        //! Saves an int64 to the current node
        void saveValue(int64_t i64) { itsWriter.Int64(i64); }
        //! Saves a uint64 to the current node
        void saveValue(uint64_t u64) { itsWriter.Uint64(u64); }
        //! Saves a double to the current node
        void saveValue(double d) { itsWriter.Double(d); }
        //! Saves a string to the current node
        void saveValue(std::string const & s) {
            if(!_writing_defaults)
            {
                for(const auto& str : _sm)
                {
                    if(str.second.empty())
                        continue;
                    auto pos = s.find(str.second);
                    if(pos != std::string::npos)
                    {
                        std::string write_str = s.substr(0, pos) + "${" + str.first + "}" + s.substr(pos + str.second.size());
                        itsWriter.String(write_str.c_str(), static_cast<rapidjson::SizeType>(write_str.size()));
                        return;
                    }
                }
            }
            itsWriter.String(s.c_str(), static_cast<rapidjson::SizeType>(s.size()));
        }
        //! Saves a const char * to the current node
        void saveValue(char const * s) { itsWriter.String(s); }
        //! Saves a nullptr to the current node
        void saveValue(std::nullptr_t) { itsWriter.Null(); }

    protected:
        // Some compilers/OS have difficulty disambiguating the above for various flavors of longs, so we provide
        // special overloads to handle these cases.

        //! 32 bit signed long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) == sizeof(std::int32_t),
            std::is_signed<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T l) { saveValue(static_cast<std::int32_t>(l)); }

        //! non 32 bit signed long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) != sizeof(std::int32_t),
            std::is_signed<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T l) { saveValue(static_cast<std::int64_t>(l)); }

        //! 32 bit unsigned long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) == sizeof(std::int32_t),
            std::is_unsigned<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T lu) { saveValue(static_cast<std::uint32_t>(lu)); }

        //! non 32 bit unsigned long saving to current node
        template <class T, cereal::traits::EnableIf<sizeof(T) != sizeof(std::int32_t),
            std::is_unsigned<T>::value> = cereal::traits::sfinae> inline
            void saveLong(T lu) { saveValue(static_cast<std::uint64_t>(lu)); }

    public:
#ifdef _MSC_VER
        //! MSVC only long overload to current node
        void saveValue(unsigned long lu) { saveLong(lu); };
#else // _MSC_VER
        //! Serialize a long if it would not be caught otherwise
        template <class T, cereal::traits::EnableIf<std::is_same<T, long>::value,
            !std::is_same<T, std::int32_t>::value,
            !std::is_same<T, std::int64_t>::value> = cereal::traits::sfinae> inline
            void saveValue(T t) { saveLong(t); }

        //! Serialize an unsigned long if it would not be caught otherwise
        template <class T, cereal::traits::EnableIf<std::is_same<T, unsigned long>::value,
            !std::is_same<T, std::uint32_t>::value,
            !std::is_same<T, std::uint64_t>::value> = cereal::traits::sfinae> inline
            void saveValue(T t) { saveLong(t); }
#endif // _MSC_VER

        //! Save exotic arithmetic as strings to current node
        /*! Handles long long (if distinct from other types), unsigned long (if distinct), and long double */
        template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value,
            !std::is_same<T, long>::value,
            !std::is_same<T, unsigned long>::value,
            !std::is_same<T, std::int64_t>::value,
            !std::is_same<T, std::uint64_t>::value,
            (sizeof(T) >= sizeof(long double) || sizeof(T) >= sizeof(long long))> = cereal::traits::sfinae> inline
            void saveValue(T const & t)
        {
            std::stringstream ss; ss.precision(std::numeric_limits<long double>::max_digits10);
            ss << t;
            saveValue(ss.str());
        }

        //! Write the name of the upcoming node and prepare object/array state
        /*! Since writeName is called for every value that is output, regardless of
        whether it has a name or not, it is the place where we will do a deferred
        check of our node state and decide whether we are in an array or an object.

        The general workflow of saving to the JSON archive is:

        1. (optional) Set the name for the next node to be created, usually done by an NVP
        2. Start the node
        3. (if there is data to save) Write the name of the node (this function)
        4. (if there is data to save) Save the data (with saveValue)
        5. Finish the node
        */
        void writeName()
        {
            NodeType const & nodeType = itsNodeStack.top();

            // Start up either an object or an array, depending on state
            if (nodeType == NodeType::StartArray)
            {
                itsWriter.StartArray();
                itsNodeStack.top() = NodeType::InArray;
            }
            else if (nodeType == NodeType::StartObject)
            {
                itsNodeStack.top() = NodeType::InObject;
                itsWriter.StartObject();
            }

            // Array types do not output names
            if (nodeType == NodeType::InArray) return;

            if (itsNextName == nullptr)
            {
                std::string name = "value" + std::to_string(itsNameCounter.top()++) + "\0";
                saveValue(name);
            }
            else
            {
                saveValue(itsNextName);
                itsNextName = nullptr;
            }
        }

        //! Designates that the current node should be output as an array, not an object
        void makeArray()
        {
            itsNodeStack.top() = NodeType::StartArray;
        }

        //! @}

    protected:
    }; // JSONOutputArchive

       // ######################################################################
       //! An input archive designed to load data from JSON
       /*! This archive uses RapidJSON to read in a JSON archive.

       As with the output JSON archive, the preferred way to use this archive is in
       an RAII fashion, ensuring its destruction after all data has been read.

       Input JSON should have been produced by the JSONOutputArchive.  Data can
       only be added to dynamically sized containers (marked by JSON arrays) -
       the input archive will determine their size by looking at the number of child nodes.
       Only JSON originating from a JSONOutputArchive is officially supported, but data
       from other sources may work if properly formatted.

       The JSONInputArchive does not require that nodes are loaded in the same
       order they were saved by JSONOutputArchive.  Using name value pairs (NVPs),
       it is possible to load in an out of order fashion or otherwise skip/select
       specific nodes to load.

       The default behavior of the input archive is to read sequentially starting
       with the first node and exploring its children.  When a given NVP does
       not match the read in name for a node, the archive will search for that
       node at the current level and load it if it exists.  After loading an out of
       order node, the archive will then proceed back to loading sequentially from
       its new position.

       Consider this simple example where loading of some data is skipped:

       @code{cpp}
       // imagine the input file has someData(1-9) saved in order at the top level node
       ar( someData1, someData2, someData3 );        // XML loads in the order it sees in the file
       ar( cereal::make_nvp( "hello", someData6 ) ); // NVP given does not
       // match expected NVP name, so we search
       // for the given NVP and load that value
       ar( someData7, someData8, someData9 );        // with no NVP given, loading resumes at its
       // current location, proceeding sequentially
       @endcode

       \ingroup Archives */
    class JSONInputArchive : public cereal::JSONInputArchive
    {
    protected:
        using ReadStream = rapidjson::IStreamWrapper;
        typedef rapidjson::GenericValue<rapidjson::UTF8<>> JSONValue;
        typedef JSONValue::ConstMemberIterator MemberIterator;
        typedef JSONValue::ConstValueIterator ValueIterator;
        typedef rapidjson::Document::GenericValue GenericValue;

    public:
        std::map<std::string, std::map<std::string, InputInfo>> input_mappings;
        std::map<std::string, std::vector<std::string>> parent_mappings;
        const std::map<std::string, std::string>& variable_replace_mapping;
        const std::map<std::string, std::string>& string_replace_mapping;
        std::string preset;
        /*! @name Common Functionality
        Common use cases for directly interacting with an JSONInputArchive */
        //! @{

        //! Construct, reading from the provided stream
        /*! @param stream The stream to read from */
        JSONInputArchive(std::istream & stream, const std::map<std::string, std::string>& vm, const std::map<std::string, std::string>& sm, const std::string& preset_ = "Default") :
            cereal::JSONInputArchive(stream),
            variable_replace_mapping(vm),
            string_replace_mapping(sm),
            preset(preset_)
        {

        }

        ~JSONInputArchive() CEREAL_NOEXCEPT = default;

        //! Loads some binary data, encoded as a base64 string
        /*! This will automatically start and finish a node to load the data, and can be called directly by
        users.

        Note that this follows the same ordering rules specified in the class description in regards
        to loading in/out of order */
        void loadBinaryValue(void * data, size_t size, const char * name = nullptr)
        {
            itsNextName = name;

            std::string encoded;
            loadValue(encoded);
            auto decoded = cereal::base64::decode(encoded);

            if (size != decoded.size())
                throw cereal::Exception("Decoded binary data size does not match specified size");

            std::memcpy(data, decoded.data(), decoded.size());
            itsNextName = nullptr;
        };



        //! Searches for the expectedName node if it doesn't match the actualName
        /*! This needs to be called before every load or node start occurs.  This function will
        check to see if an NVP has been provided (with setNextName) and if so, see if that name matches the actual
        next name given.  If the names do not match, it will search in the current level of the JSON for that name.
        If the name is not found, an exception will be thrown.

        Resets the NVP name after called.

        @throws Exception if an expectedName is given and not found */
        inline bool search()
        {
            // The name an NVP provided with setNextName()
            if (itsNextName)
            {
                // The actual name of the current node
                auto const actualName = itsIteratorStack.back().name();

                // Do a search if we don't see a name coming up, or if the names don't match
                if (!actualName || std::strcmp(itsNextName, actualName) != 0) {
                    bool nameFound = itsIteratorStack.back().search(itsNextName, itsNextOptional);
                    if (!nameFound && itsNextOptional) {
                        itsLoadOptional = true;
                        itsNextName = nullptr;
                        return false;
                    }else
                    {
                        itsNextName = nullptr;
                        return true;
                    }

                }
            }

            itsNextName = nullptr;
            return true;
        }

    public:
        //! Starts a new node, going into its proper iterator
        /*! This places an iterator for the next node to be parsed onto the iterator stack.  If the next
        node is an array, this will be a value iterator, otherwise it will be a member iterator.

        By default our strategy is to start with the document root node and then recursively iterate through
        all children in the order they show up in the document.
        We don't need to know NVPs to do this; we'll just blindly load in the order things appear in.

        If we were given an NVP, we will search for it if it does not match our the name of the next node
        that would normally be loaded.  This functionality is provided by search(). */
        void startNode()
        {
            search();

            if (itsIteratorStack.back().value().IsArray())
                itsIteratorStack.emplace_back(itsIteratorStack.back().value().Begin(), itsIteratorStack.back().value().End());
            else
                itsIteratorStack.emplace_back(itsIteratorStack.back().value().MemberBegin(), itsIteratorStack.back().value().MemberEnd());
        }

        //! Finishes the most recently started node
        void finishNode()
        {
            itsIteratorStack.pop_back();
            ++itsIteratorStack.back();
        }

        //! Retrieves the current node name
        /*! @return nullptr if no name exists */
        const char * getNodeName() const
        {
            return itsIteratorStack.back().name();
        }

        //! Sets the name for the next node created with startNode
        void setNext(const char * name, bool optional)
        {
            itsNextName = name;
            itsNextOptional = false;
            itsLoadOptional = optional;
        }
        //! Gets the flag indicating to load optional value
        bool getLoadOptional()
        {
            return itsLoadOptional;
        }

        //! Loads a value from the current node - small signed overload
        template <class T, cereal::traits::EnableIf<std::is_signed<T>::value,
            sizeof(T) < sizeof(int64_t)> = cereal::traits::sfinae> inline
            void loadValue(T & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<T>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();

            val = static_cast<T>(itsIteratorStack.back().value().GetInt());
            ++itsIteratorStack.back();
        }

        //! Loads a value from the current node - small unsigned overload
        template <class T, cereal::traits::EnableIf<std::is_unsigned<T>::value,
            sizeof(T) < sizeof(uint64_t),
            !std::is_same<bool, T>::value> = cereal::traits::sfinae> inline
            void loadValue(T & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<T>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();

            val = static_cast<T>(itsIteratorStack.back().value().GetUint());
            ++itsIteratorStack.back();
        }

        //! Loads a value from the current node - bool overload
        void loadValue(bool & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<bool>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();
            if (itsLoadOptional) return;
            val = itsIteratorStack.back().value().GetBool();
            ++itsIteratorStack.back();

        }
        //! Loads a value from the current node - int64 overload
        void loadValue(int64_t & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<int64_t>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();
            if (itsLoadOptional) return;
            val = itsIteratorStack.back().value().GetInt64();
            ++itsIteratorStack.back();
        }
        //! Loads a value from the current node - uint64 overload
        void loadValue(uint64_t & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<uint64_t>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();
            if (itsLoadOptional) return;
            val = itsIteratorStack.back().value().GetUint64();
            ++itsIteratorStack.back();
        }
        //! Loads a value from the current node - float overload
        void loadValue(float & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<float>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();
            if (itsLoadOptional) return;
            val = static_cast<float>(itsIteratorStack.back().value().GetDouble());
            ++itsIteratorStack.back();
        }
        //! Loads a value from the current node - double overload
        void loadValue(double & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = boost::lexical_cast<double>(itr->second);
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();
            if (itsLoadOptional) return;
            val = itsIteratorStack.back().value().GetDouble();
            ++itsIteratorStack.back();
        }
        //! Loads a value from the current node - string overload
        void loadValue(std::string & val)
        {
            if (itsNextName)
            {
                auto itr = variable_replace_mapping.find(itsNextName);
                if (itr != variable_replace_mapping.end())
                {
                    val = itr->second;
                    itsNextName = nullptr;
                    ++itsIteratorStack.back();
                    return;
                }
            }
            search();
            if (itsLoadOptional) return;
            val = itsIteratorStack.back().value().GetString();
            for(auto& itr : string_replace_mapping)
            {
                auto pos = val.find(itr.first);
                if(pos != std::string::npos)
                {
                    val = val.substr(0,pos) + itr.second + val.substr(pos + itr.first.size());
                }
            }
            ++itsIteratorStack.back();
        }
        //! Loads a nullptr from the current node
        void loadValue(std::nullptr_t&) { search(); if (itsLoadOptional) return; CEREAL_RAPIDJSON_ASSERT(itsIteratorStack.back().value().IsNull()); ++itsIteratorStack.back(); }

        // Special cases to handle various flavors of long, which tend to conflict with
        // the int32_t or int64_t on various compiler/OS combinations.  MSVC doesn't need any of this.




    };

    // ######################################################################
    // JSONArchive prologue and epilogue functions
    // ######################################################################

    // ######################################################################
    //! Prologue for NVPs for JSON archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void prologue(JSONOutputArchive &, cereal::NameValuePair<T> const &)
    { }

    //! Prologue for NVPs for JSON archives
    template <class T> inline
        void prologue(JSONInputArchive &, cereal::NameValuePair<T> const &)
    { }

    // ######################################################################
    //! Epilogue for NVPs for JSON archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void epilogue(JSONOutputArchive &, cereal::NameValuePair<T> const &)
    { }

    //! Epilogue for NVPs for JSON archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void epilogue(JSONInputArchive &, cereal::NameValuePair<T> const &)
    { }

    //! Prologue for NVPs for JSON output archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void prologue(JSONOutputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    //! Prologue for NVPs for JSON input archives
    template <class T> inline
        void prologue(JSONInputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    // ######################################################################
    //! Epilogue for NVPs for JSON output archives
    /*! NVPs do not start or finish nodes - they just set up the names */
    template <class T> inline
        void epilogue(JSONOutputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    //! Epilogue for NVPs for JSON input archives
    template <class T> inline
        void epilogue(JSONInputArchive &, cereal::OptionalNameValuePair<T> const &)
    { }

    // ######################################################################

    // ######################################################################
    //! Prologue for SizeTags for JSON archives
    /*! SizeTags are strictly ignored for JSON, they just indicate
    that the current node should be made into an array */
    template <class T> inline
        void prologue(JSONOutputArchive & ar, cereal::SizeTag<T> const &)
    {
        ar.makeArray();
    }

    //! Prologue for SizeTags for JSON archives
    template <class T> inline
        void prologue(JSONInputArchive &, cereal::SizeTag<T> const &)
    { }

    // ######################################################################
    //! Epilogue for SizeTags for JSON archives
    /*! SizeTags are strictly ignored for JSON */
    template <class T> inline
        void epilogue(JSONOutputArchive &, cereal::SizeTag<T> const &)
    { }

    //! Epilogue for SizeTags for JSON archives
    template <class T> inline
        void epilogue(JSONInputArchive &, cereal::SizeTag<T> const &)
    { }

    // ######################################################################
    //! Prologue for all other types for JSON archives (except minimal types)
    /*! Starts a new node, named either automatically or by some NVP,
    that may be given data by the type about to be archived

    Minimal types do not start or finish nodes */
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_output_serialization, JSONOutputArchive>::value,
        !cereal::traits::has_minimal_output_serialization<T, JSONOutputArchive>::value> = cereal::traits::sfinae>
        inline void prologue(JSONOutputArchive & ar, T const &)
    {
        ar.startNode();
    }

    //! Prologue for all other types for JSON archives
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_input_serialization, JSONInputArchive>::value,
        !cereal::traits::has_minimal_input_serialization<T, JSONInputArchive>::value> = cereal::traits::sfinae>
        inline void prologue(JSONInputArchive & ar, T const &)
    {
        ar.startNode();
    }

    // ######################################################################
    //! Epilogue for all other types other for JSON archives (except minimal types)
    /*! Finishes the node created in the prologue

    Minimal types do not start or finish nodes */
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_output_serialization, JSONOutputArchive>::value,
        !cereal::traits::has_minimal_output_serialization<T, JSONOutputArchive>::value> = cereal::traits::sfinae>
        inline void epilogue(JSONOutputArchive & ar, T const &)
    {
        ar.finishNode();
    }

    //! Epilogue for all other types other for JSON archives
    template <class T, cereal::traits::EnableIf<!std::is_arithmetic<T>::value,
        !cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_input_serialization, JSONInputArchive>::value,
        !cereal::traits::has_minimal_input_serialization<T, JSONInputArchive>::value> = cereal::traits::sfinae>
        inline void epilogue(JSONInputArchive & ar, T const &)
    {
        ar.finishNode();
    }

    // ######################################################################
    //! Prologue for arithmetic types for JSON archives
    inline
        void prologue(JSONOutputArchive & ar, std::nullptr_t const &)
    {
        ar.writeName();
    }

    //! Prologue for arithmetic types for JSON archives
    inline
        void prologue(JSONInputArchive &, std::nullptr_t const &)
    { }

    // ######################################################################
    //! Epilogue for arithmetic types for JSON archives
    inline
        void epilogue(JSONOutputArchive &, std::nullptr_t const &)
    { }

    //! Epilogue for arithmetic types for JSON archives
    inline
        void epilogue(JSONInputArchive &, std::nullptr_t const &)
    { }

    // ######################################################################
    //! Prologue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void prologue(JSONOutputArchive & ar, T const &)
    {
        ar.writeName();
    }

    //! Prologue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void prologue(JSONInputArchive &, T const &)
    { }

    // ######################################################################
    //! Epilogue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void epilogue(JSONOutputArchive &, T const &)
    { }

    //! Epilogue for arithmetic types for JSON archives
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void epilogue(JSONInputArchive &, T const &)
    { }

    // ######################################################################
    //! Prologue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void prologue(JSONOutputArchive & ar, std::basic_string<CharT, Traits, Alloc> const &)
    {
        ar.writeName();
    }

    //! Prologue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void prologue(JSONInputArchive &, std::basic_string<CharT, Traits, Alloc> const &)
    { }

    // ######################################################################
    //! Epilogue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void epilogue(JSONOutputArchive &, std::basic_string<CharT, Traits, Alloc> const &)
    { }

    //! Epilogue for strings for JSON archives
    template<class CharT, class Traits, class Alloc> inline
        void epilogue(JSONInputArchive &, std::basic_string<CharT, Traits, Alloc> const &)
    { }

    // ######################################################################
    // Common JSONArchive serialization functions
    // ######################################################################
    //! Serializing NVP types to JSON
    template <class T> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, cereal::NameValuePair<T> const & t)
    {
        ar.setNextName(t.name);
        ar(t.value);
    }

    template <class T> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, cereal::NameValuePair<T> & t)
    {
        ar.setNext(t.name, false);
        ar(t.value);
    }

    //! Serializing optional NVP types to JSON
    template <class T> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, cereal::OptionalNameValuePair<T> & t)
    {
        ar.setNext(t.name, true);
        ar(t.value);
        if (ar.getLoadOptional())
        {
            t.value = t.defaultValue;
        }
    }

    //! Saving for nullptr to JSON
    inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, std::nullptr_t const & t)
    {
        ar.saveValue(t);
    }

    //! Loading arithmetic from JSON
    inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, std::nullptr_t & t)
    {
        ar.loadValue(t);
    }

    //! Saving for arithmetic to JSON
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, T const & t)
    {
        ar.saveValue(t);
    }

    //! Loading arithmetic from JSON
    template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, T & t)
    {
        ar.loadValue(t);
    }

    //! saving string to JSON
    template<class CharT, class Traits, class Alloc> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive & ar, std::basic_string<CharT, Traits, Alloc> const & str)
    {
        ar.saveValue(str);
    }

    //! loading string from JSON
    template<class CharT, class Traits, class Alloc> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, std::basic_string<CharT, Traits, Alloc> & str)
    {
        ar.loadValue(str);
    }

    // ######################################################################
    //! Saving SizeTags to JSON
    template <class T> inline
        void CEREAL_SAVE_FUNCTION_NAME(JSONOutputArchive &, cereal::SizeTag<T> const &)
    {
        // nothing to do here, we don't explicitly save the size
    }

    //! Loading SizeTags from JSON
    template <class T> inline
        void CEREAL_LOAD_FUNCTION_NAME(JSONInputArchive & ar, cereal::SizeTag<T> & st)
    {
        ar.loadSize(st.size);
    }
} // namespace aq

namespace cereal
{
    inline void save(JSONOutputArchive& ar, rcc::shared_ptr<aq::IDataStream> const & stream)
    {
        auto nodes = stream->getAllNodes();
        ar(CEREAL_NVP(nodes));
    }
    struct ImplicitParamInfo{
        std::string type;
        std::string flags;
        template<class AR>
        void save(AR& ar) const{
            ar(CEREAL_NVP(type));
            if(!flags.empty())
                ar(CEREAL_OPTIONAL_NVP(flags, flags));
        }

        template<class AR>
        void load(AR& ar){
            ar(CEREAL_NVP(type));
            ar(CEREAL_OPTIONAL_NVP(flags, flags));
        }
    };

    inline void save(JSONOutputArchive& ar, std::vector<std::shared_ptr<mo::IParam>> const& params){
          std::map<std::string, ImplicitParamInfo> types;
          for (auto& param : params){
              if (!param->checkFlags(mo::Control_e)){
                  types[param->getName()] = {mo::Demangle::typeToName(param->getTypeInfo()), mo::paramFlagsToString(param->getFlags())};
                  continue;
              }
              auto func1 = mo::SerializationFactory::instance()->getJsonSerializationFunction(param->getTypeInfo());
              if (func1){
                  if (func1(param.get(), ar)){
                      types[param->getName()] = {mo::Demangle::typeToName(param->getTypeInfo()), mo::paramFlagsToString(param->getFlags())};
                  }else{
                      MO_LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
                  }
              }else{
                  MO_LOG(debug) << "No serialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
              }
          }
          ar(CEREAL_NVP(types));
    }

    inline void save(JSONOutputArchive& ar, std::vector<mo::IParam*> const& parameters){
        for (auto& param : parameters){
            if (!param->checkFlags(mo::Control_e))
                continue;
            auto func1 = mo::SerializationFactory::instance()->getJsonSerializationFunction(param->getTypeInfo());
            if (func1){
                if (!func1(param, ar)){
                    MO_LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
                }
            }else{
                MO_LOG(debug) << "No serialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
            }
        }
    }

    inline void save(JSONOutputArchive& ar, std::vector<mo::InputParam*> const& parameters)
    {
        for (auto& param : parameters)
        {

            mo::InputParam* input_param = dynamic_cast<mo::InputParam*>(param);
            aq::InputInfo info;
            info.type = "Direct";
            if (input_param)
            {
                mo::IParam* _input_param = input_param->getInputParam();
                std::string input_name;
                if (_input_param)
                {
                    input_name = _input_param->getTreeName();
                    auto pos = input_name.find(" buffer for ");
                    if(pos != std::string::npos)
                    {
                        input_name = input_name.substr(0, input_name.find_first_of(' '));
                    }

                    if(_input_param->checkFlags(mo::Buffer_e))
                    {
                        mo::Buffer::IBuffer* buffer = dynamic_cast<mo::Buffer::IBuffer*>(_input_param);
                        if(buffer)
                        {
                            info.type = mo::paramTypeToString(buffer->getBufferType());
                            auto size = buffer->getFrameBufferCapacity();
                            if(size)
                                info.buffer_size = *size;
                            auto time_padding = buffer->getTimePaddingCapacity();
                            if(time_padding)
                                info.buffer_time = time_padding;
                        }
                    }
                    info.name = input_name;
                }
            }
            ar(cereal::make_nvp(param->getName(), info));
        }
    }
    inline void save(JSONOutputArchive& ar, rcc::weak_ptr<aq::nodes::Node> const& node)
    {
        std::string name = node->getTreeName();
        ar(CEREAL_NVP(name));
    }
    inline void save(JSONOutputArchive& ar, rcc::weak_ptr<aq::Algorithm> const& obj)
    {
        auto parameters = obj->getParams();
        std::string type = obj->GetTypeName();
        const auto& components = obj->getComponents();
        ar(CEREAL_NVP(type));
        if(parameters.size())
            ar(CEREAL_NVP(parameters));
        if(components.size())
            ar(CEREAL_NVP(components));
    }

    inline void save(JSONOutputArchive& ar, rcc::shared_ptr<aq::nodes::Node> const& node){
        aq::JSONOutputArchive& ar_ = dynamic_cast<aq::JSONOutputArchive&>(ar);
        auto parameters = node->getParams();
        std::string type = node->GetTypeName();
        std::string name = node->getTreeName();
        const auto& components = node->getComponents();
        ar(CEREAL_NVP(type));
        ar(CEREAL_NVP(name));
        if(ar_.preset != "Default")
            ar(cereal::make_nvp("preset", std::vector<std::string>(1, ar_.preset)));
        int control_count = 0;
        for(auto param : parameters)
            if(param->checkFlags(mo::Control_e))
                ++control_count;
        if(control_count)
            ar(CEREAL_NVP(parameters));
        auto implicit_params = node->getImplicitParams();
        if(implicit_params.size())
            ar(CEREAL_NVP(implicit_params));
        if(components.size())
            ar(CEREAL_NVP(components));
        auto inputs = node->getInputs();
        if(inputs.size())
            ar(CEREAL_NVP(inputs));
        auto parent_nodes = node->getParents();
        std::vector<std::string> parents;
        for(auto& node : parent_nodes){
            parents.emplace_back(node->getTreeName());
        }
        if(parents.size())
            ar(CEREAL_NVP(parents));
    }


    inline void load(JSONInputArchive& ar, rcc::shared_ptr<aq::IDataStream>& stream){
        if(stream == nullptr){
            stream = aq::IDataStream::create();
            MO_ASSERT(stream) << "Unable to create datastream.  Was aquila_core loaded correctly?";
        }
        std::vector<rcc::shared_ptr<aq::nodes::Node>> nodes;
        ar(CEREAL_NVP(nodes));
        aq::JSONInputArchive& ar_ = dynamic_cast<aq::JSONInputArchive&>(ar);
        
        for(int i = 0; i < nodes.size(); ++i){
            if(!nodes[i]){
                MO_LOG(debug) << "Unable to deserialize node at index: " << i;
                continue;
            }
            nodes[i]->setDataStream(stream.get());
            nodes[i]->postSerializeInit();
            auto& parents = ar_.parent_mappings[nodes[i]->getTreeName()];
            for (auto& parent : parents){
                bool found_parent = false;
                for(int j = 0; j < nodes.size(); ++j){
                    if(nodes[j] == nullptr){
                        MO_LOG(debug) << "Unable to deserialize node at index: " << j;
                        continue;
                    }
                    if(i != j){
                        if(nodes[j]->getTreeName() == parent){
                            nodes[j]->addChild(nodes[i]);
                            found_parent = true;
                            continue;
                        }
                    }
                }
                if(!found_parent){
                    MO_LOG(warning) << "Unable to find parent [" << parent << "] for node [" << nodes[i]->getTreeName() << "]";
                }
            }
            auto& input_mappings = ar_.input_mappings[nodes[i]->getTreeName()];
            auto input_params = nodes[i]->getInputs();
            for(auto& input : input_params){
                auto itr = input_mappings.find(input->getName());
                if(itr != input_mappings.end()){
                       auto pos = itr->second.name.find(":");
                       if(pos != std::string::npos){
                           std::string output_node_name = itr->second.name.substr(0, pos);
                           for(int j = 0; j < nodes.size(); ++j){
                               if(nodes[j] == nullptr)
                                   continue;
                                if(nodes[j]->getTreeName() == output_node_name){
                                    auto space_pos = itr->second.name.find(' ');
                                    auto output_param = nodes[j]->getOutput(itr->second.name.substr(pos + 1, space_pos - (pos + 1)));
                                    if (!output_param){
                                        MO_LOG(warning) << "Unable to find parameter " << itr->second.name.substr(pos + 1) << " in node " << nodes[j]->getTreeName();
                                        break;
                                    }

                                    std::string type = itr->second.type;
                                    if(type != "Direct"){
                                          mo::ParamType buffer_type = mo::stringToParamType(type);
                                          if (!nodes[i]->connectInput(nodes[j], output_param, input, mo::ParamType(buffer_type | mo::ForceBufferedConnection_e))){
                                              MO_LOG(warning) << "Unable to connect " << output_param->getTreeName() << " (" << output_param->getTypeInfo().name() << ") to "
                                                  << input->getTreeName() << " (" << input->getTypeInfo().name() << ")";
                                          }else{
                                              if(itr->second.buffer_size > 0){
                                                  mo::IParam* p = input->getInputParam();
                                                  if(mo::Buffer::IBuffer* b = dynamic_cast<mo::Buffer::IBuffer*>(p)){
                                                      b->setFrameBufferCapacity(itr->second.buffer_size);
                                                      if(itr->second.buffer_time)
                                                        b->setTimePaddingCapacity(*itr->second.buffer_time);
                                                  }
                                              }
                                              if(itr->second.sync){
                                                  nodes[i]->setSyncInput(input->getName());
                                              }
                                          }
                                    }else{
                                        if (!nodes[i]->connectInput(nodes[j], output_param, input)){
                                            MO_LOG(warning) << "Unable to connect " << output_param->getTreeName() << " (" << output_param->getTypeInfo().name() << ") to "
                                                << input->getTreeName() << " (" << input->getTypeInfo().name() << ")";
                                        }else{
                                           if(itr->second.sync){
                                               nodes[i]->setSyncInput(input->getName());
                                               MO_LOG(info) << "Node (" << nodes[i]->getTreeName() << ") syncs to " << input->getName();
                                           }
                                        }
                                    }
                                }
                           }
                       }else{
                           if(itr->second.name.size())
                               MO_LOG(warning) << "Invalid input format for input [" << itr->second.name << "] of node: " << nodes[i]->getTreeName();
                       }
                }else{
                    if(input->checkFlags(mo::Optional_e)){
                        MO_LOG(debug) << "Unable to find input setting for " << input->getName() << " for node " << nodes[i]->getTreeName();
                    }else{
                        MO_LOG(warning) << "Unable to find input setting for " << input->getName() << " for node " << nodes[i]->getTreeName();
                    }
                }
            }
        }
        for(int i = 0; i < nodes.size(); ++i){
            if(nodes[i] != nullptr && nodes[i]->getParents().size() == 0){
                stream->addNode(nodes[i]);
            }
        }
    }

   inline void load(JSONInputArchive& ar, std::vector<std::shared_ptr<mo::IParam>>& parameters){
       std::map<std::string, ImplicitParamInfo> types;
       ar(CEREAL_OPTIONAL_NVP(types, types));
       for(const auto& type_info : types){
          const auto& type = mo::Demangle::nameToType(type_info.second.type);
          if(type != mo::TypeInfo(typeid(void))){
              auto param = mo::ParamFactory::instance()->create(type, mo::TParam_e);
              if(param){
                  auto flag = mo::stringToParamFlags(type_info.second.flags);
                  param->setFlags(flag);
                  param->setName(type_info.first);
                  auto func1 = mo::SerializationFactory::instance()->getJsonDeSerializationFunction(param->getTypeInfo());
                   if (func1){
                       if (!func1(param.get(), ar)){
                           MO_LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
                       }
                   }else{
                       MO_LOG(debug) << "No deserialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
                   }
                   parameters.push_back(param);
              }
          }
       }
   }

    inline void load(JSONInputArchive& ar, std::vector<mo::IParam*>& parameters){
        for (auto& param : parameters){
            if (param->checkFlags(mo::Output_e) || param->checkFlags(mo::Input_e))
                continue;
            auto func1 = mo::SerializationFactory::instance()->getJsonDeSerializationFunction(param->getTypeInfo());
            if (func1){
                if (!func1(param, ar)){
                    MO_LOG(debug) << "Unable to deserialize " << param->getName() << " of type " << param->getTypeInfo().name();
                }
            }else{
                MO_LOG(debug) << "No deserialization function exists for  " << param->getName() << " of type " << param->getTypeInfo().name();
            }
        }
    }

    inline void load(JSONInputArchive& ar, std::vector<mo::InputParam*> & parameters){
        for (auto& param : parameters){
            std::string name = param->getName();
            aq::InputInfo info;
            auto nvp = cereal::make_optional_nvp(name, info, info);
            ar(nvp);
            if(nvp.success == false)
                return;
            aq::JSONInputArchive& ar_ = dynamic_cast<aq::JSONInputArchive&>(ar);
            ar_.input_mappings[param->getTreeRoot()][name] = info;
        }
    }

   inline void load(JSONInputArchive& ar, rcc::weak_ptr<aq::Algorithm>& obj){
       std::string type;
       ar(CEREAL_NVP(type));
       if(!obj){
            IObject* ptr =  mo::MetaObjectFactory::instance()->create(type.c_str());
            if(ptr){
                aq::Algorithm* alg_ptr = dynamic_cast<aq::Algorithm*>(ptr);
                if(!alg_ptr)
                    delete ptr;
                else
                    obj = alg_ptr;
            }
       }
       if(!obj){
            MO_LOG(warning) << "Unable to create algorithm of type: " << type;
            return;
       }
       auto parameters = obj->getParams();
       if(parameters.size())
          ar(CEREAL_NVP(parameters));
       std::vector<rcc::weak_ptr<aq::Algorithm>> components;
       try
       {
           ar(CEREAL_OPTIONAL_NVP(components, components));
       }catch(...){

       }

       if(components.size())
            for(auto component : components)
                obj->addComponent(component);
   }
    inline void load(JSONInputArchive& ar, rcc::shared_ptr<aq::nodes::Node>& node){
        aq::JSONInputArchive& ar_ = dynamic_cast<aq::JSONInputArchive&>(ar);
        std::string type;
        std::string name;
        std::vector<std::string> preset;
        ar(CEREAL_OPTIONAL_NVP(preset, preset));
        bool valid_preset = preset.empty();
        for(const auto& prst : preset){
           if(prst == ar_.preset || prst == "Default"){
                valid_preset = true;
           }
           if(prst[0] == '!'){
                if(prst.substr(1) == ar_.preset)
                    return;
                else
                    valid_preset = true;
            }
        }
        if(!valid_preset)
            return;
        ar(CEREAL_NVP(type));
        if(!node)
            node = mo::MetaObjectFactory::instance()->create(type.c_str());
        if (!node){
            MO_LOG(warning) << "Unable to create node with type: " << type;
            return;
        }

        std::vector<rcc::weak_ptr<aq::Algorithm>> components;
        auto components_nvp = CEREAL_OPTIONAL_NVP(components, components);
        ar(components_nvp);
        if(components_nvp.success){
            for(auto component : components){
                if(component)
                    node->addComponent(component);
            }
        }

        ar(CEREAL_NVP(name));
        node->setTreeName(name);
        auto parameters = node->getParams();
        for(auto itr = parameters.begin(); itr != parameters.end(); ){
            if((*itr)->checkFlags(mo::Input_e)){
                itr = parameters.erase(itr);
            }else{
                ++itr;
            }
        }
        if(parameters.size())
            ar(CEREAL_OPTIONAL_NVP(parameters, parameters));
        std::vector<std::shared_ptr<mo::IParam>> implicit_params;
        ar(CEREAL_OPTIONAL_NVP(implicit_params,implicit_params));
        for(const auto& imp : implicit_params)
            node->addParam(imp);
        auto inputs = node->getInputs();
        if(inputs.size())
            ar(CEREAL_OPTIONAL_NVP(inputs, inputs));
        
        ar(cereal::make_optional_nvp("parents", ar_.parent_mappings[name]));
    }
}

  // register archives for polymorphic support
CEREAL_REGISTER_ARCHIVE(aq::JSONInputArchive)
CEREAL_REGISTER_ARCHIVE(aq::JSONOutputArchive)

// tie input and output archives together
CEREAL_SETUP_ARCHIVE_TRAITS(aq::JSONInputArchive, aq::JSONOutputArchive)


