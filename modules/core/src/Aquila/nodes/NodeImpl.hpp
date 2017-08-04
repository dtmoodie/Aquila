#pragma once
namespace aq{
namespace nodes{

class NodeImpl {
public:
    long long   throw_count           = 0;
    bool        disable_due_to_errors = false;
    std::string tree_name;
    long long   iterations_since_execution    = 0;
    const char* last_execution_failure_reason = 0;
#ifdef _DEBUG
    std::vector<long long> timestamps;
#endif
};
#define EXCEPTION_TRY_COUNT 10

} // namespace aq::nodes
} // namespace aq
#define CATCH_MACRO                                                            \
    catch (mo::ExceptionWithCallStack<cv::Exception> & e) {                    \
        LOG_NODE(error) << e.what() << "\n"                                    \
                        << e.callStack();                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (thrust::system_error & e) {                                         \
        LOG_NODE(error) << e.what();                                           \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (mo::ExceptionWithCallStack<std::string> & e) {                      \
        LOG_NODE(error) << std::string(e) << "\n"                              \
                        << e.callStack();                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (mo::IExceptionWithCallStackBase & e) {                              \
        LOG_NODE(error) << "Exception thrown with callstack: \n"               \
                        << e.callStack();                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (boost::thread_resource_error & err) {                               \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (boost::thread_interrupted & err) {                                  \
        LOG_NODE(error) << "Thread interrupted";                               \
        /* Needs to pass this back up to the chain to the processing thread.*/ \
        /* That way it knowns it needs to exit this thread */                  \
        throw err;                                                             \
    }                                                                          \
    catch (boost::thread_exception & err) {                                    \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (cv::Exception & err) {                                              \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (boost::exception & err) {                                           \
        LOG_NODE(error) << "Boost error";                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (std::exception & err) {                                             \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (...) {                                                              \
        LOG_NODE(error) << "Unknown exception";                                \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }
