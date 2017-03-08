#pragma once

#include <boost/core/noncopyable.hpp>
#include <atomic>
namespace mo
{
	class RefCounter
	{
	public:
		friend void intrusive_ptr_add_ref(RefCounter* p)
		{
			++p->_counter;
		}
		friend void intrusive_ptr_release(RefCounter* p)
		{	
			if (--p->_counter == 0)
				delete p;
		}
	protected:
		RefCounter(): _counter(0) {}
		RefCounter(const RefCounter&) : _counter(0) {}
		virtual ~RefCounter() = 0 {}
		RefCounter& operator=(const RefCounter&) { return *this; }
		void RefCounter::swap(RefCounter&) {}
	private:
		mutable std::atomic_size_t _counter;
	};
}