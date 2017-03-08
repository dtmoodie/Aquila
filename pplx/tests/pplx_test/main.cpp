#define CPPREST_FORCE_PPLX 1
#include "pplx/pplx.h"
#include "pplx/pplxwin.h"
#include "pplx/pplxtasks.h"
#include <iostream>

int function1(int value)
{
    std::cout << "Function 1 " << value;
    return value * 10;
}
int function2(int value)
{
    std::cout << "Function2 " << value;
    return value * 5;
}
int main()
{
    int x = 10;
    auto task = pplx::create_task([x]()->int
    {
        std::cout << "Task 1 " << x  << std::endl; 
        return function1(x);
    }).then([](int x)->void
    {
        std::cout << "Task 2 " << x;
    }).wait();
    
    return 0;
}