#include <iostream>

/*
In this code the following templates are used:
    F, G, H: function typenames for the function at hand, its derivative and the stopping criterion, respectively.
    T: typnames for the initial value and the tolerance parameter. 
*/

//Newton step function for one step. It also checks the zero division possibility.
template<typename F, typename G, typename T>
T newton_step(F func, G d_func, T x_begin, double epsilon = 1e-14)
{
    //Calculate value of the function.
    auto y = func(x_begin);

    //Calculate value of the derivative.
    auto d_y = d_func(x_begin);

    //Check zero division possibility.
    if(std::abs(d_y) < epsilon)
    {
        throw "Iteration did not converge: too small value encountered at division!";
    }

    //Calculate next guess.
    T x_next = x_begin - y/d_y;
    return x_next;
}

//Stopping criterion defined as required for this homework.
template<typename T>
bool criterion(T x_begin, T x_next, T tolerance = 1e-7)
{
    return std::abs(x_next - x_begin) > tolerance;
}

//Newton iteration function. Convergence is reaching the level of tolerance within the maximum number of iterations. Otherwise current guess is returned.
template<typename F, typename G, typename T, typename H>
T newton_it(F func, G d_func, T x_begin, H stop_criterion, T tolerance = 1e-7, int max_iter = 20, double epsilon = 1e-14)
{
    for(int i = 0; i < max_iter; i++)
    {
        T x_next = newton_step(func, d_func, x_begin, epsilon);

        if(stop_criterion(x_begin, x_next, tolerance))
        {
            x_begin = x_next;
        }
        else
        {
            return x_next;
        }
    }

    std::cout << "Iteration did not converge within " << max_iter << " steps. "
              << "Current guess is returned." << std::endl;
    
    return x_begin;
}

int main(int, char**) {
    std::cout.precision(24);
    try{
        //Required function call is possible. The result is satisfying compared to the value given in the homework.
        double x0 = newton_it([](double x){return x*x - 612.0;}, [](double x){return 2.0*x;}, 10.0, criterion<double>, 1e-13);
        std::cout << x0 << std::endl;
    }catch(const char* msg){
        std::cerr << msg << std::endl;
        }
    }