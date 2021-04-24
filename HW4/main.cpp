#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

template<typename State, typename T, typename RHS, typename Callback>
auto solve_euler(State y0, T t0, T t1, T h, RHS f, Callback cb)
{
    T t = t0; State y = y0;
    std::vector<State> data;
    data.push_back(y);
    while(t < t1)
    {
        if(t + h > t1)
        {
            h = t1 - t;
        }
        y = y + h * f(t, y);
        t = t + h; cb(t, y);

        data.push_back(y);
    }

    return data;
}

template<typename State, typename T, typename RHS, typename Callback>
auto solve_rk4(State y0, T t0, T t1, T h, RHS f, Callback cb)
{
    T t = t0; State y = y0;
    std::vector<State> data;
    data.push_back(y);
    while(t < t1)
    {
        if(t + h > t1)
        {
            h = t1 - t;
            std::cout << h << std::endl;
        }
        State k1 = f(t, y);
        State k2 = f(t + h * (T)0.5, y + (h * (T)0.5) * k1);
        State k3 = f(t + h * (T)0.5, y + (h * (T)0.5) * k2);
        State k4 = f(t + h, y + h * k3);
        y = y + (k1 + k4 + (T)2 * (k2 + k3)) * (h / (T)6);
        t = t + h; cb(t, y);

        data.push_back(y);
    }

    return data;
}

template<typename T, typename State>
auto rhs(T t, State y)
{
    return 1 + y*y;
}

template<typename T, typename State>
void callback(T t, State y)
{
    //std::cout << y << std::endl;
    return;
}


template<typename T, typename State>
auto analytic_sol(State y0, T t0, T t, T h)
{
    int q = (int)((t - t0) / h);
    T r = (t - t0) - h * q;
    std::vector<State> data((int)ceil(q + (r / (t - t0)) + 1));
    std::generate(data.begin(), data.end(), [n = 0, h, t0, y0] () mutable { n += 1; return tan(h * (n - 1 + t0) + atan(y0)); });
    std::cout << r << std::endl;
    return data;
}

template<typename State>
void vecs_to_file(std::string file_name, std::vector<std::vector<State>> vecs, int idx_of_analytic_data)
{
    std::ofstream output(file_name);
    if(output.is_open())
    {
        std::vector<std::string> data_header = {"Euler", "Runge-Kutta", "Analytic"};

        std::copy(data_header.begin(), data_header.end(), std::ostream_iterator<std::string>(output, " "));

        output << "\n";

        for(int i = 0; i < static_cast<int>(vecs[idx_of_analytic_data].size()); i++)
        {
            output << vecs[0][i] << ", " << vecs[1][i] << ", " << vecs[2][i] << "\n" << std::endl;
        }
    }else
    {
        std::cout << "Could not open output file\n";
    }
}

int main(int, char**) {

    double y0 = 0.; double t0 = 0.; double t = 1.57; double h = 0.001;
    
    auto y_rk = solve_rk4(y0, t0, t, h, rhs<double, double>, callback<double, double>);

    auto y_e = solve_euler(y0, t0, t, h, rhs<double, double>, callback<double, double>);

    auto data = analytic_sol(y0, t0, t, h);

    std::vector<std::vector<double>> vecs{y_e, y_rk, data};

    std::string str("solution.txt");

    vecs_to_file(str, vecs, 2);
}
