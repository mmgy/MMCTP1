#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

template<typename T>
class Catenary
{
    T a, b, h, F, q;

    public:

        void setParams(T const& set_a, T const& set_b, T const& set_h, T const& set_q, T const& set_F)
        {
            a = set_a;
            b = set_b;
            h = set_h;
            q = set_q;
            F = set_F;

            std::cout << "The the poles are at " << a << " and " << b << " meters and are " << h << " meters high." << std::endl;
            std::cout << "The cable has a specific weight of " << q << " N/m and the horizontal component of the straining force is " << F << " N." << std::endl; 
        }

        T cable(T const& x);

        T D(T const& x, T const& dh);

        T richardson_o6(T const& x, T const& dh);

        T integral_centric(int const& n, T const& dh);

        T integral_trapezoid(int const& n, T const& dh);

        T integral_simpson_1o3(int const& n, T const& dh);

        T length_analytic();

        T M(int const& i);        

        int find_n(T const& eps, T const& M, int const& k, int const& c);
};

//Function to calculate central difference
template<typename T>
T Catenary<T>::D(T const& x, T const& dh)
{
    T d = (cable(x + dh) - cable(x - dh)) / (2*dh);
    return d;
}

//Function to perform the Richardson extrapolation to have an O(dh^6) first derivative 
template<typename T>
T Catenary<T>::richardson_o6(T const& x, T const& dh)
{   
    T d = 64/45 * D(x, dh) - 4/9 * D(x, 2*dh) + 1/45 * D(x, 4*dh);
    return d;
}

//Curve that describes the shape of the cable. The position of the poles are at a and b, a < b.
template<typename T>
T Catenary<T>::cable(T const& x)
{
    T height = F/q * (cosh(q/F * (x - (a + b)/2)) - cosh(q/F * (b - a)/2)) + h;
    return height;
}

//Centric integral
template<typename T>
T Catenary<T>::integral_centric(int const& n, T const& dh)
{
    T dx = (b - a) / n;
    T sum = 0;
    for(int i = 1; i <= n; i++)
    {
        T m = (2*i - 1)/2 * dx + a;
        T s = sqrt(1 + pow(richardson_o6(m, dh), 2));
        sum += dx * s;
    }

    return sum;
}

//Trapezoid integral
template<typename T>
T Catenary<T>::integral_trapezoid(int const& n, T const& dh)
{   
    T dx = (b - a) / n;
    T sum = dx/2 * (sqrt(1 + pow(richardson_o6(a, dh), 2)) + sqrt(1 + pow(richardson_o6(b, dh), 2)));
    for(int i = 1; i < n; i++)
    {
        T n = a + i*dx;
        T s = sqrt(1 + pow(richardson_o6(n, dh), 2));
        sum += dx * s;
    }

    return sum;
}

//Simpson 1/3 integral
template<typename T>
T Catenary<T>::integral_simpson_1o3(int const& n, T const& dh)
{
    T sum = (2 * integral_centric(n, dh) + integral_trapezoid(n, dh)) / 3;

    return sum;
}

//Analytic value of the length of the cable without using the Richardson extrapolation
template<typename T>
T Catenary<T>::length_analytic()
{
    return 2*F / q *sinh(q/F * (b - a)/2);
}

//Function to find the value of the maximum of the ith derivative of the integrand for the upper bound of the error
template<typename T>
T Catenary<T>::M(int const& i)
{
    return pow((q / F), i) * ((1 - pow(-1, i)) / 2 * sinh(q/F * (b - a)/2) + (1 + pow(-1, i)) / 2 * cosh(q/F * (b - a)/2));
}

//Find the smallest integer n to have an error less than eps. The k and c parameters are there in order to be able to calculate the error term for multiple methods of integration.
template<typename T>
int Catenary<T>::find_n(T const& eps, T const& M, int const& k, int const& c)
{
    T n = ceil(pow(pow((b - a), k) / c / eps * M, 1. / (k - 1)));
    return (int) n;
}


int main(int, char**) {
    Catenary<double> c;

    c.setParams(-100, 100, 35, 1.8, 900);

    std::cout << "Analytic value of length: " << c.length_analytic() << std::endl;

    //Finding the n according to the error estimations
    int c_n = c.find_n(0.01, c.M(2), 3, 24);
    int tr_n = c.find_n(0.01, c.M(2), 3, 12);
    int s_n = c.find_n(0.01, c.M(4), 5, 180);

    //Vector of the centric, trapezoid, simpson and analytic results for the integral, respectively, using the above determined n values
    //I found it sufficient to set the step size of the numerical derivative 1. This is because the cable is almost horizontal and the slope of the curve is very small too.
    std::vector<double> data = {c.integral_centric(c_n, 1), c.integral_trapezoid(tr_n, 1), c.integral_simpson_1o3(s_n, 1), c.length_analytic()};
    std::vector<std::string> data_header = {"Centric", "Trapezoid", "Simpson 1/3", "Analytic"};

    std::cout << "Centric n: " << c_n << " Centric integral: " << data[0] << std::endl;
    std::cout << "Trapezoid n: " << tr_n << " Trapeziod integral: " << data[1] << std::endl;
    std::cout << "Simpson 1/3 n: " << s_n << " Simpson 1/3 integral: " << data[2] << std::endl;

    //Writing out to the results.txt file
    std::ofstream output("results.txt");
    if(output.is_open())
    {   
        std::copy(data_header.begin(), data_header.end(), std::ostream_iterator<std::string>(output, " "));
        output << "\n";
        std::copy(data.begin(), data.end(), std::ostream_iterator<double>(output, " "));
    }else
    {
        std::cout << "Could not open output file\n";
    }
}
