#include <iostream>
#include "vector2.h"

#define PI 3.14159265


int main(int, char**) {

    Vector2D<double> v;
    Vector2D<double> u;
    std::cout << "Enter vector v:\n";
    std::cin >> v;
    std::cout << "Enter vector u:\n";
    std::cin >> u;
    std::cout << "The two vectors are v = " << v << ", u = " << u << "\n";
    
    v += u;
    u -= v;
    int a = 30;

    std::cout << "Modify v: v+=u: "<< v << "\n"
              << "Modify u: u-=v: "<< u << "\n"
              << "dot(u, v): " << dot(u, v) << "\n"
              << "length of v: " << length(v) << "\n"
              << "sqlength of v: " << sqlength(v) << "\n"
              << "normalized u: " << normalize(u) << "\n"
              << "sum u+v: " << u + v << "\n"
              << "difference u-v: " << u - v << "\n"
              << "v multiplied by 30 from the left: " << a * v << "\n"
              << "v multiplied by 30 from the right: " << v * a << "\n"
              << "v divided by 30: " << v / a << "\n"
              << "mirror of u to x axis: " << mirror_x(u) << "\n"
              << "mirror of u to y axis: " << mirror_y(u) << "\n"
              << "swap components of u: " << swap(u) << "\n"
              << "rotate u with 90 degrees: " << rotate(u, PI * 3 * a / 180) << std::endl;

    return 0;
}
