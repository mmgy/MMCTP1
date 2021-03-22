//vector2.cpp

#include <iostream>
#include "vector2.h"
#include <cmath>

template<typename T>
Vector2D<T>& Vector2D<T>::operator+=(Vector2D<T> const& v)
{
    x += v.x; y += v.y;
    return *this;
}

template<typename T>
Vector2D<T>& Vector2D<T>::operator-=(Vector2D<T> const& v)
{
    x -= v.x; y -= v.y;
    return *this;
}

template<typename T>
std::ostream& operator<<(std::ostream& o, Vector2D<T> const& v)
{
    o << "(" << v.x << ", " << v.y << ")";
    return o;
}

template<typename T>
std::istream& operator>>(std::istream& i, Vector2D<T>& v)
{
    i >> v.x;
    i >> v.y;
    return i;
}

template<typename T>
T dot(Vector2D<T> const& v, Vector2D<T> const& u)
{
    return v.x * u.x + v.y * u.y;
}

template<typename T>
T length(Vector2D<T> const& v)
{
    return sqrt(v.x * v.x + v.y * v.y);
}

template<typename T>
T sqlength(Vector2D<T> const& v)
{
    return v.x * v.x + v.y * v.y;
}

template<typename T>
Vector2D<T> normalize(Vector2D<T> const& v)
{
    return v / length(v);
}

template<typename T>
Vector2D<T> operator+(Vector2D<T> const& v, Vector2D<T> const& u)
{
    return Vector2D<T>{v.x + u.x, v.y + u.y};
}

template<typename T>
Vector2D<T> operator-(Vector2D<T> const& v, Vector2D<T> const& u)
{
    return Vector2D<T>{v.x - u.x, v.y - u.y};
}

template<typename T, typename P>
Vector2D<T> operator*(P const& a, Vector2D<T> const& v)
{
    return Vector2D<T>{a * v.x, a * v.y};
}

template<typename T, typename P>
Vector2D<T> operator*(Vector2D<T> const& v, P const& a)
{
    return Vector2D<T>{a * v.x, a * v.y};
}

template<typename T, typename P>
Vector2D<T> operator/(Vector2D<T> const& v, P const& a)
{
    return Vector2D<T>{v.x / a, v.y / a};
}

template<typename T>
Vector2D<T> mirror_x(Vector2D<T> const& v)
{
    return Vector2D<T>{v.x, -v.y};
}

template<typename T>
Vector2D<T> mirror_y(Vector2D<T> const& v)
{
    return Vector2D<T>{-v.x, v.y};
}

template<typename T>
Vector2D<T> swap(Vector2D<T> const& v)
{
    return Vector2D<T>{v.y, v.x};
}

template<typename T, typename P>
Vector2D<T> rotate(Vector2D<T> const& v, P const& a)
{
    return Vector2D<T>{cos(a) * v.x - sin(a) * v.y, sin(a) * v.x + cos(a) * v.y};
}

