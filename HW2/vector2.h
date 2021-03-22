#include <iostream>

#ifndef VECTOR2_H
#define VECTOR2_H

template<typename T>
class Vector2D
{   public:
        T x, y;
    Vector2D<T>& operator+=(Vector2D<T> const&);
    Vector2D<T>& operator-=(Vector2D<T> const&);
};

template<typename T>
std::ostream& operator<<(std::ostream&, Vector2D<T> const&);

template<typename T>
std::istream& operator>>(std::istream&, Vector2D<T>&);

template<typename T>
T dot(Vector2D<T> const&, Vector2D<T> const&);

template<typename T>
T length(Vector2D<T> const&);

template<typename T>
T sqlength(Vector2D<T> const&);

template<typename T>
Vector2D<T> normalize(Vector2D<T> const&);

template<typename T>
Vector2D<T> operator+(Vector2D<T> const&, Vector2D<T> const&);

template<typename T>
Vector2D<T> operator-(Vector2D<T> const&, Vector2D<T> const&);

template<typename T, typename P>
Vector2D<T> operator*(P const&, Vector2D<T> const&);

template<typename T, typename P>
Vector2D<T> operator*(Vector2D<T> const&, P const&);

template<typename T, typename P>
Vector2D<T> operator/(Vector2D<T> const&, P const&);

template<typename T>
Vector2D<T> mirror_x(Vector2D<T> const&);

template<typename T>
Vector2D<T> mirror_y(Vector2D<T> const&);

template<typename T>
Vector2D<T> swap(Vector2D<T> const&);

template<typename T, typename P>
Vector2D<T> rotate(Vector2D<T> const&, P const&);

#include "vector2.tpp"


#endif