// This file is distributed under the MIT license.
// See the LICENSE file for details.


#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>

// Color with 8-bit RGB
struct Color {
    unsigned char r, g, b;
    Color(unsigned char r_=0, unsigned char g_=0, unsigned char b_=0)
        : r(r_), g(g_), b(b_) {}
};

// 3D vector for position, direction, etc.
struct Vec3 {
    double x, y, z;

    Vec3(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}

    // Basic arithmetic
    Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    Vec3 operator*(double b)     const { return Vec3(x * b, y * b, z * b); }
    Vec3 operator/(double b)     const { return Vec3(x / b, y / b, z / b); }
    //negation
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    // Dot product
    double dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }

    // Cross product
    Vec3 cross(const Vec3& b) const {
        return Vec3(
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        );
    }

    // Normalize vector (unit length)
    Vec3 normalize() const {
        double len = std::sqrt(dot(*this));
        return *this / len;
    }

    double length() const {
        return std::sqrt(dot(*this));
    }
};

struct Plane {
    Vec3 point;   // a point on the plane
    Vec3 normal;  // normalized normal vector
    Color color;

    Plane(const Vec3& p, const Vec3& n, const Color& c) : point(p), normal(n.normalize()), color(c) {}
};

template<typename T>
struct Bounded {
	int x_min;
	int x_max;
	T val;

	Bounded(int x_min_, int x_max_, const T& val_) :
		x_min(x_min_), x_max(x_max_), val(val_)
	{}

	bool operator<(const Bounded<T> other) {
		return x_min < other.x_min;
	}
};

#endif

