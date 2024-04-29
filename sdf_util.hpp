#ifndef __SDF_UTIL__
#define __SDF_UTIL__

#define PHI ((1.0f + sqrtf(5)) / 2.0f)
#define M_PI 3.14159265359

#include <cuda_runtime.h>
#include <glm/glm.hpp>


// Mod function that doesn't change sign on negative number input, unlike fmod. 
inline float __host__ __device__ mmod(float x, float y) 
{
	return x - y * floor(x / y);
}

inline glm::vec3 __host__ __device__ mmod(glm::vec3 x, float y) 
{
	return glm::vec3(x.x - y * floor(x.x / y), x.y - y * floor(x.y / y), x.z - y * floor(x.z / y));
}

inline glm::vec3 __host__ __device__ mmod(glm::vec3 x, glm::vec3 y) {
	return glm::vec3(x.x - y.x * floor(x.x / y.x), x.y - y.y * floor(x.y / y.y), x.z - y.z * floor(x.z / y.z));
}

// Many of the SDFs here are based on Inigo Quilez' fantastic work.
// http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm

inline float __host__ __device__ sdfUnion(float a, float b)
{
	return min(a, b);
}

inline float __host__ __device__ sdfDifference(float a, float b)
{
	return max(a, -b);
}

inline float __host__ __device__ sdfIntersection(float a, float b)
{
	return max(a, b);
}

inline float __host__ __device__ sdfSphere(glm::vec3 pos, float radius)
{
	return length(pos) - radius;
}

inline float __host__ __device__ sdfPlane(glm::vec3 pos, glm::vec3 n)
{
	return dot(pos, n);
}

inline float __host__ __device__ sdfBox(glm::vec3 pos, glm::vec3 dim) 
{
	glm::vec3 d = abs(pos) - dim;
	return min(max(d.x, max(d.y, d.z)), 0.0f) + length(glm::max(d, glm::vec3(0.0f)));
}

inline glm::vec3 __host__ __device__ rotate(glm::vec3 pos, glm::vec3 axis, float angle) 
{
	// https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
	glm::vec3 c1 = glm::vec3(
		cos(angle) + axis.x * axis.x * (1 - cos(angle)),
		axis.x * axis.y * (1 - cos(angle)) + axis.z * sin(angle),
		axis.z * axis.x * (1 - cos(angle)) - axis.y * sin(angle));
	glm::vec3 c2 = glm::vec3(
		axis.x * axis.y * (1 - cos(angle)) - axis.z * sin(angle),
		cos(angle) + axis.y * axis.y * (1 - cos(angle)),
		axis.z * axis.y * (1 - cos(angle)) + axis.x * sin(angle));
	glm::vec3 c3 = glm::vec3(
		axis.x * axis.z * (1 - cos(angle)) + axis.y * sin(angle),
		axis.y * axis.z * (1 - cos(angle)) - axis.x * sin(angle),
		cos(angle) + axis.z * axis.z * (1 - cos(angle)));

	glm::vec3 p = glm::vec3(c1.x * pos.x + c2.x * pos.y + c3.x * pos.z,
		c1.y * pos.x + c2.y * pos.y + c3.y * pos.z,
		c1.z * pos.x + c2.z * pos.y + c3.z * pos.z);

	return p;
}

inline glm::vec3 __host__ __device__ boxFold(glm::vec3 pos, glm::vec3 dim) 
{
	return glm::min(glm::max(pos, -dim), dim) * 2.0f - pos;
}

inline glm::vec3 __host__ __device__ sphereFold(glm::vec3 pos, float radius, float inner) 
{
	float r = length(pos);
	glm::vec3 p = pos;
	if (r < inner) p = p * (radius * radius) / (inner * inner);
	else if (r < radius) p = (p * radius * radius) / (r * r);
	return p;
}

inline glm::vec3 __host__ __device__ tetraFold(glm::vec3 pos) 
{
	glm::vec3 p = pos;
	if (p.x - p.y < 0) { float tmp = p.y; p.y = p.x; p.x = tmp; }
	if (p.x - p.z < 0) { float tmp = p.z; p.z = p.x; p.x = tmp; }
	if (p.y - p.z < 0) { float tmp = p.z; p.z = p.y; p.y = tmp; }
	if (p.x + p.y < 0) { float tmp = -p.y; p.y = -p.x; p.x = tmp; }
	if (p.x + p.z < 0) { float tmp = -p.z; p.z = -p.x; p.x = tmp; }
	if (p.y + p.z < 0) { float tmp = -p.z; p.z = -p.y; p.y = tmp; }
	return p;
}

inline glm::vec3 __host__ __device__ cubicFold(glm::vec3 pos) 
{
	return abs(pos);
}

inline glm::vec3 __host__ __device__ octaFold(glm::vec3 pos) 
{
	glm::vec3 p = abs(pos);
	if (p.x - p.y < 0){ float tmp = p.y; p.y = p.x; p.x = tmp; }
	if (p.x - p.z < 0){ float tmp = p.z; p.z = p.x; p.x = tmp; }
	if (p.y - p.z < 0){ float tmp = p.z; p.z = p.y; p.y = tmp; }
	return p;
}

inline glm::vec3 __host__ __device__ dodecaFold(glm::vec3 pos) 
{
	glm::vec3 p = abs(pos);
	
	glm::vec3 n1 = normalize(glm::vec3(PHI * PHI, 1.0f, -PHI));
	glm::vec3 n2 = normalize(glm::vec3(-PHI, PHI * PHI, 1.0f));
	glm::vec3 n3 = normalize(glm::vec3(1.0f, -PHI, PHI * PHI));
	glm::vec3 n4 = normalize(glm::vec3(-PHI * (1 + PHI), PHI * PHI - 1, 1 + PHI));
	glm::vec3 n5 = normalize(glm::vec3(1 + PHI, -PHI * (1 + PHI), PHI * PHI - 1));

	if (dot(p, n1) < 0) p = reflect(p, n1);
	if (dot(p, n2) < 0) p = reflect(p, n2);
	if (dot(p, n3) < 0) p = reflect(p, n3);
	if (dot(p, n4) < 0) p = reflect(p, n4);
	if (dot(p, n5) < 0) p = reflect(p, n5);
	
	return p;
}

inline glm::vec3 __host__ __device__ icosaFold(glm::vec3 pos) 
{
	glm::vec3 p = abs(pos);

	glm::vec3 n1 = normalize(glm::vec3(1.0f - PHI, -1.0f, PHI));
	glm::vec3 n2 = normalize(glm::vec3(-PHI, PHI - 1.0f, 1.0f));
	glm::vec3 n3 = normalize(glm::vec3(-PHI, PHI - 1.0f, -1.0f));
	glm::vec3 n4 = normalize(glm::vec3(1.0f - PHI, -1.0f, PHI));
	glm::vec3 n5 = normalize(glm::vec3(0, -1, 0));

	if (dot(p, n1) < 0) p = reflect(p, n1);
	if (dot(p, n2) < 0) p = reflect(p, n2);
	if (dot(p, n3) < 0) p = reflect(p, n3);
	if (dot(p, n4) < 0) p = reflect(p, n4);
	if (dot(p, n5) < 0) p = reflect(p, n5);

	return p;
}

inline float __host__ __device__ mandelbulb(glm::vec3 pos, int iterations, float bail, float power, float time = 1.0f)
{
	// http://iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
	glm::vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
	for (int i = 0; i < iterations; i++) {
		r = length(z);
		if (r > bail) break;

		// convert to polar coordinates
		float theta = asin(z.z / r);
		float phi = atan2(z.y, z.x);
		dr = pow(r, power - 1.0f) * power * dr + 1.0f;

		// scale and rotate the point
		float zr = pow(r, power);
		theta = theta * power;
		phi = phi * power;

		// convert back to cartesian coordinates
		z = zr * glm::vec3(cos(theta)*cos(phi), sin(phi)*cos(theta), sin(theta));
		z += pos;
	}

	return 0.5f*log(r)*r / dr;
}

inline glm::vec3 __host__ __device__ mandelbulbColor(glm::vec3 pos, int iterations, float bail, float power, float time = 1.0f)
{
	glm::vec3 color = glm::vec3(1, 0, 1);

	glm::vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
	for (int i = 0; i < iterations; i++) {
		r = length(z);
		if (r > bail) break;

		// convert to polar coordinates
		float theta = asin(z.z / r);
		float phi = atan2(z.y, z.x);
		dr = pow(r, power - 1.0f) * power * dr + 1.0f;

		color = glm::vec3(cos(theta * 1.1f)*cos(phi / 1.45f), sin(phi / 1.8f)*cos(theta / 1.8f), sin(theta / 1.6f));

		// scale and rotate the point
		float zr = pow(r, power);
		theta = theta * power;
		phi = phi * power;

		// convert back to cartesian coordinates
		z = zr * glm::vec3(cos(theta)*cos(phi), sin(phi)*cos(theta), sin(theta));
		z += pos;
	}

	return color;
}

inline float __host__ __device__ mengerCross(glm::vec3 pos)
{
	float a = sdfBox(glm::vec3(pos.x, pos.y, pos.z), glm::vec3(100.0f, 1.025f, 1.025f));
	float b = sdfBox(glm::vec3(pos.y, pos.z, pos.x), glm::vec3(1.025f, 100.0f, 1.025f));
	float c = sdfBox(glm::vec3(pos.z, pos.x, pos.y), glm::vec3(1.025f, 1.025f, 100.0f));
	return sdfUnion(sdfUnion(a, b), c);
}

inline float __host__ __device__ mengerBox(glm::vec3 pos, int iterations, float time = 1.0f) 
{
	glm::vec3 p = pos;
	// http://iquilezles.org/www/articles/menger/menger.htm
	float main = sdfBox(p, glm::vec3(1.0f));
	float scale = 1.0f;

	
	for (int i = 0; i < iterations; i++)
	{
		glm::vec3 a = mmod(p * scale, 2.0f) - 1.0f;
		scale *= 3.0f;
		glm::vec3 r = 1.0f - 3.0f * abs(a);
		float c = mengerCross(r) / scale;
		main = sdfIntersection(main, c);
	}
	return main;
}

inline float __host__ __device__ mengerScene(glm::vec3 pos, int iterations) 
{
	float plane = sdfPlane(pos - glm::vec3(0, -1, 0), glm::vec3(0, 1, 0));
	float mb = mengerBox(pos / 1.5f, iterations) * 1.5f;
	float mandel = mandelbulb(pos / 2.3f, 8, 4, 8.0f) * 2.3f;
	mb = sdfIntersection(mb, mandel);
	return mb;
}

inline float __host__ __device__ testFractalScene(glm::vec3 pos, float time) 
{
	//glm::vec3 p = icosaFold(pos);
	//glm::vec3 p = rotate(pos, glm::vec3(0, 1, 0), M_PI * time * 2);
	//glm::vec3 p = dodecaFold(pos) * (1.0f - time) + time * icosaFold(pos);
	return sdfUnion(mengerBox(pos, 5, time), sdfPlane(pos - glm::vec3(0, -1.25f, 0), glm::vec3(0, 1, 0)));
}

inline float __host__ __device__ mandelbulbScene(const glm::vec3& pos, float time) 
{
	//glm::vec3 p = boxFold(sphereFold(pos, 1.3f, 1.0f), glm::vec3(0.15f, 0.15f, 0.15f));;
	//p = octaFold(p);
	glm::vec3 p = dodecaFold(pos);
	float mb = mandelbulb(p / 2.3f, 8, 4, 1.0f + 9.0f * time) * 2.3f;
	return mb;
}

inline glm::vec3 __host__ __device__ mandelbulbSceneColor(const glm::vec3& pos, float time)
{
	//glm::vec3 p = boxFold(sphereFold(pos, 1.3f, 1.0f), glm::vec3(0.15f, 0.15f, 0.15f));;
	//p = octaFold(p);
	return mandelbulbColor(pos / 2.3f, 8, 4, 1.0f + 9.0f * time);
}

inline float __host__ __device__ sphereScene(const glm::vec3& pos) 
{
	glm::vec3 mod1 = glm::vec3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0f) - 1.f);
	float spheres1 = sdfSphere(mod1, 0.5f);

	glm::vec3 mod2 = glm::vec3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0) - 1.f);
	float spheres2 = sdfSphere(mod2, 0.5f);

	glm::vec3 mod3 = glm::vec3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres3 = sdfSphere(mod3, 0.5f);

	glm::vec3 mod4 = glm::vec3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres4 = sdfSphere(mod4, 0.5f);

	float spheres = sdfUnion(sdfUnion(sdfUnion(spheres1, spheres2), spheres3), spheres4);
	float plane = sdfPlane(pos - glm::vec3(0, -2.0f, 0), glm::vec3(0, 1, 0));
	return sdfUnion(spheres, plane);
}

inline glm::vec3 __host__ __device__ sphereColor(const glm::vec3& pos) 
{
	glm::vec3 mod1 = glm::vec3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0f) - 1.f);
	float spheres1 = sdfSphere(mod1, 0.5f);

	glm::vec3 mod2 = glm::vec3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, fmodf(pos.z, 2.0) - 1.f);
	float spheres2 = sdfSphere(mod2, 0.5f);

	glm::vec3 mod3 = glm::vec3(fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres3 = sdfSphere(mod3, 0.5f);

	glm::vec3 mod4 = glm::vec3(-fmodf(pos.x, 2.0) - 1.f, pos.y + 1.5f, -fmodf(pos.z, 2.0) - 1.f);
	float spheres4 = sdfSphere(mod4, 0.5f);

	float spheres = sdfUnion(sdfUnion(sdfUnion(spheres1, spheres2), spheres3), spheres4);
	float plane = sdfPlane(pos - glm::vec3(0, -2.0f, 0), glm::vec3(0, 1, 0));
	
	if (plane < spheres) return glm::vec3(1.0f, 0.3f, 0.1f);
	return glm::vec3(0.85f);
}

inline float __host__ __device__ cornellBoxScene(const glm::vec3& pos)
{
	float rightplane = sdfBox(pos - glm::vec3(-2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
	float leftplane = sdfBox(pos - glm::vec3(2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
	float backplane = sdfBox(pos - glm::vec3(0.0f, 0.0, 1.0), glm::vec3(3.0f, 2.f, 0.05f));
	float topplane = sdfBox(pos - glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(2.05f, 0.05f, 1.0f));
	float plane = sdfPlane(pos - glm::vec3(0, -1.5f, 0), glm::vec3(0, 1, 0));

	float smallSphere = sdfSphere(pos - glm::vec3(0.7f, -1.0f, 0.6f), 0.5f);
	float menger = mengerBox((pos - glm::vec3(-0.7f, -1.0f, 0.2f)) / 0.5f, 5) * 0.5f;
	float objs = sdfUnion(smallSphere, menger);

	return sdfUnion(sdfUnion(sdfUnion(sdfUnion(sdfUnion(rightplane, plane), leftplane), backplane), topplane), objs);
}

inline glm::vec3 __host__ __device__ cornellBoxColor(const glm::vec3& pos) {
	float rightplane = sdfBox(pos - glm::vec3(-2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
	float leftplane = sdfBox(pos - glm::vec3(2.0f, 0.0, 0.0), glm::vec3(0.05f, 2.f, 1.0f));
	float backplane = sdfBox(pos - glm::vec3(0.0f, 0.0, 1.0), glm::vec3(3.0f, 2.f, 0.05f));
	float topplane = sdfBox(pos - glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(2.05f, 0.05f, 1.0f));
	float plane = sdfPlane(pos - glm::vec3(0, -1.5f, 0), glm::vec3(0.0f, 1.0f, 0.0f));

	float smallSphere = sdfSphere(pos - glm::vec3(0.7f, -1.0f, 0.6f), 0.5f);
	float bigSphere = sdfSphere(pos - glm::vec3(-0.7f, -0.9f, 0.2f), 0.6f);
	float spheres = sdfUnion(bigSphere, smallSphere);

	float whitewalls = sdfUnion(sdfUnion(topplane, plane), spheres);

	if (leftplane < rightplane && leftplane < whitewalls && leftplane < backplane) {
		return glm::vec3(0.05f, .5f, 0.8f);
	}
	else if (backplane < rightplane && backplane < whitewalls) {
		return glm::vec3(1.0f, 0.8f, 0.1f);
	}
	else if (rightplane < whitewalls) {
		return glm::vec3(.9f, 0.2f, 0.4f);
	}

	return glm::vec3(0.85f);
}

#endif