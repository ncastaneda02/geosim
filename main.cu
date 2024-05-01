#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>

#include "sdf_util.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLOCK_SIZE 8
#define BOUNCES 4
#define SAMPLES 8 // Total number of samples is SAMPLES*SAMPLES
#define EPS 1e-5
#define MINDIST 1.8e-3
#define PUSH 0.0036f
#define FRAMES 60

// Purely random pixel sample
inline glm::vec2 __device__ getRandomSample(curandState* state) 
{
	return glm::vec2(curand_uniform(state), curand_uniform(state));
}

// Random sample in nth subpixel
inline glm::vec2 __device__ getJitteredSample(int n, curandState* state) {
	glm::vec2 rand_vec = glm::vec2(curand_uniform(state) * (1.0f / SAMPLES), curand_uniform(state) * (1.0f / SAMPLES));
	glm::vec2 result = glm::vec2((n % SAMPLES) * 1.0f / SAMPLES, (n / SAMPLES) * 1.0f / SAMPLES);
	return result + rand_vec;
}

glm::vec3 __device__ orient(const glm::vec3& n, curandState* state) 
{
	// TODO: Change this whole function to be hyperbolic
	// rejection sampling hemisphere
	float x = 1.0f, y = 1.0f;

	while (x * x + y * y > 1.0f) 
	{
		x = (curand_uniform(state) - 0.5f) * 2.0f;
		y = (curand_uniform(state) - 0.5f) * 2.0f;
	}
	float z = sqrtf(1 - x * x - y * y);
	glm::vec3 in = normalize(glm::vec3(x, y, z));

	// Create vector that is not the same as n
	glm::vec3 absn = glm::abs(n);
	glm::vec3 q = n;
	if (absn.x <= absn.y && absn.x <= absn.z)  q.x = 1;
	else if (absn.y <= absn.x && absn.y <= absn.z) q.y = 1;
	else q.z = 1;

	// Basis creation, result is just a rolled out matrix multiplication of basis matrix and in vector
	glm::vec3 t = normalize(cross(n, q));
	glm::vec3 b = normalize(cross(n, t));
	return normalize(glm::vec3(t.x * in.x + b.x * in.y + n.x * in.z,
								 t.y * in.x + b.y * in.y + n.y * in.z,
								 t.z * in.x + b.z * in.y + n.z * in.z));
}

struct Hit 
{
	bool isHit = 0;
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 color;
};

struct Camera 
{
	glm::vec3 pos;
	glm::vec3 dir;
	float invhalffov;
	float maxdist = 10.0f;
	glm::vec3 up;
	glm::vec3 side;
};


// Distance estimation function
float __device__ DE(const glm::vec3& pos, float time) 
{
	//return mandelbulbScene(pos, time);
	//return sphereScene(pos);
	return cornellBoxScene(pos);
	//return mengerScene(pos, 6);
	//return testFractalScene(pos, time);
}

glm::vec3 __device__ sceneColor(const glm::vec3& pos, float time) 
{
	//return glm::vec3(0.85f);
	//return mandelbulbSceneColor(pos, time);
	//return sphereColor(pos);
	return cornellBoxColor(pos);
}

// Ray marching function, similar to intersect function in normal ray tracers
__device__ Hit march(const glm::vec3& orig, const glm::vec3& direction, float time) 
{
	// TODO: use hyperbolic length and norm
	float totaldist = 0.0f;
	float maxdist = length(direction);
	glm::vec3 pos = orig; glm::vec3 dir = normalize(direction);
	glm::vec3 col = glm::vec3(0.85f, 0.85f, 0.85f);

	Hit hit;

	while (totaldist < maxdist) 
	{
		// TODO: Change distance estimator to use euclidean distance
		float t = DE(pos, time);

		// If distance is less than this then it is a hit.
		if (t < MINDIST) 
		{
			// Calculate gradient (normal)
			// TODO: Change vector gradient for hyperbolic changes, also use hyperbolic norm
			float fx = (DE(glm::vec3(pos.x + EPS, pos.y, pos.z), time) - DE(glm::vec3(pos.x - EPS, pos.y, pos.z), time));
			float fy = (DE(glm::vec3(pos.x, pos.y + EPS, pos.z), time) - DE(glm::vec3(pos.x, pos.y - EPS, pos.z), time));
			float fz = (DE(glm::vec3(pos.x, pos.y, pos.z + EPS), time) - DE(glm::vec3(pos.x, pos.y, pos.z - EPS), time));
			glm::vec3 normal = normalize(glm::vec3(fx, fy, fz));
			// faceforward
			if (dot(-dir, normal) < 0) normal = -normal;

			// create hit
			hit.isHit = true;
			hit.pos = pos;
			hit.normal = normal;
			// TODO: tweak scene color maybe?
			hit.color = sceneColor(pos, time);
			return hit;
		}

		// step forwards by t if no hit
		totaldist += t;
		pos += t * dir;
	}

	return hit;
}

// Path tracing function
__device__ glm::vec4 trace(const glm::vec3& orig, const glm::vec3& direction, curandState* state, float time)
{
	// TODO: make sure this doesnt fuck everything up
	// also use hyperbolic length
	float raylen = length(direction);
	glm::vec3 dir = direction;
	glm::vec3 o = orig;
	glm::vec3 p = glm::vec3(0.0f); glm::vec3 n = glm::vec3(0.0f);
	glm::vec3 mask = glm::vec3(1.0f); glm::vec3 color = glm::vec3(0.0f);

	Hit rayhit = march(o, dir, time);
	
	for (int i = 0; i < BOUNCES + 1; i++) 
	{
		if (rayhit.isHit) 
		{
			p = rayhit.pos; n = rayhit.normal;
			// Create new ray direction
			// TODO: Change new ray direction to flow along geodesic
			glm::vec3 d = orient(n, state);
			o = p + n * PUSH;
			mask *= rayhit.color;
			dir = raylen * d;
			// Fire new ray if there are bounces left
			if (i < BOUNCES) rayhit = march(o, dir, time);
		}
		else if (i == 0) return glm::vec4(glm::vec3(0.0f), 0.0f); // black background
		else 
		{
			color += glm::vec3(1.0f) * mask; // add color when light (sky) is hit
			break;
		}
	}
	
	return glm::vec4(color, 1.0f);
}

__global__ void render(int width, int height, float* result, Camera cam, unsigned long long seed, float time)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	glm::vec4 color = glm::vec4(0.0f);

	int block = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned long long idx = block * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	
	curandState state;
	curand_init(idx + seed, 0, 0, &state);

	glm::vec2 samp = glm::vec2(x, y);
	
	for (int i = 0; i < SAMPLES * SAMPLES; i++) {
		//glm::vec2 offset = getRandomSample(&state);
		glm::vec2 offset = getJitteredSample(i, &state);
		glm::vec2 sample = samp + offset;
		// TODO: CHANGE FLOW OF RAY OUT OF CAMERA TO BE ALONG GEODESIC
		float nx = (sample.x / float(width) - 0.5f) * 2.0f;
		float ny = -(sample.y / float(height) - 0.5f) * 2.0f;
		ny *= float(height) / float(width);
		glm::vec3 raydir = normalize(cam.side * nx + cam.up * ny + cam.dir * cam.invhalffov);
		color += trace(cam.pos, raydir * cam.maxdist, &state, time);
	}
	
	color /= (SAMPLES * SAMPLES);

	result[x * 4 + 4 * y * width + 0] = color.x;
	result[x * 4 + 4 * y * width + 1] = color.y;
	result[x * 4 + 4 * y * width + 2] = color.z;
	result[x * 4 + 4 * y * width + 3] = color.w;
}

void saveImage(std::string path, int width, int height, const float colors[]) 
{
    std::vector<unsigned char> output;
    output.resize(4 * width * height);
    const int channels = 4;
    for (int i = 0; i < width * height; i++)
    {
        output[i * 4 + 0] = static_cast<unsigned char>(std::fmax(std::fmin(colors[i * 4 + 0] * 255, 255), 0));
        output[i * 4 + 1] = static_cast<unsigned char>(std::fmax(std::fmin(colors[i * 4 + 1] * 255, 255), 0));
        output[i * 4 + 2] = static_cast<unsigned char>(std::fmax(std::fmin(colors[i * 4 + 2] * 255, 255), 0));
        output[i * 4 + 3] = static_cast<unsigned char>(std::fmax(std::fmin(colors[i * 4 + 3] * 255, 255), 0));
    }

    // Use stbi_write_png to save the image
    int stride_in_bytes = width * channels; // Assuming no padding between rows
    int result = stbi_write_png(path.c_str(), width, height, channels, output.data(), stride_in_bytes);
    if (!result) {
	    std::cout << "Error when outputting png. code " << result << std::endl;
    }

    return;
}

void moveCamera(Camera &cam, double theta, int r) {
	double newx = r * cos(theta);
	double newz = r * sin(theta);
	cam.pos.x = newx;
	cam.pos.z = newz;
	cam.dir.x = -newx;
	cam.dir.z = -newz;
	cam.dir = normalize(cam.dir);
	cam.side = normalize(cross(cam.dir, glm::vec3(0, 1, 0)));
	cam.up = normalize(cross(cam.side, cam.dir));
	float fov = 128.0f / 180.0f * float(M_PI);
	cam.invhalffov = 1.0f / std::tan(fov / 2.0f);
}

int main()
{
	int width = 1920, height = 1080;
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocks(width / threads.x + 1, height / threads.y + 1);
	
	Camera cam;
	cam.pos = glm::vec3(0.0f, 1.0f, -4.0f);
	//cam.pos = glm::vec3(-1.0f, 1.5f, -3.0f);
	//cam.pos = glm::vec3(0, 0.4f, -1.4f);
	cam.dir = normalize(glm::vec3(0.0f, 0.0f, 1.0f));
	cam.side = normalize(cross(cam.dir, glm::vec3(0, 1, 0)));
	cam.up = normalize(cross(cam.side, cam.dir));
	float fov = 128.0f / 180.0f * float(M_PI);
	cam.invhalffov = 1.0f / std::tan(fov / 2.0f);
	double radius = 4.0;
	double step = 2.0 * M_PI / (FRAMES - 2.0);
	std::cout << "step size: " << step << std::endl;
	for (int i = 0; i < FRAMES; i++) {
		auto frame_start = std::chrono::high_resolution_clock::now();
		float *deviceImage;
		cudaMalloc(&deviceImage, 4 * width * height * sizeof(float));
		
		unsigned long long seed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		
		float t = 1.0f;
		if (FRAMES > 1) t = float(i) / (FRAMES - 1.0f);

		render << <blocks, threads >> >(width, height, deviceImage, cam, seed, t);
		
		float *hostImage = (float*)malloc(4 * width * height * sizeof(float));
		cudaMemcpy(hostImage, deviceImage, 4 * width * height * sizeof(float), cudaMemcpyDeviceToHost);
		std::string imageName = "../renders/render_" + std::to_string(i + 1) + ".png";
		saveImage(imageName, width, height, hostImage);
		cudaFree(deviceImage);
		free(hostImage);

    	auto frame_time =std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - frame_start);
		std::cout << "Frame " << (i + 1) << " done! Took " << frame_time.count() << "ms to generate. Saved as " << imageName << "." << std::endl;
		moveCamera(cam, step * i - (M_PI / 2.0), 3);
	}

	return 0;
}
