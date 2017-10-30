#include <time.h>

#include <vector_types.h>
#include <commons.h>

extern bool print_kernel_timing;
extern struct timespec tick_clockData;
extern struct timespec tock_clockData;

#ifdef __APPLE__
	
	#define TICK()    {if (print_kernel_timing) {\
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		clock_get_time(cclock, &tick_clockData);\
		mach_port_deallocate(mach_task_self(), cclock);\
		}}

	#define TOCK(str,size)  {if (print_kernel_timing) {\
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		clock_get_time(cclock, &tock_clockData);\
		mach_port_deallocate(mach_task_self(), cclock);\
		std::cerr<< str << " ";\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
		std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}
#else
	
	#define TICK() { if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);} }

	#define TOCK(str,size)  {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tock_clockData); std::cerr<< str << " ";\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
		std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}
#endif



void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize, const float e_d, const int r) {
	
	TICK();
	uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
	unsigned int y;
	
	#pragma omp parallel shared(out), private(y)
	for (y = 0; y < outSize.y; y++) {
		for (unsigned int x = 0; x < outSize.x; x++) {
			uint2 pixel = make_uint2(x, y);
			const uint2 centerPixel = 2 * pixel;

			float sum = 0.0f;
			float t = 0.0f;
			const float center = in[centerPixel.x + centerPixel.y * inSize.x];
			for (int i = -r + 1; i <= r; ++i) {
				for (int j = -r + 1; j <= r; ++j) {
					uint2 cur = make_uint2(
						clamp(make_int2(centerPixel.x + j, centerPixel.y + i), make_int2(0),
							make_int2(2 * outSize.x - 1, 2 * outSize.y - 1)));
					float current = in[cur.x + cur.y * inSize.x];
					if (fabsf(current - center) < e_d) {
						sum += 1.0f;
						t += current;
					}
				}
			}
			out[pixel.x + pixel.y * outSize.x] = t / sum;
		}
	}
	TOCK("halfSampleRobustImageKernel", outSize.x * outSize.y);
}

void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize, const Matrix4 invK) {
	TICK();
	unsigned int x, y;

    #pragma omp parallel for shared(vertex), private(x, y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {

			if (depth[x + y * imageSize.x] > 0) {
				vertex[x + y * imageSize.x] = depth[x + y * imageSize.x]
						* (rotate(invK, make_float3(x, y, 1.f)));
			} else {
				vertex[x + y * imageSize.x] = make_float3(0);
			}
		}
	}
	TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

void trackKernel(TrackData* output, const float3* inVertex,
	const float3* inNormal,  uint2 inSize,  const float3* refVertex,
	const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
	const Matrix4 view, const float dist_threshold,
	const float normal_threshold) {

	TICK();
	uint2 pixel = make_uint2(0, 0);
	unsigned int pixely, pixelx;
	
	#pragma omp parallel for shared(output), private(pixel,pixelx,pixely)
	for (pixely = 0; pixely < inSize.y; pixely++) {
		for (pixelx = 0; pixelx < inSize.x; pixelx++) {
			pixel.x = pixelx;
			pixel.y = pixely;

			TrackData & row = output[pixel.x + pixel.y * refSize.x];

			if (inNormal[pixel.x + pixel.y * inSize.x].x == KFUSION_INVALID) {
				row.result = -1;
				continue;
			}

			const float3 projectedVertex = Ttrack * inVertex[pixel.x + pixel.y * inSize.x];
			const float3 projectedPos = view * projectedVertex;
			const float2 projPixel = make_float2(
					projectedPos.x / projectedPos.z + 0.5f,
					projectedPos.y / projectedPos.z + 0.5f);
			if (projPixel.x < 0 || projPixel.x > refSize.x - 1
					|| projPixel.y < 0 || projPixel.y > refSize.y - 1) {
				row.result = -2;
				continue;
			}

			const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
			const float3 referenceNormal = refNormal[refPixel.x + refPixel.y * refSize.x];

			if (referenceNormal.x == KFUSION_INVALID) {
				row.result = -3;
				continue;
			}

			const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x] - projectedVertex;
			const float3 projectedNormal = rotate(Ttrack, inNormal[pixel.x + pixel.y * inSize.x]);

			if (length(diff) > dist_threshold) {
				row.result = -4;
				continue;
			}
			if (dot(projectedNormal, referenceNormal) < normal_threshold) {
				row.result = -5;
				continue;
			}
			row.result = 1;
			row.error = dot(referenceNormal, diff);
			((float3 *) row.J)[0] = referenceNormal;
			((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
		}
	}
	TOCK("trackKernel", inSize.x * inSize.y);
}
