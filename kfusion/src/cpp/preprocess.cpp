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



void mm2metersKernel(float * out, uint2 outSize, const ushort * in, uint2 inSize) {
	TICK();
	// Check for unsupported conditions
	if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize.x / outSize.x;
	unsigned int y;

	#pragma omp parallel for shared(out), private(y)
	for (y = 0; y < outSize.y; y++)
		for (unsigned int x = 0; x < outSize.x; x++) {
			out[x + outSize.x * y] = in[x * ratio + inSize.x * y * ratio]
					/ 1000.0f;
		}
	TOCK("mm2metersKernel", outSize.x * outSize.y);
}

void bilateralFilterKernel(float* out, const float* in, uint2 size, const float * gaussian, float e_d, int r) {
	TICK()
	uint y;
	float e_d_squared_2 = e_d * e_d * 2;

	#pragma omp parallel for shared(out),private(y)
		for (y = 0; y < size.y; y++) {
			for (uint x = 0; x < size.x; x++) {
				uint pos = x + y * size.x;
				if (in[pos] == 0) {
					out[pos] = 0;
					continue;
				}

				float sum = 0.0f;
				float t = 0.0f;

				const float center = in[pos];

				for (int i = -r; i <= r; ++i) {
					for (int j = -r; j <= r; ++j) {
						uint2 curPos = make_uint2(clamp(x + i, 0u, size.x - 1), clamp(y + j, 0u, size.y - 1));
						const float curPix = in[curPos.x + curPos.y * size.x];
						if (curPix > 0) {
							const float mod = sq(curPix - center);
							const float factor = gaussian[i + r]
									* gaussian[j + r]
									* expf(-mod / e_d_squared_2);
							t += factor * curPix;
							sum += factor;
						}
					}
				}
				out[pos] = t / sum;
			}
		}
		TOCK("bilateralFilterKernel", size.x * size.y);
}
