#include <kernels.h>

#ifdef __APPLE__
    #include <mach/clock.h>
    #include <mach/mach.h>

    clock_serv_t cclock;
	mach_timespec_t tick_clockData;
	mach_timespec_t tock_clockData;

	#define TICK() { if (print_kernel_timing) {\
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		clock_get_time(cclock, &tick_clockData);\
		mach_port_deallocate(mach_task_self(), cclock);\
	}}

	#define TOCK(str,size) { if (print_kernel_timing) {\
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);\
		clock_get_time(cclock, &tock_clockData);\
		mach_port_deallocate(mach_task_self(), cclock);\
		std::cerr<< str << " ";\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
        std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;\
    }}
#else
    struct timespec tick_clockData;
    struct timespec tock_clockData;

	#define TICK() { if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}}

	#define TOCK(str,size)  {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tock_clockData); std::cerr<< str << " ";\
		if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
        std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;\
    }}
#endif

bool print_kernel_timing = false;

// input once
float * gaussian;

// inter-frame
Volume volume;
float3 * vertex;
float3 * normal;

// intra-frame
TrackData * trackingResult;
float* reductionoutput;
float ** ScaledDepth;
float * floatDepth;
Matrix4 oldPose;
Matrix4 raycastPose;
float3 ** inputVertex;
float3 ** inputNormal;

Kfusion::~Kfusion() {
    
        free(floatDepth);
        free(trackingResult);
    
        free(reductionoutput);
        for (unsigned int i = 0; i < iterations.size(); ++i) {
            free(ScaledDepth[i]);
            free(inputVertex[i]);
            free(inputNormal[i]);
        }
        free(ScaledDepth);
        free(inputVertex);
        free(inputNormal);
    
        free(vertex);
        free(normal);
        free(gaussian);
    
        volume.release();
    }
    void Kfusion::reset() {
        initVolumeKernel(volume);
    }

void Kfusion::languageSpecificConstructor() {
    if (getenv("KERNEL_TIMINGS")) {
        print_kernel_timing = true;
    }

    // internal buffers to initialize
    reductionoutput = (float*) calloc(sizeof(float) * 8 * 32, 1);

    ScaledDepth = (float**) calloc(sizeof(float*) * iterations.size(), 1);
    inputVertex = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);
    inputNormal = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);

    for (unsigned int i = 0; i < iterations.size(); ++i) {
        ScaledDepth[i] = (float*)  calloc(sizeof(float)*(computationSize.x*computationSize.y)/(int)pow(2,i),1);
        inputVertex[i] = (float3*) calloc(sizeof(float3)*(computationSize.x*computationSize.y)/(int)pow(2,i),1);
        inputNormal[i] = (float3*) calloc(sizeof(float3)*(computationSize.x*computationSize.y)/(int)pow(2,i),1);
    }

    floatDepth = (float*) calloc(sizeof(float)*computationSize.x*computationSize.y,1);
    vertex = (float3*)    calloc(sizeof(float3)*computationSize.x*computationSize.y,1);
    normal = (float3*)    calloc(sizeof(float3)*computationSize.x*computationSize.y,1);
    trackingResult = (TrackData*) calloc(sizeof(TrackData)*computationSize.x*computationSize.y,1);

    // ********* BEGIN : Generate the gaussian *************
    size_t gaussianS = radius*2 + 1;
    gaussian = (float*) calloc(gaussianS*sizeof(float),1);

    // Todo: STL here
    for (unsigned int i=0; i<gaussianS; i++) {
        gaussian[i] = expf(-((i-2) * (i-2)) / (2 * delta * delta));
    }
    // ********* END : Generate the gaussian *************

    volume.init(volumeResolution, volumeDimensions);
    reset();
}
    
void initVolumeKernel(Volume volume) {}

void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize, const Matrix4 invK) {}

void reduceKernel(float * out, TrackData* J, const uint2 Jsize, const uint2 size) {}

// TrackData includes the errors (for far I am to the pixel in front of me)
void trackKernel(TrackData* output, const float3* inVertex,
		const float3* inNormal, uint2 inSize, const float3* refVertex,
		const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
		const Matrix4 view, const float dist_threshold,
		const float normal_threshold) {}

void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize) {}

void halfSampleRobustImageKernel(float* out, const float* in, uint2 imageSize,
		const float e_d, const int r) {}

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold) {

    return false;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output,
        uint2 imageSize, float track_threshold) {

    return false;
}

// Given the point cloud, how to update the volume
// In-place transformation (side-effect to an argument) because we (usually) don't have enough memory
// to store two Volumes in RAM
void integrateKernel(Volume vol, const float* depth, uint2 imageSize,
		const Matrix4 invTrack, const Matrix4 K, const float mu,
		const float maxweight) {}

void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
		const Volume integration, const Matrix4 view, const float nearPlane,
        const float farPlane, const float step, const float largestep) {}
        
void computeFrame(const ushort * inputDepth, const uint2 inputSize,
        float4 k, uint integration_rate, uint tracking_rate,
        float icp_threshold, float mu, const uint frame);



bool Kfusion::preprocessing(const ushort * inputDepth, const uint2 inputSize) {
    mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
    bilateralFilterKernel(ScaledDepth[0], floatDepth, computationSize, gaussian, e_delta, radius);
	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate, uint frame) {

    return false;
}

bool Kfusion::raycasting(float4 k, float mu, uint frame) {

    return false;
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame) {

    return false;
}

void Kfusion::dumpVolume(const char *filename) {}

void Kfusion::renderVolume(uchar4 * out, const uint2 outputSize, int frame, int rate, float4 k, float mu) {}

void Kfusion::renderTrack(uchar4 * out, const uint2 outputSize) {}

void Kfusion::renderDepth(uchar4* out, uint2 outputSize) {}

void synchroniseDevices() {}

