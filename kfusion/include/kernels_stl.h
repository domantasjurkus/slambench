#ifndef _KERNELS_STL_
#define _KERNELS_STL_

#include <cstdlib>
#include <commons.h>

////////////////////////// COMPUTATION KERNELS PROTOTYPES //////////////////////

void initVolumeKernel(Volume volume);

void mm2metersKernel(std::vector<float> &out,
		uint2 outSize,
		const std::vector<ushort> in,
		uint2 inSize);

void bilateralFilterKernel(std::vector<float> &out,
		const std::vector<float> in,
		const std::vector<uint> pixels,
		uint2 inSize,
		const std::vector<float> gaussian,
		float e_d,
		int r);

void halfSampleRobustImageKernel(std::vector<float> &out,
		std::vector<float> in,
		uint2 imageSize,
		const float e_d,
		const int r);

void depth2vertexKernel(std::vector<float3> &vertex,
		const std::vector<float> depth,
		const std::vector<uint> pixels,
		uint2 imageSize,
		const Matrix4 invK);

void vertex2normalKernel(std::vector<float3> &out,
		const std::vector<float3> in,
		uint2 imageSize);

void trackKernel(std::vector<TrackData> &output,
		const std::vector<float3> inVertex,
		const std::vector<float3> inNormal,
		uint2 inSize,
		const std::vector<std::pair<float3, float3>> vertex_normals,
		uint2 refSize,
		const Matrix4 Ttrack,
		const Matrix4 view,
		const float dist_threshold,
		const float normal_threshold);

void reduceKernel(std::vector<float> &out,
		std::vector<TrackData> trackData,
		const uint2 Jsize,
		const uint2 out_size);

void integrateKernel(Volume vol,
		const std::vector<float> depth,
		uint2 imageSize,
		const Matrix4 invTrack,
		const Matrix4 K,
		const float mu,
		const float maxweight);

bool updatePoseKernel(Matrix4 & pose, const std::vector<float> output, float icp_threshold);
//bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold);

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const std::vector<float> output, uint2 imageSize, float track_threshold);
//bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output, uint2 imageSize, float track_threshold);

void raycastKernel(std::vector<std::pair<float3, float3>> &vertex_normals,
		uint2 inputSize,
		const Volume integration,
		const std::vector<uint> pixels,
		const Matrix4 view,
		const float nearPlane,
		const float farPlane,
		const float step,
		const float largestep);

////////////////////////// RENDER KERNELS PROTOTYPES //////////////////////
//template <class T>
//void renderDepthKernel(std::vector<uchar4> out, std::vector<float> depth, uint2 depthSize, const float nearPlane, const float farPlane, sycl::sycl_execution_policy<T> policy);
void renderDepthKernel(std::vector<uchar4> out, std::vector<float> depth, uint2 depthSize, const float nearPlane, const float farPlane);

//void renderNormaKernell(uchar3* out, const float3* normal, uint2 normalSize);

//void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize);
void renderTrackKernel(std::vector<uchar4> out, const std::vector<TrackData> data, uint2 outSize);

void renderVolumeKernel(std::vector<uchar4> out,
		const uint2 depthSize,
		const Volume volume,
		const std::vector<uint> pixels,
		const Matrix4 view,
		const float nearPlane,
		const float farPlane,
		const float step,
		const float largestep,
		const float3 light,
		const float3 ambient);

////////////////////////// MULTI-KERNELS PROTOTYPES //////////////////////
float4 raycast(const Volume volume,
		const uint2 pos,
		const Matrix4 view,
		const float3 origin,
        const float nearPlane,
		const float farPlane,
		const float step,
        const float largestep);

void computeFrame(Volume & integration, float3 * vertex, float3 * normal,
		std::vector<TrackData> trackingResult, Matrix4 & pose, const float * inputDepth,
		const uint2 inputSize, const float * gaussian,
        const std::vector<int> iterations, float4 k, const uint frame);

void init();

void clean();

#endif
