#ifndef _KERNELS_STL_
#define _KERNELS_STL_

#include <cstdlib>
#include <commons.h>

////////////////////////// COMPUTATION KERNELS PROTOTYPES //////////////////////

void initVolumeKernel(Volume volume);

void bilateralFilterKernel(std::vector<float> &out, const std::vector<float> in, uint2 inSize, const float * gaussian, float e_d, int r);
//void bilateralFilterKernel(float* out, const float* in, uint2 inSize, const float * gaussian, float e_d, int r);

void depth2vertexKernel(std::vector<float3> &vertex, const std::vector<float> depth, uint2 imageSize, const Matrix4 invK);
//void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize, const Matrix4 invK);

void reduceKernel(std::vector<float> &out, std::vector<TrackData> trackData, const uint2 Jsize, const uint2 out_size);
//void reduceKernel(float * out, TrackData* J, const uint2 Jsize, const uint2 size);


void trackKernel(std::vector<TrackData> &output, const std::vector<float3> inVertex,
		const std::vector<float3> inNormal, uint2 inSize, const float3* refVertex,
		const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
		const Matrix4 view, const float dist_threshold,
		const float normal_threshold);
/*void trackKernel(TrackData* output, const std::vector<float3> inVertex,
		const std::vector<float3> inNormal, uint2 inSize, const float3* refVertex,
		const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
		const Matrix4 view, const float dist_threshold,
		const float normal_threshold);*/

void vertex2normalKernel(std::vector<float3> &out, const std::vector<float3> in, uint2 imageSize);
//void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize);

void mm2metersKernel(std::vector<float> &out, uint2 outSize, const ushort * in, uint2 inSize);
//void mm2metersKernel(float * out, uint2 outSize, const ushort * in, uint2 inSize);

void halfSampleRobustImageKernel(std::vector<float> &out, std::vector<float> in, uint2 imageSize, const float e_d, const int r);
//void halfSampleRobustImageKernel(float* out, const float* in, uint2 imageSize, const float e_d, const int r);

bool updatePoseKernel(Matrix4 & pose, const std::vector<float> output, float icp_threshold);
//bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold);

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const std::vector<float> output, uint2 imageSize, float track_threshold);
//bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output, uint2 imageSize, float track_threshold);

void integrateKernel(Volume vol, const float* depth, uint2 imageSize,
		const Matrix4 invTrack, const Matrix4 K, const float mu,
		const float maxweight);

void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
		const Volume integration, const Matrix4 view, const float nearPlane,
		const float farPlane, const float step, const float largestep);

////////////////////////// RENDER KERNELS PROTOTYPES //////////////////////

void renderDepthKernel(uchar4* out, float * depth, uint2 depthSize, const float nearPlane, const float farPlane);

void renderNormaKernell(uchar3* out, const float3* normal, uint2 normalSize);

//void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize);
void renderTrackKernel(uchar4* out, const std::vector<TrackData> data, uint2 outSize);

void renderVolumeKernel(uchar4* out, const uint2 depthSize, const Volume volume,
		const Matrix4 view, const float nearPlane, const float farPlane,
		const float step, const float largestep, const float3 light,
		const float3 ambient);

////////////////////////// MULTI-KERNELS PROTOTYPES //////////////////////
float4 raycast(const Volume volume, const uint2 pos, const Matrix4 view,
        const float nearPlane, const float farPlane, const float step,
        const float largestep);

void computeFrame(Volume & integration, float3 * vertex, float3 * normal,
		std::vector<TrackData> trackingResult, Matrix4 & pose, const float * inputDepth,
		const uint2 inputSize, const float * gaussian,
        const std::vector<int> iterations, float4 k, const uint frame);

void init();

void clean();

#endif
