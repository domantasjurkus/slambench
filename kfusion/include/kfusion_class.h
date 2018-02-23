#ifndef _KFUSION_
#define _KFUSION_

#include <cstdlib>
#include <commons.h>

class Kfusion {
private:
	uint2 computationSize;
	float step;
	Matrix4 pose;
	Matrix4 *viewPose;
	float3 volumeDimensions;
	uint3 volumeResolution;
	std::vector<int> iterations;
	bool _tracked;
	bool _integrated;
	float3 _initPose;

	void raycast(uint frame, const float4& k, float mu);

public:
	void kfusion_init(uint3 volumeResolution, float3 volumeDimensions, std::vector<int> & pyramid) {
		this->volumeDimensions = volumeDimensions;
		this->volumeResolution = volumeResolution;
		
		this->iterations.clear();
		this->iterations = std::move(pyramid);
		
		this->step = min(volumeDimensions) / max(volumeResolution);
		this->languageSpecificConstructor();
	}

	Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeDimensions,
			float3 initPose, std::vector<int> & pyramid) :
			computationSize(make_uint2(inputSize.x, inputSize.y)) {

		kfusion_init(volumeResolution, volumeDimensions, pyramid);
		
		this->_initPose = initPose;
		pose = toMatrix4(TooN::SE3<float>(
			TooN::makeVector(initPose.x, initPose.y, initPose.z, 0,0, 0)
		));
		viewPose = &pose;
	}

	// initPose with initial orientation
	Kfusion(uint2 inputSize, uint3 volumeResolution, float3 volumeDimensions,
			Matrix4 initPose, std::vector<int> & pyramid) :
			computationSize(make_uint2(inputSize.x, inputSize.y)) {

		kfusion_init(volumeResolution, volumeDimensions, pyramid);

		this->_initPose = getPosition();
		pose = initPose;
		viewPose = &pose;
	}

	void languageSpecificConstructor();
	~Kfusion();

	void reset();
	bool getTracked() {
		return (_tracked);
	}
	bool getIntegrated() {
		return (_integrated);
	}
	float3 getPosition() {
		//std::cerr << "InitPose =" << _initPose.x << "," << _initPose.y  <<"," << _initPose.z << "    ";
		//std::cerr << "pose =" << pose.data[0].w << "," << pose.data[1].w  <<"," << pose.data[2].w << "    ";
		float xt = pose.data[0].w - _initPose.x;
		float yt = pose.data[1].w - _initPose.y;
		float zt = pose.data[2].w - _initPose.z;
		return (make_float3(xt, yt, zt));
	}

	bool preprocessing(const ushort *inputDepth, const uint2 inputSize);
	bool preprocessing(const std::vector<uint16_t> inputDepth, const uint2 inputSize);
	
	void computeFrame(const ushort *inputDepth, const uint2 inputSize,
			float4 k, uint integration_rate, uint tracking_rate,
			float icp_threshold, float mu, const uint frame);

	// Unused
	/*void computeFrame(const std::vector<ushort> inputDepth, const uint2 inputSize,
		float4 k, uint integration_rate, uint tracking_rate,
		float icp_threshold, float mu, const uint frame);*/

	bool tracking(float4 k, float icp_threshold, uint tracking_rate, uint frame);
	bool raycasting(float4 k, float mu, uint frame);
	bool integration(float4 k, uint integration_rate, float mu, uint frame);

	void dumpVolume(const char* filename);

	// Two versions of each function, one to match the API of existing implementations,
	// another for using STL containers
	void renderDepth(uchar4 *out, uint2 outputSize);
	void renderDepth(std::vector<uchar4> out, uint2 outputSize);

	void renderTrack(uchar4 *out, const uint2 outputSize);
	void renderTrack(std::vector<uchar4> out, const uint2 outputSize);

	void renderVolume(uchar4 * out, const uint2 outputSize, int frame, int rate, float4 k, float mu);
	void renderVolume(std::vector<uchar4> out, const uint2 outputSize, int frame, int rate, float4 k, float mu);

	Matrix4 getPose() {
		return pose;
	}
	void setViewPose(Matrix4 *value = NULL) {
		if (value == NULL)
			viewPose = &pose;
		else
			viewPose = value;
	}
	Matrix4 *getViewPose() {
		return (viewPose);
	}
	float3 getModelDimensions() {
		return (volumeDimensions);
	}
	uint3 getModelResolution() {
		return (volumeResolution);
	}
	uint2 getComputationResolution() {
		return (computationSize);
	}

};

void synchroniseDevices(); // Synchronise CPU and GPU

#endif
