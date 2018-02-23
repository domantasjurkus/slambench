#include <kernels_stl.h>
#include <kfusion_class.h>

#include <sycl/execution_policy>
#include <range/v3/all.hpp>

inline double tock() {
	synchroniseDevices();
#ifdef __APPLE__
		clock_serv_t cclock;
		mach_timespec_t clockData;
		host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
		clock_get_time(cclock, &clockData);
		mach_port_deallocate(mach_task_self(), cclock);
#else
		struct timespec clockData;
		clock_gettime(CLOCK_MONOTONIC, &clockData);
#endif
		return (double) clockData.tv_sec + clockData.tv_nsec / 1000000000.0;
}

// input once
std::vector<float> gaussian;

// inter-frame
Volume volume;
//std::vector<float3> vertex;
//std::vector<float3> normal;
std::vector<std::pair<float3, float3>> vertex_normals;

// intra-frame
//std::vector<TrackData> trackingResult;
std::vector<std::vector<TrackData>> trackingResult;
Matrix4 oldPose;
Matrix4 raycastPose;
std::vector<float> reductionoutput;

std::vector<float> floatDepthVector;
std::vector<std::vector<float>> scaledDepthVector;

std::vector<std::vector<float3>> inputVertex;
std::vector<std::vector<float3>> inputNormal;

std::vector<uint> pixels;

void Kfusion::languageSpecificConstructor() {
	// internal buffers to initialize
	reductionoutput.resize(8*32);
	scaledDepthVector.resize(iterations.size());

	inputVertex.resize(iterations.size());
	inputNormal.resize(iterations.size());
	trackingResult.resize(iterations.size());

	for (auto i=0; i<iterations.size(); ++i) {
		scaledDepthVector[i].resize((computationSize.x * computationSize.y) / (int) pow(2, i));
		inputVertex[i].resize((computationSize.x * computationSize.y) / (int) pow(2, i));
		inputNormal[i].resize((computationSize.x * computationSize.y) / (int) pow(2, i));

		trackingResult[i].resize((computationSize.x * computationSize.y) / (int) pow(2, i));
	}

	floatDepthVector.resize(computationSize.x * computationSize.y);
	vertex_normals.resize(computationSize.x * computationSize.y);
	
	pixels.resize(computationSize.x * computationSize.y);
	std::iota(pixels.begin(), pixels.end(), 0);

	// ********* BEGIN : Generate the gaussian *************
	gaussian.resize(radius*2 + 1);
	auto g = ranges::view::ints(0, radius*2+1);

	ranges::transform(g, gaussian.begin(), [](float i) {
		return expf(-((i-2) * (i-2)) / (2 * delta * delta));
	});

	// This is another way of generating a gaussian, but it looks more complicated than it has to be
	// std::generate(gaussian.begin(), gaussian.end(), [i=0] () mutable {
	// 	int x = i-2;
	// 	i++;
	// 	return expf(-(x*x) / (2*delta*delta));
	// });

	// Original
	/*for (auto i=0; i<gaussianS; i++) {
		gaussian[i] = expf(-((i-2) * (i-2)) / (2 * delta * delta));
	}*/
	// ********* END : Generate the gaussian *************

	volume.init(volumeResolution, volumeDimensions);
	reset();
}

Kfusion::~Kfusion() {

	volume.release();
}
void Kfusion::reset() {
	initVolumeKernel(volume);
}

void init() {};

void clean() {};

void initVolumeKernel(Volume volume) {
	for (unsigned int x = 0; x < volume.size.x; x++) {
		for (unsigned int y = 0; y < volume.size.y; y++) {
			for (unsigned int z = 0; z < volume.size.z; z++) {
				volume.setints(x, y, z, make_float2(1.0f, 0.0f));
			}
        }
    }
}

bool updatePoseKernel(Matrix4 & pose, const std::vector<float> output, float icp_threshold) {
	bool res = false;
	// Update the pose regarding the tracking result
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output.data());
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	pose = toMatrix4(delta) * pose;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold)
		res = true;

	return res;
}

// Check the tracking result, and go back to the previous camera position if necessary
bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const std::vector<float> output, uint2 imageSize, float track_threshold) {
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output.data());

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
			|| (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold)) {
		pose = oldPose;
		return false;
    }
	return true;
}

bool Kfusion::preprocessing(const std::vector<uint16_t> inputDepth, const uint2 inputSize) {
	mm2metersKernel(floatDepthVector, computationSize, inputDepth, inputSize);
	bilateralFilterKernel(scaledDepthVector[0], floatDepthVector, pixels, computationSize, gaussian, e_delta, radius);
	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate, uint frame) {
	if (frame % tracking_rate != 0)
		return false;

	// half sample the input depth maps into the pyramid levels
	for (uint i=1; i<iterations.size(); ++i) {
		halfSampleRobustImageKernel(scaledDepthVector[i], scaledDepthVector[i-1],
				make_uint2(computationSize.x / (int) pow(2, i-1),
						   computationSize.y / (int) pow(2, i-1)), e_delta*3, 1);
	}

	// prepare the 3D information from the input depth maps
	uint2 localimagesize = computationSize;

	for (uint i=0; i<iterations.size(); ++i) {
		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
		depth2vertexKernel(inputVertex[i], scaledDepthVector[i], pixels, localimagesize, invK);
		vertex2normalKernel(inputNormal[i], inputVertex[i], pixels, localimagesize);
		localimagesize = make_uint2(localimagesize.x/2, localimagesize.y/2);
	}

	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

	// levels = 2,1,0
	for (int level=iterations.size()-1; level>=0; --level) {
		uint2 localimagesize = make_uint2(computationSize.x / (int) pow(2, level),
										  computationSize.y / (int) pow(2, level));

		for (int i=0; i<iterations[level]; ++i) {

			double a = tock();
			trackKernel(trackingResult[level],
					inputVertex[level],
					inputNormal[level],
					localimagesize,
					vertex_normals,
					computationSize,
					pose,
					projectReference,
					dist_threshold,
					normal_threshold);
			double b = tock();
					
			// sum of the errors
			reduceKernel(reductionoutput, trackingResult[level], computationSize, localimagesize);
			
			// correct for the errors
			if (updatePoseKernel(pose, reductionoutput, icp_threshold)) {
				break;
			}
		}
	}
	return checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame) {
	bool doIntegrate = checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);
	
	if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
		integrateKernel(volume, floatDepthVector, computationSize, inverse(pose), getCameraMatrix(k), mu, maxweight);
		doIntegrate = true;
	} else {
		doIntegrate = false;
	}
	return doIntegrate;
}

bool Kfusion::raycasting(float4 k, float mu, uint frame) {
	bool doRaycast = false;

	if (frame > 2) {
		raycastPose = pose;
		raycastKernel(vertex_normals,
				computationSize,
				volume,
				pixels,
				raycastPose * getInverseCameraMatrix(k),
				nearPlane,
				farPlane,
				step,
				0.75f*mu);
	}

	return doRaycast;
}

void Kfusion::dumpVolume(const char *filename) {
	std::ofstream fDumpFile;

	if (filename == NULL) {
		return;
	}

	std::cout << "Dumping the volumetric representation on file: " << filename
			<< std::endl;
	fDumpFile.open(filename, std::ios::out | std::ios::binary);
	if (fDumpFile.fail()) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}

	// Dump on file without the y component of the short2 variable
	for (unsigned int i = 0; i < volume.size.x * volume.size.y * volume.size.z; i++) {
		fDumpFile.write((char *) (volume.data + i), sizeof(short));
	}

	fDumpFile.close();
}

void Kfusion::renderDepth(std::vector<uchar4> out, uint2 outputSize) {
	renderDepthKernel(out, floatDepthVector, outputSize, nearPlane, farPlane);
}

void Kfusion::renderTrack(std::vector<uchar4> out, uint2 outputSize) {
	renderTrackKernel(out, trackingResult[0], outputSize);
}

void Kfusion::renderVolume(std::vector<uchar4> out, uint2 outputSize, int frame, int raycast_rendering_rate, float4 k, float largestep) {
	if (frame % raycast_rendering_rate == 0) {
		renderVolumeKernel(out, outputSize, volume, pixels,
				*(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
				farPlane * 2.0f, step, largestep, light, ambient);
	}
}

// Unused
/*void Kfusion::computeFrame(const std::vector<ushort> inputDepth, const uint2 inputSize,
			float4 k, uint integration_rate, uint tracking_rate,
			float icp_threshold, float mu, const uint frame) {
    preprocessing(inputDepth, inputSize);
    _tracked = tracking(k, icp_threshold, tracking_rate, frame);
    _integrated = integration(k, integration_rate, mu, frame);
    raycasting(k, mu, frame);
}*/


void synchroniseDevices() {
	// Nothing to do in the C++ implementation
}
