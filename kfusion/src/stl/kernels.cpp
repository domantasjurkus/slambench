#include <kernels_stl.h>

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
float * gaussian;

// inter-frame
Volume volume;
float3 * vertex;
float3 * normal;

// intra-frame
TrackData * trackingResult;
float* reductionoutput;
Matrix4 oldPose;
Matrix4 raycastPose;
float3 ** inputVertex;
float3 ** inputNormal;

// Change from arrays to vectors
//float * floatDepth;
//float ** ScaledDepth;
std::vector<float> floatDepthVector;
std::vector<std::vector<float>> scaledDepthVector;
	
void Kfusion::languageSpecificConstructor() {
	// internal buffers to initialize
	reductionoutput = (float*) calloc(sizeof(float) * 8 * 32, 1);

	// Added by Dom
	scaledDepthVector.resize(sizeof(float*) * iterations.size());

	//ScaledDepth = (float**)  calloc(sizeof(float*)  * iterations.size(), 1);
	inputVertex = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);
	inputNormal = (float3**) calloc(sizeof(float3*) * iterations.size(), 1);

	for (unsigned int i=0; i<iterations.size(); ++i) {
		//ScaledDepth[i] = (float*) calloc(sizeof(float) * (computationSize.x * computationSize.y) / (int) pow(2, i), 1);
		inputVertex[i] = (float3*) calloc(sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), 1);
		inputNormal[i] = (float3*) calloc(sizeof(float3) * (computationSize.x * computationSize.y) / (int) pow(2, i), 1);

		// Added by Dom
		scaledDepthVector[i].resize((computationSize.x * computationSize.y) / (int) pow(2, i));
	}

	// Added by Dom
	floatDepthVector.resize(computationSize.x * computationSize.y);
	//floatDepth = (float*) calloc(sizeof(float) * computationSize.x * computationSize.y, 1);

	vertex = (float3*) calloc(sizeof(float3) * computationSize.x * computationSize.y, 1);
	normal = (float3*) calloc(sizeof(float3) * computationSize.x * computationSize.y, 1);
	trackingResult = (TrackData*) calloc(sizeof(TrackData) * computationSize.x * computationSize.y, 1);

	// ********* BEGIN : Generate the gaussian *************
	size_t gaussianS = radius * 2 + 1;
	gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
	int x;
	for (unsigned int i = 0; i < gaussianS; i++) {
		x = i - 2;
		gaussian[i] = expf(-(x * x) / (2 * delta * delta));
	}
	// ********* END : Generate the gaussian *************

	volume.init(volumeResolution, volumeDimensions);
	reset();
}

Kfusion::~Kfusion() {
	//free(floatDepth);
	free(trackingResult);
	free(reductionoutput);
	for (unsigned int i=0; i<iterations.size(); ++i) {
		//free(ScaledDepth[i]);
		free(inputVertex[i]);
		free(inputNormal[i]);
	}
	//free(ScaledDepth);
	free(inputVertex);
	free(inputNormal);

	free(vertex);
	free(normal);
	free(gaussian);

	// Vectors?

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
				//std::cout <<  x << " " << y << " " << z <<"\n";
				volume.setints(x, y, z, make_float2(1.0f, 0.0f));
			}
        }
    }
}

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold) {
	bool res = false;
	//TICK();
	// Update the pose regarding the tracking result
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	pose = toMatrix4(delta) * pose;

	// Return validity test result of the tracking
	if (norm(x) < icp_threshold)
		res = true;

	//TOCK("updatePoseKernel", 1);
	return res;
}

// Check the tracking result, and go back to the previous camera position if necessary
bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output, uint2 imageSize, float track_threshold) {
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2)
			|| (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold)) {
		pose = oldPose;
		return false;
    }
	return true;
}

bool Kfusion::preprocessing(const ushort * inputDepth, const uint2 inputSize) {
	//mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
	mm2metersKernel(floatDepthVector, computationSize, inputDepth, inputSize);
	//bilateralFilterKernel(ScaledDepth[0], floatDepth, computationSize, gaussian, e_delta, radius);
	bilateralFilterKernel(scaledDepthVector[0], floatDepthVector, computationSize, gaussian, e_delta, radius);
	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold, uint tracking_rate, uint frame) {
	if (frame % tracking_rate != 0)
		return false;

	// half sample the input depth maps into the pyramid levels
	for (unsigned int i=1; i<iterations.size(); ++i) {
		halfSampleRobustImageKernel(scaledDepthVector[i], scaledDepthVector[i-1],
				make_uint2(computationSize.x / (int) pow(2, i-1),
						   computationSize.y / (int) pow(2, i-1)), e_delta*3, 1);
	}

	// prepare the 3D information from the input depth maps
	uint2 localimagesize = computationSize;

	//printf("arbitrary scaledDepth: %f\n", scaledDepthVector[0][123]);

	for (unsigned int i=0; i<iterations.size(); ++i) {
		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
		depth2vertexKernel(inputVertex[i], scaledDepthVector[i], localimagesize, invK);
		vertex2normalKernel(inputNormal[i], inputVertex[i], localimagesize);
		localimagesize = make_uint2(localimagesize.x/2, localimagesize.y/2);
	}

	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

	// Debugging - these are as expected
	/*for (int h=0; h<10; h++) {
		printf("i, n: %f %f\n", inputVertex[0][h].y, inputNormal[0][h].y);
	}*/

    // ICP
	// Start from smallest image, then bigger, then biggest
	//printf("Iterations: %d %d %d\n", iterations[0], iterations[1], iterations[2]);
	for (int level=iterations.size()-1; level>=0; --level) {
		uint2 localimagesize = make_uint2(computationSize.x / (int) pow(2, level),
										  computationSize.y / (int) pow(2, level));
		// For each pyramid level
		//printf("iterations[%d] = %d, %f\n", level, iterations[level], icp_threshold);
		for (int i=0; i<iterations[level]; ++i) {

			// both point clouds
			// compute the error
			double a = tock();
			//printf("%f %f\n", inputVertex[0][i], inputNormal[0][i]);
			trackKernel(trackingResult, inputVertex[level], inputNormal[level],
						localimagesize, vertex, normal, computationSize, pose,
						projectReference, dist_threshold, normal_threshold);
			double b = tock();
			//printf("tracking step: %f\n", b-a);
					
			// sum of the errors
			reduceKernel(reductionoutput, trackingResult, computationSize, localimagesize);
			
			// correct for the errors
			if (updatePoseKernel(pose, reductionoutput, icp_threshold)) {
				//printf("breaking level %d at i=%d\n", level, i);
				break;
			}
		}
	}
	return checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);
}

bool Kfusion::raycasting(float4 k, float mu, uint frame) {
	bool doRaycast = false;

	if (frame > 2) {
		raycastPose = pose;
		raycastKernel(vertex, normal, computationSize, volume,
				raycastPose * getInverseCameraMatrix(k), nearPlane, farPlane,
				step, 0.75f * mu);
	}

	return doRaycast;
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame) {
	bool doIntegrate = checkPoseKernel(pose, oldPose, reductionoutput, computationSize, track_threshold);

	if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
		// Commented out due to change from floatDepth to floatDepthVector
		integrateKernel(volume, floatDepthVector.data(), computationSize, inverse(pose), getCameraMatrix(k), mu, maxweight);
		doIntegrate = true;
	} else {
		doIntegrate = false;
	}

	return doIntegrate;
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

void Kfusion::renderVolume(uchar4 * out, uint2 outputSize, int frame,
		int raycast_rendering_rate, float4 k, float largestep) {
	if (frame % raycast_rendering_rate == 0)
		renderVolumeKernel(out, outputSize, volume,
				*(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
				farPlane * 2.0f, step, largestep, light, ambient);
}

void Kfusion::renderTrack(uchar4 * out, uint2 outputSize) {
	renderTrackKernel(out, trackingResult, outputSize);
}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize) {
	renderDepthKernel(out, floatDepthVector.data(), outputSize, nearPlane, farPlane);
}

void Kfusion::computeFrame(const ushort * inputDepth, const uint2 inputSize,
			 float4 k, uint integration_rate, uint tracking_rate,
			 float icp_threshold, float mu, const uint frame) {
    preprocessing(inputDepth, inputSize);
    _tracked = tracking(k, icp_threshold, tracking_rate, frame);
    _integrated = integration(k, integration_rate, mu, frame);
    raycasting(k, mu, frame);
}


void synchroniseDevices() {
	// Nothing to do in the C++ implementation
}
