#include <kernels_stl.h>

// Given the point cloud, how to update the volume
// In-place transformation (side-effect to an argument) because we (usually) don't have enough memory
// to store two Volumes in RAM
void integrateKernel(Volume vol, const float* depth, uint2 depthSize,
		const Matrix4 invTrack, const Matrix4 K, const float mu,
		const float maxweight) {

	const float3 delta = rotate(invTrack, make_float3(0, 0, vol.dim.z / vol.size.z));
	const float3 cameraDelta = rotate(K, delta);

	std::vector<int> rows = iota(vol.size.y);
    std::vector<int> cols = iota(vol.size.x);

    // Map/Gather, from depth to volume
	//for (uint y=0; y<vol.size.y; y++) {
	std::for_each(rows.begin(), rows.end(), [&](int y) {

		//for (uint x=0; x<vol.size.x; x++) {
		std::for_each(cols.begin(), cols.end(), [&](int x) {

			uint3 pix = make_uint3(x, y, 0); //pix.x = x;pix.y = y;
			float3 pos = invTrack * vol.pos(pix);
			float3 cameraX = K * pos;

			for (pix.z=0; pix.z<vol.size.z; ++pix.z, pos+=delta, cameraX+=cameraDelta) {

				if (pos.z < 0.0001f) { // some near plane constraint
					continue;
				}
				const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f, cameraX.y / cameraX.z + 0.5f);
				if (pixel.x < 0 || pixel.x > depthSize.x - 1 || pixel.y < 0 || pixel.y > depthSize.y - 1) {
					continue;
				}
                const uint2 px = make_uint2(pixel.x, pixel.y);
				if (depth[px.x + px.y * depthSize.x] == 0) {
                    continue;
				}
				
                const float diff = (depth[px.x+px.y*depthSize.x]-cameraX.z) * std::sqrt(1+sq(pos.x/pos.z)+sq(pos.y/pos.z));
                
				if (diff > -mu) {
					const float sdf = fminf(1.f, diff / mu);
					float2 data = vol[pix];
					data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f, 1.f);
					data.y = fminf(data.y + 1, maxweight);
					vol.set(pix, data);
				}
			}
		});
    });
}