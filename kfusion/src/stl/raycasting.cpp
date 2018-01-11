#include <kernels_stl.h>

// Voxel grid (Volume) to point cloud
// Takes volume, produces point cloud
// Volume does not include normals
// inputSize should be called outputSize
void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
    const Volume integration, const Matrix4 view, const float nearPlane,
    const float farPlane, const float step, const float largestep) {

    std::vector<int> rows = iota(inputSize.y);

    //for (uint y=0; y<inputSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](int y) {
        for (uint x=0; x<inputSize.x; x++) {

            uint2 pos = make_uint2(x, y);

            // view = used for camera transformation
            const float4 hit = raycast(integration, pos, view, nearPlane, farPlane, step, largestep);
            if (hit.w > 0.0) {
                vertex[pos.x + pos.y * inputSize.x] = make_float3(hit);
                float3 surfNorm = integration.grad(make_float3(hit));
                if (length(surfNorm) == 0) {
                    //normal[pos] = normalize(surfNorm); // APN added
                    normal[pos.x + pos.y * inputSize.x].x = KFUSION_INVALID;
                } else {
                    normal[pos.x + pos.y * inputSize.x] = normalize(surfNorm);
                }
            } else {
                //std::cerr<< "RAYCAST MISS "<<  pos.x << " " << pos.y <<"  " << hit.w <<"\n";
                vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
                normal[pos.x + pos.y * inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
            }
        }
    });
}