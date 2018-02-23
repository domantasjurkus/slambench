#include <kernels_stl.h>

// Voxel grid (Volume) to point cloud
// Takes volume, produces point cloud
// inputSize should be called outputSize
void raycastKernel(std::vector<std::pair<float3, float3>> &vertex_normals,
        uint2 inputSize,
        const Volume integration,
        const Matrix4 view,
        const float nearPlane,
        const float farPlane,
        const float step,
        const float largestep) {

    std::vector<uint> pixels(inputSize.x*inputSize.y);
    std::iota(pixels.begin(), pixels.end(), 0);

    std::transform(pixels.begin(), pixels.end(), vertex_normals.begin(), [=](uint pix) {
        uint x = pix % inputSize.x;
        uint y = pix / inputSize.x ;
        uint2 pos = make_uint2(x, y);

        std::pair<float3, float3> new_pair(make_float3(0), make_float3(KFUSION_INVALID,0,0));

        const float3 origin = get_translation(view);
        const float4 hit = raycast(integration, pos, view, origin, nearPlane, farPlane, step, largestep);

        if (hit.w > 0.0) {
            new_pair.first = make_float3(hit);
            float3 surfNorm = integration.grad(make_float3(hit));
            if (length(surfNorm) != 0) {
                new_pair.second = normalize(surfNorm);
            }
        }

        return new_pair;
    });

    // for (uint y=0; y<inputSize.y; y++) {
    //     for (uint x=0; x<inputSize.x; x++) {
    //         uint2 pos = make_uint2(x, y);

    //         // view = used for camera transformation
    //         const float4 hit = raycast(integration, pos, view, nearPlane, farPlane, step, largestep);
    //         if (hit.w > 0.0) {
    //             vertex_normals.first[pos.x + pos.y*inputSize.x] = make_float3(hit);
    //             float3 surfNorm = integration.grad(make_float3(hit));
    //             if (length(surfNorm) == 0) {
    //                 vertex_normals.second[pos.x + pos.y*inputSize.x].x = KFUSION_INVALID;
    //             } else {
    //                 vertex_normals.second[pos.x + pos.y*inputSize.x] = normalize(surfNorm);
    //             }
    //         } else {
    //             // Raycast miss
    //             vertex_normals.first[pos.x + pos.y*inputSize.x] = make_float3(0);
    //             vertex_normals.second[pos.x + pos.y*inputSize.x] = make_float3(KFUSION_INVALID, 0, 0);
    //         }
    //     };
    // };
}