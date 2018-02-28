#include <kernels_stl.h>

// inputSize should be called outputSize
void raycastKernel(std::vector<std::pair<float3, float3>> &vertex_normals,
        uint2 inputSize,
        const Volume integration,
        const std::vector<uint> pixels,
        const Matrix4 view,
        const float nearPlane,
        const float farPlane,
        const float step,
        const float largestep) {

    // lambda capture with pointer fields
    //std::experimental::parallel::transform(par, pixels.begin(), pixels.end(), vertex_normals.begin(), [=](uint pix) {
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
}