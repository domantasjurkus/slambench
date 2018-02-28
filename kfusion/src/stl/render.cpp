#include <kernels_stl.h>
#include <cstdlib>
#include <commons.h>

//#include <experimental/algorithm>
//#include <sycl/execution_policy>-
//#include "pstl/algorithm"

/*namespace {
    sycl::sycl_execution_policy<class render_depth> render_depth_par{};
    sycl::sycl_execution_policy<class render_track> render_track_par{};
}*/

void renderDepthKernel(std::vector<uchar4> out, std::vector<float> depth, uint2 depthSize, const float nearPlane, const float farPlane) {
    float rangeScale = 1 / (farPlane - nearPlane);

    //std::experimental::parallel::transform(render_depth_par, depth.begin(), depth.end(), out.begin(), [=](float depthValue) {
    //std::experimental::parallel::transform(std::execution::par, depth.begin(), depth.end(), out.begin(), [=](float depthValue) {
    std::transform(depth.begin(), depth.end(), out.begin(), [=](float depthValue) {
        if (depthValue < nearPlane) { return make_uchar4(255, 255, 255, 0); }
        if (depthValue > farPlane)  { return make_uchar4(0, 0, 0, 0); }

        const float d = (depthValue - nearPlane) * rangeScale;
        return gs2rgb(d);
    });
}

void renderTrackKernel(std::vector<uchar4> out, const std::vector<TrackData> data, uint2 outSize) {

    //std::experimental::parallel::transform(render_track_par, data.begin(), data.end(), out.begin(), [=](TrackData td) {
    std::transform(data.begin(), data.end(), out.begin(), [=](TrackData td) {
        uint r = td.result;
        if (r== 1) return make_uchar4(128, 128, 128, 0);
        if (r==-1) return make_uchar4(0, 0, 0, 0);
        if (r==-2) return make_uchar4(255, 0, 0, 0);
        if (r==-3) return make_uchar4(0, 255, 0, 0);
        if (r==-4) return make_uchar4(0, 0, 255, 0);
        if (r==-5) return make_uchar4(255, 255, 0, 0);
        return make_uchar4(255, 128, 128, 0);
    });
}

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
        const float3 ambient) {

    const float3 origin = get_translation(view);

    // lambda capture with pointer fields
    std::transform(pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
        uint x = pos % depthSize.x;
        uint y = pos / depthSize.x;

        float4 hit = raycast(volume, make_uint2(x, y), view, origin, nearPlane, farPlane, step, largestep);
        if (hit.w > 0) {
            const float3 test = make_float3(hit);
            const float3 surfNorm = volume.grad(test);
            if (length(surfNorm) > 0) {
                const float3 diff = normalize(light - test);
                const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
                const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
                return make_uchar4(col.x, col.y, col.z, 0);
            }
            return make_uchar4(0, 0, 0, 0);
        }
        return make_uchar4(0, 0, 0, 0);
    });
}