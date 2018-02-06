#include <kernels_stl.h>
#include <cstdlib>
#include <commons.h>

#include <experimental/algorithm>
#include <sycl/execution_policy>

// Unnamed/annonymous namespace
// Internal linkage - linker guarantees that the variables inside the namespace
// are not leaked to other translation units
namespace {
    sycl::sycl_execution_policy<class render_depth> render_depth_par{};
    sycl::sycl_execution_policy<class render_track> render_track_par{};
}

void renderDepthKernel(std::vector<uchar4> out, std::vector<float> depth, uint2 depthSize, const float nearPlane, const float farPlane) {
    float rangeScale = 1 / (farPlane - nearPlane);

    for (uint y=0; y<depthSize.y; y++) {
        int offset = y*depthSize.x;
        auto input_start = depth.begin() + offset;
        auto input_end = depth.begin() + offset + depthSize.x;
        auto output_start = out.begin() + offset;

        std::experimental::parallel::transform(render_depth_par, input_start, input_end, output_start, [=](float depthValue) {
        //std::transform(input_start, input_end, output_start, [=](float depthValue) {
            if (depthValue < nearPlane) { return make_uchar4(255, 255, 255, 0); }
            if (depthValue > farPlane)  { return make_uchar4(0, 0, 0, 0); }

            const float d = (depthValue - nearPlane) * rangeScale;
            return gs2rgb(d);
        });
    }
    
    // for (uint y=0; y<depthSize.y; y++) {
    //     for (uint x=0; x<depthSize.x; x++) {

    //         uint pos = x + y*depthSize.x;

    //         if (depth[pos] < nearPlane) {
    //             // The forth value is a padding in order to align memory
    //             out[pos] = make_uchar4(255, 255, 255, 0);
    //         } else {
    //             if (depth[pos] > farPlane) {
    //                 out[pos] = make_uchar4(0, 0, 0, 0);
    //             } else {
    //                 const float d = (depth[pos] - nearPlane) * rangeScale;
    //                 out[pos] = gs2rgb(d);
    //             }
    //         }
    //     };
    // };
}

void renderTrackKernel(std::vector<uchar4> out, const std::vector<TrackData> data, uint2 outSize) {

    for (uint y=0; y<outSize.y; y++) {
        int offset = y*outSize.x;
        auto input_start = data.begin() + offset;
        auto input_end = data.begin() + offset + outSize.x;
        auto output_start = out.begin() + offset;

        std::experimental::parallel::transform(render_track_par, input_start, input_end, output_start, [](TrackData td) {
        //std::transform(input_start, input_end, output_start, [](TrackData td) {
            uint r = td.result;
            if (r== 1) return make_uchar4(128, 128, 128, 0);
            if (r==-1) return make_uchar4(0, 0, 0, 0);
            if (r==-2) return make_uchar4(255, 0, 0, 0);
            if (r==-3) return make_uchar4(0, 255, 0, 0);
            if (r==-4) return make_uchar4(0, 0, 255, 0);
            if (r==-5) return make_uchar4(255, 255, 0, 0);
            return make_uchar4(255, 128, 128, 0);
        });
    };
}

// Exactly the same as a raycast
void renderVolumeKernel(std::vector<uchar4> out, const uint2 depthSize, const Volume volume,
        const Matrix4 view, const float nearPlane, const float farPlane,
        const float step, const float largestep, const float3 light,
        const float3 ambient) {

    for (uint y=0; y<depthSize.y; y++) {
        /*int offset = y*depthSize.x;
        auto input_start = out.begin() + offset;
        auto input_end = out.begin() + offset + depthSize.x;
        auto output_start = out.begin() + offset;*/

        //
        // Cannot std::transform due to the need for x?
        //
        //std::experimental::parallel::transform(render_track_par, input_start, input_end, output_start, [](TrackData td) {
        /*std::transform(input_start, input_end, output_start, [=](uchar4 unused) {

        }*/

        for (uint x=0; x<depthSize.x; x++) {
            const uint pos = x + y*depthSize.x;

            float4 hit = raycast(volume, make_uint2(x, y), view, nearPlane, farPlane, step, largestep);
            if (hit.w > 0) {
                const float3 test = make_float3(hit);
                const float3 surfNorm = volume.grad(test);
                if (length(surfNorm) > 0) {
                    const float3 diff = normalize(light - test);
                    const float dir = fmaxf(dot(normalize(surfNorm), diff), 0.f);
                    const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
                    out[pos] = make_uchar4(col.x, col.y, col.z, 0);
                } else {
                    out[pos] = make_uchar4(0, 0, 0, 0);
                }
            } else {
                out[pos] = make_uchar4(0, 0, 0, 0);
            }
        }
    }
}