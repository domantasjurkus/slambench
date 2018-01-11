#include <kernels_stl.h>

void renderDepthKernel(uchar4* out, float* depth, uint2 depthSize, const float nearPlane, const float farPlane) {
    float rangeScale = 1 / (farPlane - nearPlane);

    std::vector<int> rows = iota(depthSize.y);

    // Map
    //for (uint y=0; y<depthSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](int y) {
        int rowOffeset = y * depthSize.x;
        std::vector<int> cols = iota(depthSize.x, rowOffeset);

        for (uint x=0; x<depthSize.x; x++) {
        // Memory map error
        //std::for_each(cols.begin(), cols.end(), [&](int x) {
            uint pos = rowOffeset + x;

            if (depth[pos] < nearPlane) {
                // The forth value is a padding in order to align memory
                out[pos] = make_uchar4(255, 255, 255, 0);
            } else {
                if (depth[pos] > farPlane) {
                    out[pos] = make_uchar4(0, 0, 0, 0);
                } else {
                    const float d = (depth[pos] - nearPlane) * rangeScale;
                    out[pos] = gs2rgb(d);
                }
            }
        //});
        };
    });
}

void renderTrackKernel(uchar4* out, const std::vector<TrackData> data, uint2 outSize) {
    std::vector<int> rows = iota(outSize.y);

    // Map
    //for (uint y=0; y<outSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](int y) {
        int rowOffeset = y * outSize.x;
        //std::vector<int> cols = iota(outSize.x, rowOffeset);

        for (uint x=0; x<outSize.x; x++) {
        // Segfault
        //std::for_each(cols.begin(), cols.end(), [&](int x) {
            uint pos = rowOffeset + x;

            switch (data[pos].result) {
            case 1:
                out[pos] = make_uchar4(128, 128, 128, 0);  // ok	 GREY
                break;
            case -1:
                out[pos] = make_uchar4(0, 0, 0, 0);      // no input BLACK
                break;
            case -2:
                out[pos] = make_uchar4(255, 0, 0, 0);        // not in image RED
                break;
            case -3:
                out[pos] = make_uchar4(0, 255, 0, 0);    // no correspondence GREEN
                break;
            case -4:
                out[pos] = make_uchar4(0, 0, 255, 0);        // to far away BLUE
                break;
            case -5:
                out[pos] = make_uchar4(255, 255, 0, 0);     // wrong normal YELLOW
                break;
            default:
                out[pos] = make_uchar4(255, 128, 128, 0);
                break;
            }
        //});
        }
    });
}

// Exactly the same as a raycast
// Output is a color
void renderVolumeKernel(uchar4* out, const uint2 depthSize, const Volume volume,
        const Matrix4 view, const float nearPlane, const float farPlane,
        const float step, const float largestep, const float3 light,
        const float3 ambient) {

    std::vector<int> rows = iota(depthSize.y);

    //for (uint y=0; y<depthSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](int y) {
        int rowOffeset = y * depthSize.x;
        std::vector<int> cols = iota(depthSize.x, rowOffeset);

        for (uint x=0; x<depthSize.x; x++) {
        // Memory map
        //std::for_each(cols.begin(), cols.end(), [&](int x) {
            const uint pos = x + rowOffeset;

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
        };
    });
}