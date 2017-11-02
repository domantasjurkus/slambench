#include <kernels.h>

void mm2metersKernel(float * out, uint2 outSize, const ushort * in, uint2 inSize) {
    //TICK();
    // Check for unsupported conditions
    if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
        std::cerr << "Invalid ratio." << std::endl;
        exit(1);
    }
    if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
        std::cerr << "Invalid ratio." << std::endl;
        exit(1);
    }
    if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
        std::cerr << "Invalid ratio." << std::endl;
        exit(1);
    }
    
    int ratio = inSize.x / outSize.x;
    unsigned int y;

    // Can i STL this?
    // map
    for (y = 0; y < outSize.y; y++) {
        for (unsigned int x = 0; x < outSize.x; x++) {
            // This is a map
            // in size > out size
            // filter()
            out[x + outSize.x*y] = in[x*ratio + inSize.x*y*ratio] / 1000.0f;
        }
    }

    //TOCK("mm2metersKernel", outSize.x * outSize.y);
}

void bilateralFilterKernel(float* out, const float* in, uint2 size, const float * gaussian, float e_d, int r) {
    //TICK()
    uint y;
    float e_d_squared_2 = e_d * e_d * 2;

    // STL the for loop?
    // For each output value:
    for (y = 0; y < size.y; y++) {
        for (uint x = 0; x < size.x; x++) {
            uint pos = x + y * size.x;
            if (in[pos] == 0) {
                out[pos] = 0;
                continue;
            }

            float sum = 0.0f;
            float t = 0.0f;

            const float center = in[pos];

            // r will be 2 by default
            // reduction to t and sum
            for (int i = -r; i <= r; ++i) {
                for (int j = -r; j <= r; ++j) {
                    uint2 curPos = make_uint2(clamp(x+i, 0u, size.x-1),
                                              clamp(y+j, 0u, size.y-1));
                    const float curPix = in[curPos.x + curPos.y * size.x];
                    if (curPix > 0) {
                        const float mod = sq(curPix - center);
                        const float factor = gaussian[i+r]
                            * gaussian[j+r]
                            * expf(-mod / e_d_squared_2);
                        t += factor * curPix;
                        sum += factor;
                    }
                }
            }
            out[pos] = t / sum;
        }
    }
    //TOCK("bilateralFilterKernel", size.x * size.y);
}

