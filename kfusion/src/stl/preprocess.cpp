#include <commons.h>

void mm2metersKernel(std::vector<float> &out, uint2 outSize, const ushort * in, uint2 inSize) {
    if ((inSize.x < outSize.x) || (inSize.y < outSize.y))           { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x / outSize.x != inSize.y / outSize.y))             { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    
    int ratio = inSize.x / outSize.x;

    // Subsampilng in parallel would result in inconsistent results as
    // there would be data races fow computing the x and y values
    std::generate(out.begin(), out.end(), [x=0,y=0,in,inSize,outSize,ratio]() mutable {
        float ret = in[x*ratio + inSize.x*y*ratio] / 1000.0f;
		x++;
		if (x == outSize.x) {
			x = 0;
			y++;
		}
		return ret;
    });

    // for (y = 0; y < outSize.y; y++) {
	// 	for (unsigned int x = 0; x < outSize.x; x++) {
	// 		out[x + outSize.x * y] = in[x * ratio + inSize.x * y * ratio] / 1000.0f;
	// 	}
    // }
}

void bilateralFilterKernel(std::vector<float> &out, const std::vector<float> in, uint2 size, const float * gaussian, float e_d, int r) {
    uint y;
    float e_d_squared_2 = e_d * e_d * 2;

    // Two options:
    // 1. Generate all (x,y) pairs, then run a single foreach loop (paralelizable)
    // 2. Paralellise only the inner loop
    for (y=0; y<size.y; y++) {
        for (uint x=0; x<size.x; x++) {
            uint pos = x + y*size.x;
            if (in[pos] == 0) {
                out[pos] = 0;
                continue;
            }

            float sum = 0.0f;
            float t = 0.0f;

            const float center = in[pos];

            // We know the r value will be small (2 by default)
            // Therefore it may be sensible to generate a vector
            // of all (i,j) pairs, then loop std::for_each
            for (int i=-r; i<=r; i++) {
                for (int j=-r; j<=r; j++) {
                    uint2 curPos = make_uint2(clamp(x+i, 0u, size.x-1),
                                              clamp(y+j, 0u, size.y-1));
                    const float curPix = in[curPos.x + curPos.y*size.x];
                    if (curPix > 0) {
                        const float mod = sq(curPix - center);
                        const float factor = gaussian[i+r]*gaussian[j+r]*expf(-mod / e_d_squared_2);
                        t += factor * curPix;
                        sum += factor;
                    }
                }
            }
            // for (int i=-r; i<=r; i++) {
            //     for (int j=-r; j<=r; j++) {
            //         uint2 curPos = make_uint2(clamp(x+i, 0u, size.x-1),
            //                                   clamp(y+j, 0u, size.y-1));
            //         const float curPix = in[curPos.x + curPos.y*size.x];
            //         if (curPix > 0) {
            //             const float mod = sq(curPix - center);
            //             const float factor = gaussian[i+r]*gaussian[j+r]*expf(-mod / e_d_squared_2);
            //             t += factor * curPix;
            //             sum += factor;
            //         }
            //     }
            // }
            out[pos] = t / sum;
        }
    }
}
