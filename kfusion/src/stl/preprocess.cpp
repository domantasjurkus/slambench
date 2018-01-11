#include <kernels_stl.h>

#include <range/v3/algorithm/for_each.hpp>
#include <range/v3/view.hpp>

void mm2metersKernel(std::vector<float> &out, uint2 outSize, const ushort * in, uint2 inSize) {
    if ((inSize.x < outSize.x) || (inSize.y < outSize.y))           { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x / outSize.x != inSize.y / outSize.y))             { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    
    int ratio = inSize.x / outSize.x;

    // Subsampilng in parallel would result in inconsistent results as
    // there would be data races when computing the x and y values
    // Gather
    // std::generate(out.begin(), out.end(), [x=0,y=0,in,inSize,outSize,ratio]() mutable {
    //     float ret = in[x*ratio + inSize.x*y*ratio] / 1000.0f;
	// 	x++;
	// 	if (x == outSize.x) {
	// 		x = 0;
	// 		y++;
	// 	}
	// 	return ret;
    // });
    std::vector<int> rows = iota(outSize.y);

    //for (y=0; y<outSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](uint y) {   
		for (uint x=0; x<outSize.x; x++) {
			out[x + outSize.x*y] = in[x*ratio + inSize.x*y*ratio] / 1000.0f;
		}
    });

    // Original
    // for (y = 0; y < outSize.y; y++) {
	// 	for (unsigned int x = 0; x < outSize.x; x++) {
	// 		out[x + outSize.x * y] = in[x * ratio + inSize.x * y * ratio] / 1000.0f;
	// 	}
    // }
}

void bilateralFilterKernel(std::vector<float> &out, const std::vector<float> in, uint2 size, const float* gaussian, float e_d, int r) {
    float e_d_squared_2 = e_d * e_d * 2;

    std::vector<int> rows = iota(size.y);
    std::vector<int> cols = iota(size.x);
    
    // Stencil
    //for (auto y=0; y<size.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](uint y) {
    //ranges::for_each(rows, [&](uint y) {

        //for (int x=0; x<size.x; x++) {
        std::for_each(cols.begin(), cols.end(), [&](uint x) {
            uint pos = x + y*size.x;
            if (in[pos] == 0) {
                out[pos] = 0;
                return;
            }

            const float center = in[pos];
            float t = 0.0f;
            float sum = 0.0f;

            std::vector<int2> pairs = generate_int_pairs(-r,r,-r,r);

            //This will likely lead to race conditions - use atomic<float>?
            std::for_each(pairs.begin(), pairs.end(), [&](int2 p) {
                uint2 curPos = make_uint2(clamp(x+p.x, 0u, size.x-1),
                                          clamp(y+p.y, 0u, size.y-1));
                const float curPix = in[curPos.x + curPos.y*size.x];
                if (curPix > 0) {
                    const float mod = sq(curPix - center);
                    const float factor = gaussian[p.x+r]*gaussian[p.y+r]*expf(-mod / e_d_squared_2);
                    t += factor * curPix;
                    sum += factor;
                }
            });

            // Really bad idea: reduce to sum and t separately
            // Definitely slower when sequential - how about par?
            // t = std::accumulate(pairs.begin(), pairs.end(), 0.0f, [&](float acc, int2 p) {
            //     uint2 curPos = make_uint2(clamp(x+p.x, 0u, size.x-1),
            //                               clamp(y+p.y, 0u, size.y-1));
            //     const float curPix = in[curPos.x + curPos.y*size.x];
            //     if (curPix > 0) {
            //         const float mod = sq(curPix - center);
            //         const float factor = gaussian[p.x+r]*gaussian[p.y+r]*expf(-mod / e_d_squared_2);
            //         return acc + factor * curPix;
            //     }
            //     return acc;
            // });
            // sum = std::accumulate(pairs.begin(), pairs.end(), 0.0f, [&](float acc, int2 p) {
            //     uint2 curPos = make_uint2(clamp(x+p.x, 0u, size.x-1),
            //                               clamp(y+p.y, 0u, size.y-1));
            //     const float curPix = in[curPos.x + curPos.y*size.x];
            //     if (curPix > 0) {
            //         const float mod = sq(curPix - center);
            //         const float factor = gaussian[p.x+r]*gaussian[p.y+r]*expf(-mod / e_d_squared_2);
            //         return acc + factor;
            //     }
            //     return acc;
            // });
            out[pos] = t / sum;
            
            // Original
            //
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
        });
    });
}
