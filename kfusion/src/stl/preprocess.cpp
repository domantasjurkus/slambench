#include <kernels_stl.h>

//#include <range/v3/all.hpp>
//#include <experimental/algorithm>
//#include <sycl/execution_policy>

#include "pstl/execution"
#include "pstl/algorithm"

/*namespace {
    sycl::sycl_execution_policy<class bilateral_filter> bilateral_filter_par;
}*/

void mm2metersKernel(std::vector<float> &out,
        uint2 outSize,
        const std::vector<uint> pixels,
        const std::vector<uint16_t> in,
        uint2 inSize) {
            
    if ((inSize.x < outSize.x) || (inSize.y < outSize.y))           { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x / outSize.x != inSize.y / outSize.y))             { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    
    int ratio = inSize.x / outSize.x;

    // Gather
    // lambda capture with pointer fields
    //std::transform(pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
    std::transform(std::execution::par, pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
        uint x = pos % outSize.x;
        uint y = pos / outSize.x;
        return in[x*ratio + inSize.x*y*ratio] / 1000.0f;
    });

    /*for (uint y=0; y<outSize.y; y++) {
		for (uint x=0; x<outSize.x; x++) {
			out[x + outSize.x*y] = in[x*ratio + inSize.x*y*ratio] / 1000.0f;
		}
    };*/

    // Boost one-line wonder
    //using namespace boost::assign;
    //boost::copy(input | boost::adaptors::strided(ratio), out.begin());

    // Ranges (memory corruption?)
    //out = in | ranges::view::stride(ratio) | ranges::view::transform([](float f) {return f/1.000f;});
}

void bilateralFilterKernel(std::vector<float> &out,
        const std::vector<float> in,
        const std::vector<uint> pixels,
        uint2 size,
        const std::vector<float> gaussian,
        float e_d,
        int r) {
    float e_d_squared_2 = e_d*e_d*2;

    // lambda capture with pointer fields
    //std::transform(pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
    //std::experimental::parallel::transform(bilateral_filter_par, pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
    std::transform(std::execution::par, pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
        // uint pos = x + y*size.x;
        uint x = pos % size.x;
        uint y = pos / size.x;

        if (in[pos] == 0) {
            //out[pos] = 0;
            return 0.0f;
        }

        const float center = in[pos];

        std::vector<int2> pairs = generate_int_pairs(-r,r,-r,r);

        // Much, much slower
        /*float2 sum_and_t = make_float2(0.0f, 0.0f);
        sum_and_t = std::accumulate(pairs.begin(), pairs.end(), make_float2(0.0f, 0.0f), [=](float2 acc, int2 p) {
            uint2 curPos = make_uint2(clamp(x+p.x, 0u, size.x-1),
                                      clamp(y+p.y, 0u, size.y-1));

            const float curPix = in[curPos.x + curPos.y*size.x];

            if (curPix > 0) {
                const float mod = sq(curPix - center);
                const float factor = gaussian[p.x+r]*gaussian[p.y+r]*expf(-mod / e_d_squared_2);
                acc.y += factor * curPix;
                acc.x += factor;
            }
            return acc;
        });
        return sum_and_t.y/sum_and_t.x;*/
        
        float t = 0.0f;
        float sum = 0.0f;
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
        return t/sum;
    });
}
