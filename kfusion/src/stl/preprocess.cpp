#include <kernels_stl.h>

// #include <boost/range/adaptor/strided.hpp>
// #include <boost/range/algorithm/copy.hpp>
// #include <boost/assign.hpp>

#include <experimental/algorithm>
#include <sycl/execution_policy>

namespace {
    sycl::sycl_execution_policy<class bilateral_filter> bilateral_filter_par;
}

void mm2metersKernel(std::vector<float> &out, uint2 outSize, const std::vector<uint16_t> in, uint2 inSize) {
    if ((inSize.x < outSize.x) || (inSize.y < outSize.y))           { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    if ((inSize.x / outSize.x != inSize.y / outSize.y))             { std::cerr << "Invalid ratio." << std::endl; exit(1); }
    
    int ratio = inSize.x / outSize.x;

    // This looks ugly
    // std::generate(out.begin(), out.end(), [x=0,y=0,=]() mutable {
    //     float ret = in[x*ratio + inSize.x*y*ratio] / 1000.0f;
	// 	x++;
	// 	if (x == outSize.x) {
	// 		x = 0;
	// 		y++;
	// 	}
	// 	return ret;
    // });

    //std::transform(in.begin(), in.end(), out.begin(), [](uint16_t inValue) { return (float) inValue / 1000.0f; });

    //
    // Gather - is there any way to make an iterator skip evety other element?
    //
    for (uint y=0; y<outSize.y; y++) {
		for (uint x=0; x<outSize.x; x++) {
			out[x + outSize.x*y] = in[x*ratio + inSize.x*y*ratio] / 1000.0f;
		}
    };

    // Boost
    // One-line wonder
    //using namespace boost::assign;
    //boost::copy(input | boost::adaptors::strided(ratio), out.begin());
}

void bilateralFilterKernel(std::vector<float> &out, const std::vector<float> in, uint2 size, const std::vector<float> gaussian, float e_d, int r) {
    float e_d_squared_2 = e_d * e_d * 2;
    
    // Stencil
    std::vector<uint> pixels(size.x*size.y);
    std::iota(pixels.begin(), pixels.end(), 0);

    // for (uint y=0; y<size.y; y++) {
    //     for (uint x=0; x<size.x; x++) {
    //std::for_each(pixels.begin(), pixels.end(), [&](uint pos) {
    std::transform(pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
    //std::experimental::parallel::transform(bilateral_filter_par, pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
        // uint pos = x + y*size.x;
        uint x = pos % size.x;
        uint y = pos / size.x;

        //if (pos%10 == 0)
        //   std::cout << pos << " " << y << " " << x << std::endl;
        
        if (in[pos] == 0) {
            //out[pos] = 0;
            return 0.0f;
        }

        const float center = in[pos];
        float t = 0.0f;
        float sum = 0.0f;

        /*std::vector<int2> pairs = generate_int_pairs(-r,r,-r,r);
        std::vector<std::pair<int2, float>> near_depths(pairs.size());

        std::transform(pairs.begin(), pairs.end(), near_depths.begin(), [=](int2 p) {
            uint2 curPos = make_uint2(clamp(x+p.x, 0u, size.x-1), clamp(y+p.y, 0u, size.y-1));
            return std::make_pair(p, in[curPos.x + curPos.y*size.x]);
        });

        auto reduce_neighbourhood = [=](float2 acc, std::pair<int2, float> p) {
            if (p.second > 0) {
                const float mod = std::pow(p.second - center, 2);
                const float factor = gaussian[p.first.x+r]*gaussian[p.first.y+r] * expf(-mod / e_d_squared_2);
                acc.x += (factor * p.second);
                acc.y += factor;
            }
            return acc;
        };*/

        //float2 result = std::accumulate(near_depths.begin(), near_depths.end(), make_float2(0.0f, 0.0f), reduce_neighbourhood);
        //float2 result = std::experimental::parallel::reduce(bilateral_filter_par, near_depths.begin(), near_depths.end(), make_float2(0.0f, 0.0f), reduce_neighbourhood);

        std::vector<int2> pairs = generate_int_pairs(-r,r,-r,r);
        std::for_each(pairs.begin(), pairs.end(), [&](int2 p) {
            uint2 curPos = make_uint2(clamp(x+p.x, 0u, size.x-1), clamp(y+p.y, 0u, size.y-1));
            const float curPix = in[curPos.x + curPos.y*size.x];
            if (curPix > 0) {
                const float mod = sq(curPix - center);
                const float factor = gaussian[p.x+r]*gaussian[p.y+r]*expf(-mod / e_d_squared_2);
                t += factor * curPix;
                sum += factor;
            }
        });

        // Original            
        // for (int i=-r; i<=r; i++) {
        //     for (int j=-r; j<=r; j++) {
        //         uint2 curPos = make_uint2(clamp(x+i, 0u, size.x-1), clamp(y+j, 0u, size.y-1));
        //         const float curPix = in[curPos.x + curPos.y*size.x];
        //         if (curPix > 0) {
        //             const float mod = sq(curPix - center);
        //             const float factor = gaussian[i+r]*gaussian[j+r]*expf(-mod / e_d_squared_2);
        //             t += factor * curPix;
        //             sum += factor;
        //         }
        //     }
        // }
        //out[pos] = t/sum;
        //out[pos] = result.x / result.y;
        return t/sum;
    });
}
