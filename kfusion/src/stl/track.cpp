#include <kernels_stl.h>

//#include <experimental/algorithm>
//#include <sycl/execution_policy>
//#include <experimental/numeric>

void halfSampleRobustImageKernel(std::vector<float> &out, std::vector<float> in, uint2 inSize, const float e_d, const int r) {
    uint2 outSize = make_uint2(inSize.x/2, inSize.y/2);

    std::vector<int> rows = iota(outSize.y);
    std::vector<int> cols = iota(outSize.x);

    //for (uint y=0; y<outSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](int y) {

        //for (uint x=0; x<outSize.x; x++) {
        std::for_each(cols.begin(), cols.end(), [&](int x) {
            uint2 pixel = make_uint2(x,y);
            const uint2 centerPixel = pixel*2;

            float sum = 0.0f;
            float t = 0.0f;
            const float center = in[centerPixel.x + centerPixel.y*inSize.x];

            std::vector<int2> pairs = generate_int_pairs(-r+1, r, -r+1, r);
            
            // Stencil
            std::for_each(pairs.begin(), pairs.end(), [&](int2 p) {
                uint2 cur = make_uint2(clamp(make_int2(centerPixel.x+p.x, centerPixel.y+p.y), make_int2(0),
                                             make_int2(outSize.x*2-1, outSize.y*2-1)));
                float current = in[cur.x + cur.y*inSize.x];
                if (fabsf(current - center) < e_d) {
                    sum += 1.0f;
                    t += current;
                }
            });
            
            out[pixel.x + pixel.y*outSize.x] = t/sum;
        });
    });
}

void depth2vertexKernel(std::vector<float3> &vertex, const std::vector<float> depth, uint2 imageSize, const Matrix4 invK) {

    std::vector<int> rows = iota(imageSize.y);
    std::vector<int> cols = iota(imageSize.x);

    //for (uint y=0; y<imageSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](int y) {

        int offset = y*imageSize.x;

        // Can't do a transform since we need x coordinate in rotate()
        // auto left  = vertex.begin() + imageSize.x * y;
        // auto right = vertex.begin() + imageSize.x * (y+1) - 1;

        // When doing a transform on a range, you can only see the depth values, not the x coordinate
        // vertex[x + offset] = std::transform(left, right, [=](int ptr) {
        //     if (depth[ptr] > 0) {
        //          return depth[x + offset] * rotate(invK, make_float3(x,y,1.f));
        //     }
        //     return vertex[x + offset] = make_float3(0);
        // });

        //for (uint x=0; x<imageSize.x; x++) {
        std::for_each(cols.begin(), cols.end(), [&](int x) {
            if (depth[x + offset] > 0) {
                vertex[x + offset] = depth[x + offset] * rotate(invK, make_float3(x,y,1.f));
            } else {
                vertex[x + offset] = make_float3(0);
            }
        });
    });
}

void vertex2normalKernel(std::vector<float3> &out, const std::vector<float3> in, uint2 imageSize) {

    std::vector<int> rows = iota(imageSize.y);
    std::vector<int> cols = iota(imageSize.x);

    //for (uint y=0; y<imageSize.y; y++) {
    std::for_each(rows.begin(), rows.end(), [&](int y) {

        //for (uint x=0; x<imageSize.x; x++) {
        std::for_each(cols.begin(), cols.end(), [&](int x) {
			const uint2 pleft  = make_uint2(max(int(x) - 1, 0), y);
			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1), y);
			const uint2 pup    = make_uint2(x, max(int(y) - 1, 0));
			const uint2 pdown  = make_uint2(x, min(y + 1, ((int) imageSize.y) - 1));

            // Stencil
			const float3 left  = in[pleft.x + imageSize.x * pleft.y];
			const float3 right = in[pright.x + imageSize.x * pright.y];
			const float3 up    = in[pup.x + imageSize.x * pup.y];
			const float3 down  = in[pdown.x + imageSize.x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y * imageSize.x].x = KFUSION_INVALID;
				return;
			}
			const float3 dxv = right - left;
			const float3 dyv = down - up;
			out[x + y*imageSize.x] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
		});
	});
}

// TrackData includes the errors (how far I am to the pixel in front of me)
void trackKernel(std::vector<TrackData> &output, const std::vector<float3> inVertex,
        const std::vector<float3> inNormal, uint2 inSize, const float3* refVertex,
        const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
        const Matrix4 view, const float dist_threshold,
        const float normal_threshold) {

    uint2 pixel = make_uint2(0, 0);
    //uint pixely, pixelx;

    std::vector<int> rows = iota(inSize.y);
    std::vector<int> cols = iota(inSize.x);

    //for (pixely=0; pixely<inSize.y; pixely++) {
    std::for_each(rows.begin(), rows.end(), [&](int pixely) {
        
        //for (pixelx=0; pixelx<inSize.x; pixelx++) {
        std::for_each(cols.begin(), cols.end(), [&](int pixelx) {
            pixel.x = pixelx;
            pixel.y = pixely;

            TrackData &row = output[pixel.x + pixel.y*refSize.x];

            // If the input normal is invalid
            if (inNormal[pixel.x + pixel.y*inSize.x].x == KFUSION_INVALID) {
                row.result = -1;
                return;
            }

            // If the projected pixel is out of the frame
            const float3 projectedVertex = Ttrack * inVertex[pixel.x + pixel.y*inSize.x];
            const float3 projectedPos = view * projectedVertex;
            const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
                                                 projectedPos.y / projectedPos.z + 0.5f);
            if (projPixel.x < 0 || projPixel.x > refSize.x - 1
                    || projPixel.y < 0 || projPixel.y > refSize.y - 1) {
                row.result = -2;
                return;
            }

            const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
            const float3 referenceNormal = refNormal[refPixel.x + refPixel.y*refSize.x];

            // If the reference normal is invalid
            if (referenceNormal.x == KFUSION_INVALID) {
                row.result = -3;
                return;
            }

            const float3 diff = refVertex[refPixel.x + refPixel.y*refSize.x] - projectedVertex;
            const float3 projectedNormal = rotate(Ttrack, inNormal[pixel.x + pixel.y * inSize.x]);

            // If the coordinate difference is beyond a threshold (outlier)
            if (length(diff) > dist_threshold) {
                row.result = -4;
                return;
            }

            // If the normal product is below a threshold
            if (dot(projectedNormal, referenceNormal) < normal_threshold) {
                row.result = -5;
                return;
            }
            row.result = 1;
            row.error = dot(referenceNormal, diff);
            ((float3 *) row.J)[0] = referenceNormal;
            ((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
        });
    });
}

// Associativity? (acc and row are of different operators)
auto reduce_single_row = [](auto &sums, TrackData row) {
    if (row.result<1) {
        sums[29] += row.result == -4 ? 1 : 0; // (sums+28)[1]
        sums[30] += row.result == -5 ? 1 : 0; // (sums+28)[2]
        sums[31] += row.result > -4 ? 1 : 0;  // (sums+28)[3]
        return sums;
    }
    // Error part
    sums[0] += row.error * row.error;

    // JTe part
    sums[1] += row.error * row.J[0];
    sums[2] += row.error * row.J[1];
    sums[3] += row.error * row.J[2];
    sums[4] += row.error * row.J[3];
    sums[5] += row.error * row.J[4];
    sums[6] += row.error * row.J[5];

    // JTJ part, unfortunatly the double loop is not unrolled well...
    sums[7] += row.J[0] * row.J[0];  // (sums+7)[0]
    sums[8] += row.J[0] * row.J[1];  // (sums+7)[1]
    sums[9] += row.J[0] * row.J[2];  // (sums+7)[2]
    sums[10] += row.J[0] * row.J[3]; // (sums+7)[3]
    sums[11] += row.J[0] * row.J[4]; // (sums+7)[4]
    sums[12] += row.J[0] * row.J[5]; // etc.
    
    sums[13] += row.J[1] * row.J[1];
    sums[14] += row.J[1] * row.J[2];
    sums[15] += row.J[1] * row.J[3];
    sums[16] += row.J[1] * row.J[4];
    sums[17] += row.J[1] * row.J[5];

    sums[18] += row.J[2] * row.J[2];
    sums[19] += row.J[2] * row.J[3];
    sums[20] += row.J[2] * row.J[4];
    sums[21] += row.J[2] * row.J[5];

    sums[22] += row.J[3] * row.J[3];
    sums[23] += row.J[3] * row.J[4];
    sums[24] += row.J[3] * row.J[5];

    sums[25] += row.J[4] * row.J[4];
    sums[26] += row.J[4] * row.J[5];

    sums[27] += row.J[5] * row.J[5];
    sums[28] += 1; // extra info here (sums+28)[0]
    return sums;
};

// Previous workaround
/*struct reduce_single_row {
    template <typename T>
    T operator()(T &sums, TrackData row) const {
        
        sums[0] += row.error * row.error;
        sums[1] += row.error * row.J[0];
        sums[2] += row.error * row.J[1];
        ...
        sums[28] += 1; // extra info here (sums+28)[0]
        return sums;
    };
};*/

void new_reduce(std::vector<float> &out, std::vector<TrackData> trackData, const uint2 Jsize, const uint2 out_size) {

    // Loop through a generated list of block indices next
    /*std::vector<int> block_indices(8);
    std::iota(block_indices.begin(), block_indices.end(), 0);
    std::for_each(block_indices.begin(), block_indices.end(), [=]() {
    });*/

    //sycl::sycl_execution_policy<class reduce_row> par;

    // Reduction
    for (uint blockIndex=0; blockIndex<8; blockIndex++)  {
        // Prepare an iterator pointing to the output
        auto output_sums = out.begin() + blockIndex*32;
        std::fill(output_sums, output_sums+32, 0);

        // Generate y values at step 8
        std::vector<int> y(out_size.y/8 + 1);
        std::generate(y.begin(), y.end(), [n=blockIndex]() mutable {
            int retval = n; n+=8; return retval;
        });

        std::for_each(y.begin(), y.end(), [=](int y) {
            auto begin = trackData.begin() + y*Jsize.x;
            auto end = trackData.begin() + y*Jsize.x + out_size.x - 1;
            // Accumulate used unconventionally?
            std::accumulate(begin, end, output_sums, reduce_single_row);
            //output_sums = std::experimental::parallel::reduce(par, begin, end, 0.0f, reduce_single_row);
            
            // Transform from a TrackData instance to float
            //std::experimental::parallel::transform_reduce();

            /*std::vector<double> xvalues(10007, 1.0), yvalues(10007, 1.0);
 
            double result = std::experimental::parallel::transform_reduce(
                xvalues.begin(), xvalues.end(),
                yvalues.begin(), 0.0
            );
            std::cout << result << '\n';*/
        });
    }
}

void reduceKernel(std::vector<float> &out, std::vector<TrackData> trackData, const uint2 Jsize, const uint2 out_size) {
    new_reduce(out, trackData, Jsize, out_size);

    TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out.data());
    for (int j=1; j<8; ++j) {
        values[0] += values[j];
    }
}