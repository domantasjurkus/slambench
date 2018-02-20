#include <kernels_stl.h>

//#include <experimental/algorithm>
//#include <sycl/execution_policy>

/*namespace {
    sycl::sycl_execution_policy<class half_sample> half_sample_par;
    sycl::sycl_execution_policy<class track_kernel> track_kernel_par;
}*/

void halfSampleRobustImageKernel(std::vector<float> &out,
        std::vector<float> in,
        uint2 inSize,
        const float e_d,
        const int r) {

    uint2 outSize = make_uint2(inSize.x/2, inSize.y/2);

    std::vector<uint> pixels(outSize.x*outSize.y);
    std::iota(pixels.begin(), pixels.end(), 0);

    std::transform(pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
    //std::experimental::parallel::transform(half_sample_par, pixels.begin(), pixels.end(), out.begin(), [=](uint pos) {
        uint x = pos % outSize.x;
        uint y = pos / outSize.x;

        uint2 pixel = make_uint2(x,y);
        const uint2 centerPixel = pixel*2;

        float sum = 0.0f;
        float t = 0.0f;
        // in - static tdess pattern
        const float center = in[centerPixel.x + centerPixel.y*inSize.x];

        std::vector<int2> pairs = generate_int_pairs(-r+1, r, -r+1, r);
        
        std::for_each(pairs.begin(), pairs.end(), [&](int2 p) {
            uint2 cur = make_uint2(clamp(make_int2(centerPixel.x+p.x, centerPixel.y+p.y), make_int2(0),
                                         make_int2(outSize.x*2-1, outSize.y*2-1)));
            float current = in[cur.x + cur.y*inSize.x];
            if (fabsf(current - center) < e_d) {
                sum += 1.0f;
                t += current;
            }
        });
        
        //out[pixel.x + pixel.y*outSize.x] = t/sum;
        return t/sum;
    });

    /*for (uint y=0; y<outSize.y; y++) {
        for (uint x=0; x<outSize.x; x++) {
            
            uint2 pixel = make_uint2(x,y);
            const uint2 centerPixel = pixel*2;

            float sum = 0.0f;
            float t = 0.0f;
            const float center = in[centerPixel.x + centerPixel.y*inSize.x];

            std::vector<int2> pairs = generate_int_pairs(-r+1, r, -r+1, r);
            
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
        }
    }*/
}

void depth2vertexKernel(std::vector<float3> &vertex,
        const std::vector<float> depth,
        uint2 imageSize,
        const Matrix4 invK) {

    std::vector<uint> pixels(imageSize.x*imageSize.y);
    std::iota(pixels.begin(), pixels.end(), 0);

    std::transform(depth.begin(), depth.end(), pixels.begin(), vertex.begin(), [=](float d, uint pos) {
        uint x = pos % imageSize.x;
        uint y = pos / imageSize.x;

        return (d>0) ? d * rotate(invK, make_float3(x,y,1.0f)) : make_float3(0);
    });

    /*for (uint y=0; y<imageSize.y; y++) {
        int offset = y*imageSize.x;

        // Can't do a transform since we need x coordinate in rotate()
        for (uint x=0; x<imageSize.x; x++) {
            if (depth[x + offset] > 0) {
                vertex[x + offset] = depth[x + offset] * rotate(invK, make_float3(x,y,1.f));
            } else {
                vertex[x + offset] = make_float3(0);
            }
        };
    };*/
}

void vertex2normalKernel(std::vector<float3> &out, const std::vector<float3> in, uint2 imageSize) {

    // Segfault
    std::vector<uint> pixels(imageSize.x*imageSize.y);
    std::iota(pixels.begin(), pixels.end(), 0);

    // std::cout << "\n\n" << in.size() << "\n\n" << std::endl;

    // std::transform(in.begin(), in.end(), pixels.begin(), out.begin(), [=](float3 input, uint pos) {
    //     uint x = pos % imageSize.x;
    //     uint y = pos / imageSize.x;

    //     const uint2 pleft  = make_uint2(max(int(x) - 1, 0), y);
    //     const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1), y);
    //     const uint2 pup    = make_uint2(x, max(int(y) - 1, 0));
    //     const uint2 pdown  = make_uint2(x, min(y + 1, ((int) imageSize.y) - 1));

    //     std::cout << pleft.x + imageSize.x * pleft.y << std::endl;

    //     const float3 left  = in[pleft.x + imageSize.x * pleft.y];
    //     const float3 right = in[pright.x + imageSize.x * pright.y];
    //     const float3 up    = in[pup.x + imageSize.x * pup.y];
    //     const float3 down  = in[pdown.x + imageSize.x * pdown.y];

    //     if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
    //         return make_float3(KFUSION_INVALID, 0.0f, 0.0f);
    //     }
    //     const float3 dxv = right - left;
    //     const float3 dyv = down - up;
    //     return normalize(cross(dyv, dxv));
    // });

    // Stencil
    for (uint y=0; y<imageSize.y; y++) {
        for (uint x=0; x<imageSize.x; x++) {

			const uint2 pleft  = make_uint2(max(int(x) - 1, 0), y);
			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1), y);
			const uint2 pup    = make_uint2(x, max(int(y) - 1, 0));
			const uint2 pdown  = make_uint2(x, min(y + 1, ((int) imageSize.y) - 1));

			const float3 left  = in[pleft.x + imageSize.x * pleft.y];
			const float3 right = in[pright.x + imageSize.x * pright.y];
			const float3 up    = in[pup.x + imageSize.x * pup.y];
			const float3 down  = in[pdown.x + imageSize.x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y*imageSize.x].x = KFUSION_INVALID;
				return;
			}
			const float3 dxv = right - left;
			const float3 dyv = down - up;
			out[x + y*imageSize.x] = normalize(cross(dyv, dxv));
		};
	};
}

void trackKernel(std::vector<TrackData> &output,
        const std::vector<float3> inVertex,
        const std::vector<float3> inNormal,
        uint2 inSize,
        const std::vector<float3> refVertex,
        const std::vector<float3> refNormal,
        uint2 refSize,
        const Matrix4 Ttrack,
        const Matrix4 view,
        const float dist_threshold,
        const float normal_threshold) {

    std::vector<uint> pixels(inSize.x*inSize.y);
    std::iota(pixels.begin(), pixels.end(), 0);

    // for (uint pixely=0; pixely<inSize.y; pixely++) {
    //     for (uint pixelx=0; pixelx<inSize.x; pixelx++) {
    std::transform(pixels.begin(), pixels.end(), output.begin(), [=](uint pos) {
    //std::experimental::parallel::transform(track_kernel_par, pixels.begin(), pixels.end(), output.begin(), [=](uint pos) {
        uint x = pos % inSize.x;
        uint y = pos / inSize.x;

        TrackData row = output[x + y*refSize.x];

        // If the input normal is invalid
        if (inNormal[x + y*inSize.x].x == KFUSION_INVALID) {
            row.result = -1;
            //continue;
            return row;
        }

        // If the projected pixel is out of the frame
        const float3 projectedVertex = Ttrack * inVertex[x + y*inSize.x];
        const float3 projectedPos = view * projectedVertex;
        const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
                                             projectedPos.y / projectedPos.z + 0.5f);
        if (projPixel.x<0 || projPixel.x>refSize.x-1 || projPixel.y<0 || projPixel.y>refSize.y-1) {
            row.result = -2;
            //continue;
            return row;
        }

        const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
        const float3 referenceNormal = refNormal[refPixel.x + refPixel.y*refSize.x];

        // If the reference normal is invalid
        if (referenceNormal.x == KFUSION_INVALID) {
            row.result = -3;
            //continue;
            return row;
        }

        const float3 diff = refVertex[refPixel.x + refPixel.y*refSize.x] - projectedVertex;
        const float3 projectedNormal = rotate(Ttrack, inNormal[x + y*inSize.x]);

        // If the coordinate difference is beyond a threshold (outlier)
        if (length(diff) > dist_threshold) {
            row.result = -4;
            //continue;
            return row;
        }

        // If the normal product is below a threshold
        if (dot(projectedNormal, referenceNormal) < normal_threshold) {
            row.result = -5;
            //continue;
            return row;
        }
        row.result = 1;
        
        row.error = dot(referenceNormal, diff);
        ((float3 *) row.J)[0] = referenceNormal;
        ((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
        return row;
    //     };
    // };
    });
}

void new_reduce(std::vector<float> &out,
        std::vector<TrackData> trackData,   // size 19200
        const uint2 trackDataSize,          // 160x120
        const uint2 localimagesize) {       // 40x30

    std::fill(out.begin(), out.end(), 0.0f);

    // No apparent way to std::accumulate into a vector<float>

    // This works
    std::for_each(trackData.begin(), trackData.end(), [&](TrackData td) {
        if (td.result<1) {
            out[29] += td.result == -4 ? 1 : 0;
            out[30] += td.result == -5 ? 1 : 0;
            out[31] += td.result > -4 ? 1 : 0;
            return;
        }
        out[0] += td.error * td.error;
        out[1] += td.error * td.J[0];
        out[2] += td.error * td.J[1];
        out[3] += td.error * td.J[2];
        out[4] += td.error * td.J[3];
        out[5] += td.error * td.J[4];
        out[6] += td.error * td.J[5];
        out[7] += td.J[0] * td.J[0];
        out[8] += td.J[0] * td.J[1];
        out[9] += td.J[0] * td.J[2];
        out[10] += td.J[0] * td.J[3];
        out[11] += td.J[0] * td.J[4];
        out[12] += td.J[0] * td.J[5];
        out[13] += td.J[1] * td.J[1];
        out[14] += td.J[1] * td.J[2];
        out[15] += td.J[1] * td.J[3];
        out[16] += td.J[1] * td.J[4];
        out[17] += td.J[1] * td.J[5];
        out[18] += td.J[2] * td.J[2];
        out[19] += td.J[2] * td.J[3];
        out[20] += td.J[2] * td.J[4];
        out[21] += td.J[2] * td.J[5];
        out[22] += td.J[3] * td.J[3];
        out[23] += td.J[3] * td.J[4];
        out[24] += td.J[3] * td.J[5];
        out[25] += td.J[4] * td.J[4];
        out[26] += td.J[4] * td.J[5];
        out[27] += td.J[5] * td.J[5];
        out[28] += 1;
    });

    /*for (int i=0; i<entry.size(); i++) {
        out[i] = entry[i];
    }*/
}

void reduceKernel(std::vector<float> &out,
        std::vector<TrackData> trackData,
        const uint2 computationSize,
        const uint2 localimagesize) {
    
    new_reduce(out, trackData, computationSize, localimagesize);

    TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out.data());
    for (int j=1; j<8; ++j) {
        values[0] += values[j];
    }
}