#include <kernels_stl.h>

// STL TODO
void halfSampleRobustImageKernel(std::vector<float> &out, std::vector<float> in, uint2 inSize, const float e_d, const int r) {
    uint2 outSize = make_uint2(inSize.x/2, inSize.y/2);
    unsigned int y;

    // Map to output elements
    for (y=0; y<outSize.y; y++) {
        for (unsigned int x=0; x<outSize.x; x++) {
            uint2 pixel = make_uint2(x,y);
            const uint2 centerPixel = pixel*2;

            float sum = 0.0f;
            float t = 0.0f;
            const float center = in[centerPixel.x + centerPixel.y * inSize.x];
            // std::iota
            for (int i=-r+1; i<=r; ++i) {
                for (int j=-r+1; j<=r; ++j) {
                    uint2 cur = make_uint2(clamp(make_int2(centerPixel.x+j, centerPixel.y+i), make_int2(0),
                                                 make_int2(outSize.x*2-1, outSize.y*2-1)));
                    float current = in[cur.x + cur.y * inSize.x];
                    if (fabsf(current - center) < e_d) {
                        sum += 1.0f;
                        t += current;
                    }
                }
            }
            out[pixel.x + pixel.y * outSize.x] = t/sum;
        }
    }
}

// Original
/*void halfSampleRobustImageKernel(float *out, const float *in, uint2 inSize, const float e_d, const int r) {
    uint2 outSize = make_uint2(inSize.x/2, inSize.y/2);
    unsigned int y;

    // Map to output elements
    for (y=0; y<outSize.y; y++) {
        for (unsigned int x=0; x<outSize.x; x++) {
            uint2 pixel = make_uint2(x,y);
            const uint2 centerPixel = pixel*2;

            float sum = 0.0f;
            float t = 0.0f;
            const float center = in[centerPixel.x + centerPixel.y * inSize.x];
            // Reduction
            for (int i=-r+1; i<=r; ++i) {
                for (int j=-r+1; j<=r; ++j) {
                    uint2 cur = make_uint2(clamp(make_int2(centerPixel.x+j, centerPixel.y+i), make_int2(0),
                                                 make_int2(outSize.x*2-1, outSize.y*2-1)));
                    float current = in[cur.x + cur.y * inSize.x];
                    if (fabsf(current - center) < e_d) {
                        sum += 1.0f;
                        t += current;
                    }
                }
            }
            out[pixel.x + pixel.y * outSize.x] = t/sum;
        }
    }
}*/

// STL TODO
void depth2vertexKernel(std::vector<float3> &vertex, const std::vector<float> depth, uint2 imageSize, const Matrix4 invK) {
    // todo: conditional map to vertex
    for (unsigned int y=0; y<imageSize.y; y++) {
        for (unsigned int x=0; x<imageSize.x; x++) {
            if (depth[x + y*imageSize.x] > 0) {
                // invK - intrinsic matrix of the camera
                vertex[x + y*imageSize.x] = depth[x + y*imageSize.x]
                        * (rotate(invK, make_float3(x,y,1.f)));
            } else {
                vertex[x + y*imageSize.x] = make_float3(0);
            }
        }
    }
}

// Original
/*void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize, const Matrix4 invK) {
    for (unsigned int y=0; y<imageSize.y; y++) {
        for (unsigned int x=0; x<imageSize.x; x++) {
            if (depth[x + y*imageSize.x] > 0) {
                // invK - intrinsic matrix of the camera
                vertex[x + y*imageSize.x] = depth[x + y*imageSize.x]
                        * (rotate(invK, make_float3(x,y,1.f)));
            } else {
                vertex[x + y*imageSize.x] = make_float3(0);
            }
        }
    }
}*/

void vertex2normalKernel(std::vector<float3> &out, const std::vector<float3> in, uint2 imageSize) {
    // Map to out
	for (unsigned int y = 0; y < imageSize.y; y++) {
		for (unsigned int x = 0; x < imageSize.x; x++) {
			const uint2 pleft  = make_uint2(max(int(x) - 1, 0), y);
			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1), y);
			const uint2 pup    = make_uint2(x, max(int(y) - 1, 0));
			const uint2 pdown  = make_uint2(x, min(y + 1, ((int) imageSize.y) - 1));

			const float3 left  = in[pleft.x + imageSize.x * pleft.y];
			const float3 right = in[pright.x + imageSize.x * pright.y];
			const float3 up    = in[pup.x + imageSize.x * pup.y];
			const float3 down  = in[pdown.x + imageSize.x * pdown.y];

			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
				out[x + y * imageSize.x].x = KFUSION_INVALID;
				continue;
			}
			const float3 dxv = right - left;
			const float3 dyv = down - up;
			out[x + y * imageSize.x] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
		}
	}
}

// TrackData includes the errors (how far I am to the pixel in front of me)
void trackKernel(std::vector<TrackData> &output, const std::vector<float3> inVertex,
        const std::vector<float3> inNormal, uint2 inSize, const float3* refVertex,
        const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
        const Matrix4 view, const float dist_threshold,
        const float normal_threshold) {

    uint2 pixel = make_uint2(0, 0);
    unsigned int pixely, pixelx;
    for (pixely = 0; pixely < inSize.y; pixely++) {
        for (pixelx = 0; pixelx < inSize.x; pixelx++) {
            pixel.x = pixelx;
            pixel.y = pixely;

            TrackData &row = output[pixel.x + pixel.y * refSize.x];

            if (inNormal[pixel.x + pixel.y * inSize.x].x == KFUSION_INVALID) {
                row.result = -1;
                continue;
            }

            const float3 projectedVertex = Ttrack * inVertex[pixel.x + pixel.y * inSize.x];
            const float3 projectedPos = view * projectedVertex;
            const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
                                                 projectedPos.y / projectedPos.z + 0.5f);
            if (projPixel.x < 0 || projPixel.x > refSize.x - 1
                    || projPixel.y < 0 || projPixel.y > refSize.y - 1) {
                row.result = -2;
                continue;
            }

            const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
            const float3 referenceNormal = refNormal[refPixel.x + refPixel.y * refSize.x];

            if (referenceNormal.x == KFUSION_INVALID) {
                row.result = -3;
                continue;
            }

            const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x] - projectedVertex;
            const float3 projectedNormal = rotate(Ttrack, inNormal[pixel.x + pixel.y * inSize.x]);

            if (length(diff) > dist_threshold) {
                row.result = -4;
                continue;
            }
            if (dot(projectedNormal, referenceNormal) < normal_threshold) {
                row.result = -5;
                continue;
            }
            row.result = 1;
            row.error = dot(referenceNormal, diff);
            ((float3 *) row.J)[0] = referenceNormal;
            ((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
        }
    }
}

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

void new_reduce(std::vector<float> &out, std::vector<TrackData> trackData, const uint2 Jsize, const uint2 out_size) {

    // Loop through a generated list of block indices next
    /*std::vector<int> block_indices(8);
    std::iota(block_indices.begin(), block_indices.end(), 0);
    std::for_each(block_indices.begin(), block_indices.end(), [=]() {
    });*/

    for (uint blockIndex=0; blockIndex<8; blockIndex++)  {
        // Prepare an iterator pointing to the output
        auto output_sums = out.begin() + blockIndex*32;
        std::fill(output_sums, output_sums+32, 0);

        // Generate y values at step 8
        std::vector<int> y(out_size.y/8 + 1);
        std::generate(y.begin(), y.end(), [n=blockIndex]() mutable { int retval = n; n+=8; return retval; });

        std::for_each(y.begin(), y.end(), [=](int y) {
            auto input_iterator_begin = trackData.begin() + y*Jsize.x;
            auto input_iterator_end = trackData.begin() + y*Jsize.x + out_size.x;
            std::accumulate(input_iterator_begin, input_iterator_end, output_sums, reduce_single_row);
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