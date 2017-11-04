#include <cstdlib>
#include <commons.h>

float4 raycast(const Volume volume, const uint2 pos, const Matrix4 view,
    const float nearPlane, const float farPlane, const float step,
    const float largestep);