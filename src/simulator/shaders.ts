export const pathtracerVS = `#version 300 es
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

export const pathtracerFS = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform vec3 u_hogelPos; // Position of this hogel in world space

// Simple Cornell Box SDF
float map(vec3 p, out int mat) {
    float d = 1e10;
    
    // Floor
    float flr = p.y;
    if(flr < d) { d = flr; mat = 0; }
    
    // Ceiling
    float ceil = 10.0 - p.y;
    if(ceil < d) { d = ceil; mat = 0; }
    
    // Back wall
    float back = 5.0 - p.z;
    if(back < d) { d = back; mat = 0; }
    
    // Left wall
    float left = p.x + 5.0;
    if(left < d) { d = left; mat = 0; }
    
    // Right wall
    float right = 5.0 - p.x;
    if(right < d) { d = right; mat = 0; }
    
    // Light
    vec3 dLight = abs(p - vec3(0, 9.9, 0)) - vec3(1.5, 0.1, 1.5);
    float lght = min(max(dLight.x, max(dLight.y, dLight.z)), 0.0) + length(max(dLight, 0.0));
    if(lght < d) { d = lght; mat = 1; }
    
    // Sphere 1
    float sph = length(p - vec3(-2, 1.5, -1)) - 1.5;
    if(sph < d) { d = sph; mat = 2; }
    
    // Box
    vec3 bp = p - vec3(2, 1.5, 1);
    // rotate
    float s = sin(0.4), c = cos(0.4);
    bp.xz = mat2(c, -s, s, c) * bp.xz;
    vec3 dBox2 = abs(bp) - vec3(1.2, 1.5, 1.2);
    float bx = min(max(dBox2.x, max(dBox2.y, dBox2.z)), 0.0) + length(max(dBox2, 0.0));
    if(bx < d) { d = bx; mat = 3; }
    
    return d;
}

vec3 getNormal(vec3 p) {
    int m;
    vec2 e = vec2(0.001, 0);
    return normalize(vec3(
        map(p + e.xyy, m) - map(p - e.xyy, m),
        map(p + e.yxy, m) - map(p - e.yxy, m),
        map(p + e.yyx, m) - map(p - e.yyx, m)
    ));
}

void main() {
    // Generate ray direction from hogel looking into the scene
    // v_uv covers a narrow FOV around the Z axis
    vec2 ndc = v_uv * 2.0 - 1.0;
    
    // Assume Hogel panel is at z = -10, looking at z = 0
    vec3 ro = vec3(u_hogelPos.x, u_hogelPos.y, -10.0);
    
    // Calculate ray direction. Angle spread determines the size of the hologram's FOV.
    vec3 rd = normalize(vec3(ndc * 1.5, 1.0));
    
    vec3 col = vec3(0.0);
    float t = 0.0;
    for(int i=0; i<64; i++){
        vec3 p = ro + rd * t;
        int mat;
        float d = map(p, mat);
        if(d < 0.001) {
            vec3 n = getNormal(p);
            
            // Basic localized lighting
            vec3 l_dir = normalize(vec3(0, 9.8, 0) - p);
            float diff = max(dot(n, l_dir), 0.0);
            
            if(mat == 0) {
                // Room walls (Left red, Right green)
                vec3 albedo = vec3(0.8);
                if(p.x < -4.9) albedo = vec3(0.8, 0.1, 0.1);
                if(p.x > 4.9) albedo = vec3(0.1, 0.8, 0.1);
                col = albedo * diff * 0.8 + 0.1;
            } else if(mat == 1) {
                col = vec3(5.0); // emissive light
            } else if(mat == 2) {
                col = vec3(0.8,0.8,0.9) * diff + vec3(0.1);
            } else if(mat == 3) {
                col = vec3(0.9,0.8,0.2) * diff + vec3(0.1);
            }
            
            // Soft shadows
            float st = 0.1;
            float sh = 1.0;
            for(int j=0; j<16; j++){
                int sm;
                float sd = map(p + l_dir * st, sm);
                if(sd < 0.001) { sh = 0.1; break; }
                sh = min(sh, 8.0 * sd / st);
                st += sd;
            }
            if(mat != 1) col *= sh;
            break;
        }
        t += d;
        if(t > 40.0) break;
    }
    
    // Output raw color for diagnostics preview, but the fringe shader will read the luminance
    outColor = vec4(col, 1.0);
}
`;

export const fringeFS = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_lightfield;
uniform float u_time;

void main() {
    // Generate interference fringes from the Lightfield.
    // The Lightfield resolution (64x64) 
    
    float intensity = 0.0;
    float max_intensity = 0.0;
    
    // We sum up cosine plane waves, each parameterized by the ray angles (uv).
    // Note: Due to WebGL limits on loops, we can sample fewer points of the lightfield
    // or unroll, but 16x16 is stable universally. We'll use 32x32 via a step modifier.
    const int GRID = 32;
    for(int y = 0; y < GRID; y++) {
        for(int x = 0; x < GRID; x++) {
            vec2 luv = vec2(float(x), float(y)) / float(GRID);
            
            // Extract Luma mathematically from the RGB lightfield
            vec3 colorSample = texture(u_lightfield, luv).rgb;
            float L = dot(colorSample, vec3(0.2126, 0.7152, 0.0722));
            
            if(L > 0.05) {
                // Map Lightfield Coordinates (0..1) to Spatial Frequencies (-150 to +150)
                vec2 freq = (luv - 0.5) * 200.0;
                
                // Hologram Subpixel Plane Phase
                float phase = dot(freq, v_uv * 100.0); // scalar scaling of subpixels
                
                intensity += L * cos(phase);
                max_intensity += L;
            }
        }
    }
    
    // Normalize and add bias
    if(max_intensity > 0.0) {
        intensity = (intensity / max_intensity) * 0.5 + 0.5;
    } else {
        intensity = 0.5;
    }
    
    // For FFT processing, we put Real part in R, Imaginary in G (0.0)
    outColor = vec4(intensity, 0.0, 0.0, 1.0);
}
`;

export const fftFS = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_tex;
uniform int u_stage;      // 0 for bit reverse, 1..N for butterflies
uniform int u_horizontal; // 1 or 0
uniform int u_N;          // e.g. 512

const float PI = 3.14159265359;

int bitReverse(int i, int numBits) {
    int res = 0;
    for(int j=0; j<numBits; j++) {
        res = (res << 1) | (i & 1);
        i >>= 1;
    }
    return res;
}

void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    int idx = u_horizontal == 1 ? coord.x : coord.y;
    int other_idx = u_horizontal == 1 ? coord.y : coord.x;
    
    int numBits = int(log2(float(u_N)));
    
    if(u_stage == 0) {
        // purely bit reversal pass
        int rev = bitReverse(idx, numBits);
        ivec2 readCoord = u_horizontal == 1 ? ivec2(rev, other_idx) : ivec2(other_idx, rev);
        vec4 val = texelFetch(u_tex, readCoord, 0);
        outColor = val;
    } else {
        // Butterfly pass
        int span = 1 << u_stage;       // 2, 4, 8...
        int halfSpan = span >> 1;      // 1, 2, 4...
        int k = idx % span;
        
        int isTop = k < halfSpan ? 1 : 0;
        int twiddleK = k % halfSpan;
        
        // Twiddle factor
        float angle = -2.0 * PI * float(twiddleK) / float(span);
        vec2 W = vec2(cos(angle), sin(angle));
        
        // Calculate pairs
        int pairIdx = isTop == 1 ? (idx + halfSpan) : (idx - halfSpan);
        ivec2 topCoord, botCoord;
        if(isTop == 1) {
            topCoord = coord;
            botCoord = u_horizontal == 1 ? ivec2(pairIdx, coord.y) : ivec2(coord.x, pairIdx);
        } else {
            topCoord = u_horizontal == 1 ? ivec2(pairIdx, coord.y) : ivec2(coord.x, pairIdx);
            botCoord = coord;
        }
        
        vec2 a = texelFetch(u_tex, topCoord, 0).rg;
        vec2 b = texelFetch(u_tex, botCoord, 0).rg;
        
        // Complex mult: W * b
        vec2 Wb = vec2(W.x*b.x - W.y*b.y, W.x*b.y + W.y*b.x);
        
        if (isTop == 1) {
            outColor = vec4(a + Wb, 0.0, 1.0);
        } else {
            outColor = vec4(a - Wb, 0.0, 1.0);
        }
    }
}
`;

export const accumulatorFS = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_tex; // The FFT output (Far-field reconstruction)
uniform vec2 u_hogelCenter; // Normalized (-1 to 1) indicating hogel position to shift the ray footprint
uniform float u_intensity;

void main() {
    // We get the FFT which has low frequencies at the corners.
    // FFT shift:
    vec2 sampleUV = v_uv;
    sampleUV = mod(sampleUV + 0.5, 1.0);
    
    vec2 complexEnergy = texture(u_tex, sampleUV).rg;
    float mag = length(complexEnergy); // actual wave magnitude reconstructed
    
    // We add this to the sensor. The area where this hogel puts light depends
    // on its position relative to the camera lens. We'll simplify the camera optics projection natively here.
    
    // Scale intensity because FFT values will be massive for N=512
    float I = mag * u_intensity * 0.000005; 
    
    vec3 color = vec3(I, I * 0.9, I * 0.7); // slightly warm tint for physical realism
    outColor = vec4(color, 1.0);
}
`;

export const displayFS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_tex;
void main() {
    vec3 col = texture(u_tex, v_uv).rgb;
    // Simple Reinhard Tone Mapping
    col = col / (1.0 + col);
    // Gamma correction
    col = pow(col, vec3(1.0/2.2));
    outColor = vec4(col, 1.0);
}
`;
