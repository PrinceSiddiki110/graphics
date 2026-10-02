// ============================================================================
//  KUET CAMPUS v4 - OpenGL 3.3 core + GLFW + GLAD
//  Controls: W A S D move | Q/E down/up | Mouse look | Left SHIFT fast | ESC exit
//            L = EEE ground-floor lights   F = EEE ceiling fans
//            G = EEE generator + wind turbine   M = Mechanical workshop machines
//            C = Civil construction machines  P = Pond water   N = Night mode
//      ============================================================================

// ============================================================================
//  SEGMENT 1 : INCLUDES
// ============================================================================
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace std;

// ============================================================================
//  SEGMENT 2 : SMALL MATH LIBRARY (Vec3 + 4x4 matrices, column-major)
// ============================================================================
const float PI = 3.14159265f;
struct Vec3 { float x, y, z; };
Vec3 V(float x, float y, float z) { return Vec3{x, y, z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return V(a.x - b.x, a.y - b.y, a.z - b.z); }
Vec3 operator+(Vec3 a, Vec3 b) { return V(a.x + b.x, a.y + b.y, a.z + b.z); }
Vec3 operator*(Vec3 a, float s) { return V(a.x * s, a.y * s, a.z * s); }
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(Vec3 a, Vec3 b) { return V(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
Vec3 normalize(Vec3 a) { float l = sqrtf(dot(a, a)); return V(a.x / l, a.y / l, a.z / l); }
float frac(float x) { return x - floorf(x); }

struct Mat4 { float m[16]; };   // element(row,col) = m[col*4 + row]
Mat4 identity() { Mat4 r; for (int i = 0; i < 16; i++) r.m[i] = 0; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1; return r; }
const Mat4 I = identity();
Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; c++) for (int row = 0; row < 4; row++) {
        float s = 0; for (int k = 0; k < 4; k++) s += a.m[k * 4 + row] * b.m[c * 4 + k];
        r.m[c * 4 + row] = s;
    }
    return r;
}
Mat4 translateM(float x, float y, float z) { Mat4 r = identity(); r.m[12] = x; r.m[13] = y; r.m[14] = z; return r; }
Mat4 scaleM(float x, float y, float z) { Mat4 r = identity(); r.m[0] = x; r.m[5] = y; r.m[10] = z; return r; }
Mat4 rotateX(float a) { Mat4 r = identity(); float c = cosf(a), s = sinf(a); r.m[5] = c; r.m[9] = -s; r.m[6] = s; r.m[10] = c; return r; }
Mat4 rotateY(float a) { Mat4 r = identity(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[8] = s; r.m[2] = -s; r.m[10] = c; return r; }
Mat4 rotateZ(float a) { Mat4 r = identity(); float c = cosf(a), s = sinf(a); r.m[0] = c; r.m[4] = -s; r.m[1] = s; r.m[5] = c; return r; }
Mat4 perspectiveM(float fovy, float aspect, float n, float f) {
    Mat4 r; for (int i = 0; i < 16; i++) r.m[i] = 0;
    float t = 1.0f / tanf(fovy / 2);
    r.m[0] = t / aspect; r.m[5] = t; r.m[10] = (f + n) / (n - f); r.m[11] = -1; r.m[14] = (2 * f * n) / (n - f);
    return r;
}
Mat4 orthoM(float l, float r, float b, float t, float n, float f) {
    Mat4 m = identity();
    m.m[0] = 2 / (r - l); m.m[5] = 2 / (t - b); m.m[10] = -2 / (f - n);
    m.m[12] = -(r + l) / (r - l); m.m[13] = -(t + b) / (t - b); m.m[14] = -(f + n) / (f - n);
    return m;
}
Mat4 lookAtM(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye), s = normalize(cross(f, up)), u = cross(s, f);
    Mat4 r = identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye); r.m[13] = -dot(u, eye); r.m[14] = dot(f, eye);
    return r;
}
// place something at (x,y,z) turned by rot. Local +Z is the "front".
Mat4 frame(float x, float z, float rot, float y = 0) { return translateM(x, y, z) * rotateY(rot); }
// heading so that local +X points along (dx,dz)
float headingOf(float dx, float dz) { return atan2f(-dz, dx); }

// ============================================================================
//  SEGMENT 3 : SHADERS  (shadow mapping + procedural textures + wind + water)
// ============================================================================
const string SWAY_GLSL = R"(
uniform float time;
uniform int matId;
vec3 sway(vec3 w, vec3 lp) {
    if (matId == 6) {          // tree leaves move in the wind
        float h = max(w.y - 2.5, 0.0);
        w.x += sin(time*1.6 + w.z*0.30 + w.x*0.17) * 0.07 * h + sin(time*6.0 + dot(lp, vec3(9.1,13.7,5.3))) * 0.07;
        w.z += cos(time*1.3 + w.x*0.35) * 0.05 * h + cos(time*5.0 + dot(lp, vec3(7.3,3.9,11.1))) * 0.07;
        w.y += sin(time*4.0 + dot(lp, vec3(3.0,5.0,7.0))) * 0.05;
    } else if (matId == 11) {          // clothes drying on the roofs flap in the wind
        float k = 0.5 - lp.y; float g = 0.65 + 0.35*sin(time*0.6 + w.x*0.05);
        float ph = time*2.6 + w.x*0.9 + w.z*0.6;
        w.z += sin(ph)*0.22*k*g; w.x += sin(ph*0.7 + 1.3)*0.08*k*g; w.y -= abs(sin(ph))*0.03*k*g;
    }
    return w;
}
)";

const string VS_SRC = string(R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 model, view, proj, lightVP;
out vec3 vNormal; out vec3 vWorld; out vec4 vLight;
)") + SWAY_GLSL + R"(
void main() {
    vec4 w = model * vec4(aPos, 1.0);
    w.xyz = sway(w.xyz, aPos);
    vec3 n = normalize(transpose(inverse(mat3(model))) * aNormal);
    vNormal = n; vWorld = w.xyz;
    vLight = lightVP * vec4(w.xyz + n * 0.08, 1.0);
    gl_Position = proj * view * w;
}
)";

const string FS_SRC = R"(
#version 330 core
in vec3 vNormal; in vec3 vWorld; in vec4 vLight;
uniform vec3 color; uniform vec3 lightDir; uniform vec3 camPos;
uniform float time; uniform int matId;
uniform sampler2D shadowMap;
uniform sampler2DArray lampShadow;                 // one shadow map per lamp post (layer = lamp index)
uniform sampler2DArray floodShadow;                // stadium floodlights (dynamic shadows, 4 layers)
uniform sampler2D lampPosTex, lampMatTex, lampGridTex;   // every lamp: position, shadow matrix, and a grid cell -> nearby lamps list
uniform vec4 gridInfo; uniform ivec2 gridDim;
uniform int nFlood; uniform vec3 floodPos[4]; uniform mat4 floodVP[4];
uniform float roomOn; uniform vec3 roomPos[8]; uniform vec4 roomBox;   // EEE ground-floor tube lights
uniform int nSpots; uniform vec3 spotPos[8]; uniform vec3 spotDir[8]; uniform vec3 spotCol[8]; uniform vec2 spotCone[8];
uniform float night; uniform vec3 skyColor;
out vec4 FragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f*f*(3.0 - 2.0*f);
    return mix(mix(hash(i), hash(i + vec2(1.0,0.0)), f.x), mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), f.x), f.y);
}
float fbm(vec2 p) { float v = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { v += a*noise(p); p *= 2.03; a *= 0.5; } return v; }

vec3 lampLight(vec3 n) {                            // light from the lamp posts around THIS pixel (all lamps work everywhere)
    vec3 acc = vec3(0.0);
    ivec2 c = clamp(ivec2(floor((vWorld.xz - gridInfo.xy) / gridInfo.z)), ivec2(0), gridDim - ivec2(1));
    int cell = c.y * gridDim.x + c.x;
    for (int k = 0; k < 10; k++) {
        float v = texelFetch(lampGridTex, ivec2(k, cell), 0).r;
        if (v < -0.5) break;
        int li = int(v + 0.5);
        vec3 lpos = texelFetch(lampPosTex, ivec2(li, 0), 0).rgb;
        vec3 Lv = lpos - vWorld; float d = length(Lv);
        if (d > 30.0) continue;
        vec3 Ld = Lv / d; float nd = dot(n, Ld);
        if (nd <= 0.0) continue;
        float spot = smoothstep(0.2, 0.5, dot(-Ld, vec3(0.0, -1.0, 0.0)));
        float att = (1.0 / (1.0 + 0.06*d + 0.012*d*d)) * (1.0 - smoothstep(16.0, 30.0, d));
        mat4 M = mat4(texelFetch(lampMatTex, ivec2(li*4, 0), 0), texelFetch(lampMatTex, ivec2(li*4 + 1, 0), 0),
                      texelFetch(lampMatTex, ivec2(li*4 + 2, 0), 0), texelFetch(lampMatTex, ivec2(li*4 + 3, 0), 0));
        vec4 lp = M * vec4(vWorld + n*0.06, 1.0);
        vec3 q = lp.xyz / lp.w * 0.5 + 0.5;
        float sh = 0.0;
        if (q.z < 1.0 && q.x > 0.0 && q.x < 1.0 && q.y > 0.0 && q.y < 1.0) {
            float bias = 0.0004 + 0.0012*(1.0 - nd);
            vec2 ts = 0.6 / vec2(textureSize(lampShadow, 0).xy);
            for (int a = -1; a <= 1; a += 2) for (int b = -1; b <= 1; b += 2) {
                float dd = texture(lampShadow, vec3(q.xy + vec2(float(a), float(b))*ts, float(li))).r;
                sh += (q.z - bias > dd) ? 0.25 : 0.0;
            }
        }
        acc += vec3(1.0, 0.82, 0.55) * nd * att * spot * (1.0 - sh) * 2.6;
    }
    return acc;
}
vec3 floodLight(vec3 n) {                           // big stadium lights over the football field (live shadows of the players)
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        if (i >= nFlood) break;
        vec3 Lv = floodPos[i] - vWorld; float d = length(Lv); vec3 Ld = Lv / d;
        float nd = dot(n, Ld); if (nd <= 0.0) continue;
        vec4 lp = floodVP[i] * vec4(vWorld + n*0.08, 1.0);
        if (lp.w <= 0.0) continue;
        vec3 q = lp.xyz / lp.w * 0.5 + 0.5;
        if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0 || q.z > 1.0) continue;
        vec2 e = min(q.xy, 1.0 - q.xy);
        float edge = smoothstep(0.0, 0.12, min(e.x, e.y));
        float sh = 0.0; vec2 ts = 1.0 / vec2(textureSize(floodShadow, 0).xy);
        float bias = 0.0003 + 0.0010*(1.0 - nd);
        for (int a = -1; a <= 1; a++) for (int b = -1; b <= 1; b++)
            sh += (q.z - bias > texture(floodShadow, vec3(q.xy + vec2(float(a), float(b))*ts, float(i))).r) ? 1.0/9.0 : 0.0;
        acc += vec3(1.0, 0.97, 0.88) * nd * (1.0 / (1.0 + 0.0012*d*d)) * edge * (1.0 - sh) * 2.4;
    }
    return acc;
}
vec3 roomLight(vec3 n) {                            // light of the 8 tube lights under the EEE building
    if (vWorld.x < roomBox.x || vWorld.x > roomBox.y || vWorld.z < roomBox.z || vWorld.z > roomBox.w || vWorld.y > 4.05) return vec3(0.0);
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        vec3 Lv = roomPos[i] - vWorld; float d = length(Lv); if (d > 11.0) continue;
        float nd = dot(n, Lv / d); if (nd <= 0.0) continue;
        acc += vec3(1.0, 0.97, 0.88) * nd * 2.6 / (1.0 + 0.10*d*d);
    }
    return acc + vec3(0.10, 0.10, 0.09);
}
vec3 spotLight(vec3 n) {                            // ground spot lights (KUET hill, pond posts)
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        if (i >= nSpots) break;
        vec3 Lv = spotPos[i] - vWorld; float d = length(Lv); vec3 Ld = Lv / d;
        float nd = dot(n, Ld); if (nd <= 0.0) continue;
        float cone = smoothstep(spotCone[i].x, spotCone[i].y, dot(-Ld, spotDir[i]));
        acc += spotCol[i] * nd * cone / (1.0 + 0.004*d*d);
    }
    return acc;
}
vec3 spotSpec(vec3 wn, vec3 Vv) {                   // glittering reflections of those lights on the pond
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        if (i >= nSpots) break;
        vec3 Lv = spotPos[i] - vWorld; float d = length(Lv); vec3 Ld = Lv / d;
        float cone = smoothstep(spotCone[i].x, spotCone[i].y, dot(-Ld, spotDir[i]));
        acc += spotCol[i] * pow(max(dot(reflect(-Ld, wn), Vv), 0.0), 70.0) * cone / (1.0 + 0.004*d*d);
    }
    return acc * 0.5;
}
float shadowCalc(vec3 n, vec3 L) {
    vec3 p = vLight.xyz / vLight.w * 0.5 + 0.5;
    if (p.z > 1.0 || p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 0.0;
    float bias = 0.00025 + 0.0006 * (1.0 - dot(n, L));
    vec2 ts = 1.0 / vec2(textureSize(shadowMap, 0));
    float s = 0.0;
    for (int x = -1; x <= 1; x++) for (int y = -1; y <= 1; y++) {
        float d = texture(shadowMap, p.xy + vec2(x, y) * ts).r;
        s += (p.z - bias > d) ? 1.0 : 0.0;
    }
    return s / 9.0;
}
float waterH(vec2 p) {                       // flowing pond surface
    vec2 q = p + vec2(0.6, 0.25) * time * 0.8;
    float h = sin(q.x*1.3 + time*1.1)*0.5 + sin(q.y*1.7 - time*0.9)*0.4 + sin((q.x + q.y)*2.3 + time*1.7)*0.25
            + fbm(q*1.5 + vec2(time*0.25, time*0.1))*1.2;
    return h * 0.08;
}
void main() {
    vec3 n = normalize(vNormal), L = normalize(lightDir), Vv = normalize(camPos - vWorld);
    vec3 sky = skyColor;
    vec2 uv = (abs(n.y) > 0.7) ? vWorld.xz : ((abs(n.x) > abs(n.z)) ? vWorld.zy : vWorld.xy);
    vec3 base = color; float spec = 0.0;
    float diff = max(dot(n, L), 0.0);
    float sh = diff > 0.0 ? shadowCalc(n, L) : 0.0;
    vec3 emitAdd = vec3(0.0);
    vec3 col;
    if (matId == 4) {
        vec2 p = vWorld.xz; float e = 0.12;
        float h0 = waterH(p), hx = waterH(p + vec2(e, 0.0)), hz = waterH(p + vec2(0.0, e));
        vec3 wn = normalize(vec3(-(hx - h0)/e*5.0, 1.0, -(hz - h0)/e*5.0));
        float fres = pow(1.0 - max(dot(wn, Vv), 0.0), 3.0);
        vec3 deep = vec3(0.03, 0.20, 0.22), shallow = vec3(0.10, 0.36, 0.30);
        vec3 wc = mix(deep, shallow, fbm(p*0.35 + vec2(0.6, 0.25)*time*0.05));
        col = mix(wc, sky, 0.18 + 0.6*fres);
        vec3 R = reflect(-L, wn);
        col += vec3(1.0, 0.95, 0.8) * pow(max(dot(R, Vv), 0.0), 90.0) * 1.5 * (1.0 - sh);
        col += smoothstep(0.65, 0.95, fbm(p*1.6 + vec2(0.7, 0.28)*time*0.8)) * 0.07;
        col *= 1.0 - 0.45*sh;
        col *= mix(1.0, 0.4, night);
        if (night > 0.5) col += vec3(0.03, 0.08, 0.10) + spotLight(wn) * vec3(0.3, 0.55, 0.6) + spotSpec(wn, Vv);
    } else {
        if (matId == 1) { float g = 0.6*fbm(uv*0.5) + 0.3*noise(uv*11.0) + 0.15*noise(uv*37.0); base *= 0.62 + 0.75*g; base *= 1.0 + 0.05*sin(time*1.4 + uv.x*0.35 + uv.y*0.25); }
        else if (matId == 2) { base *= 0.86 + 0.24*fbm(uv*1.3) + 0.08*noise(uv*28.0) - 0.04; }
        else if (matId == 3) { base *= 0.9 + 0.12*noise(uv*5.0) + 0.06*noise(uv*35.0) + 0.05*fbm(uv*0.4); }
        else if (matId == 5) { base *= 0.55 + 0.8*noise(vec2(uv.x*16.0, uv.y*1.4)); }
        else if (matId == 6) { float l = noise(vWorld.xy*3.0 + vWorld.z*2.0) + 0.5*noise(vWorld.xz*7.0 + vWorld.y*5.0); base *= 0.62 + 0.5*l; base *= 1.0 + 0.1*sin(time*3.0 + vWorld.x*2.0 + vWorld.z*3.0); }
        else if (matId == 7) { spec = 0.7; base *= 0.92 + 0.1*noise(vec2(uv.x*30.0, uv.y*2.0)); }
        else if (matId == 8) { float fr = pow(1.0 - max(dot(n, Vv), 0.0), 2.0); base = mix(base, vec3(0.62, 0.80, 0.95), 0.25 + 0.45*fr); spec = 1.0;
            float lit = step(0.5, hash(floor(vec2(vWorld.x + vWorld.z, vWorld.y) / vec2(3.2, 4.0)))); emitAdd = vec3(1.0, 0.82, 0.5) * lit * night * 0.9; }
        else if (matId == 11) { base *= 0.9 + 0.15*noise(uv*40.0); }
        else if (matId == 9) { float st = step(0.5, fract(vWorld.x/5.0)); base *= 0.84 + 0.22*st; float g = 0.6*fbm(uv*0.7) + 0.4*noise(uv*14.0); base *= 0.75 + 0.5*g; }
        if (matId == 10) col = base;                       // glowing (lamps, switched-on lights)
        else {
            float skyl = 0.5 + 0.5*n.y;
            vec3 ambC = mix(vec3(1.0), vec3(0.55, 0.65, 1.0), night), sunC = mix(vec3(1.0), vec3(0.6, 0.7, 1.0), night);
            float ambI = mix(0.36 + 0.26*skyl, 0.14 + 0.08*skyl, night), sunI = mix(0.72, 0.30, night);
            col = base * (ambI*ambC + sunI*diff*(1.0 - sh)*sunC);
            if (night > 0.5) col += base * (lampLight(n) + floodLight(n) + spotLight(n));
            if (roomOn > 0.5) col += base * roomLight(n);
            col += vec3(1.0, 0.96, 0.88) * spec * pow(max(dot(n, normalize(L + Vv)), 0.0), 40.0) * diff * (1.0 - sh);
        }
    }
    col += emitAdd;
    float d = length(camPos - vWorld);
    float fogk = clamp((d - 170.0)/280.0, 0.0, 1.0); if (matId == 10) fogk = 0.0;
    col = mix(col, sky, fogk);
    FragColor = vec4(col, 1.0);
}
)";

const string DVS_SRC = string(R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 model, lightVP;
)") + SWAY_GLSL + R"(
void main() { vec4 w = model * vec4(aPos, 1.0); w.xyz = sway(w.xyz, aPos); gl_Position = lightVP * w; }
)";
const string DFS_SRC = "#version 330 core\nvoid main() {}\n";

GLuint compileShader(GLenum type, const string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[1024]; glGetShaderInfoLog(s, 1024, nullptr, log); printf("Shader error:\n%s\n", log); }
    return s;
}
GLuint createProgram(const string& vsrc, const string& fsrc) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc), fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

// shadow map
const int SHADOW_RES = 4096;
GLuint shadowFBO = 0, shadowTex = 0;
void initShadow() {
    glGenTextures(1, &shadowTex); glBindTexture(GL_TEXTURE_2D, shadowTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_RES, SHADOW_RES, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float b[4] = {1, 1, 1, 1}; glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, b);
    glGenFramebuffers(1, &shadowFBO); glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowTex, 0);
    glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ============================================================================
//  SEGMENT 4 : MESHES (one cube + one sphere, reused for EVERYTHING)
// ============================================================================
struct Mesh { GLuint vao = 0, vbo = 0, ebo = 0; GLsizei count = 0; };
Mesh makeMesh(const vector<float>& verts, const vector<unsigned>& idx) {
    Mesh m; m.count = (GLsizei)idx.size();
    glGenVertexArrays(1, &m.vao); glGenBuffers(1, &m.vbo); glGenBuffers(1, &m.ebo);
    glBindVertexArray(m.vao);
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned), idx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
    return m;
}
Mesh buildCube() {
    float n[6][3] = {{0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}};
    float u[6][3] = {{1,0,0},{-1,0,0},{0,0,-1},{0,0,1},{1,0,0},{1,0,0}};
    float v[6][3] = {{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,0,-1},{0,0,1}};
    float ab[4][2] = {{-1,-1},{1,-1},{1,1},{-1,1}};
    vector<float> verts; vector<unsigned> idx;
    for (int f = 0; f < 6; f++) {
        unsigned base = (unsigned)(verts.size() / 6);
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < 3; i++) verts.push_back(n[f][i]*0.5f + u[f][i]*ab[k][0]*0.5f + v[f][i]*ab[k][1]*0.5f);
            for (int i = 0; i < 3; i++) verts.push_back(n[f][i]);
        }
        unsigned t[6] = {0,1,2,0,2,3};
        for (int i = 0; i < 6; i++) idx.push_back(base + t[i]);
    }
    return makeMesh(verts, idx);
}
Mesh buildSphere(int stacks = 24, int sectors = 32) {
    vector<float> verts; vector<unsigned> idx;
    for (int i = 0; i <= stacks; i++) {
        float phi = PI * i / stacks;
        for (int j = 0; j <= sectors; j++) {
            float th = 2 * PI * j / sectors;
            float x = sinf(phi)*cosf(th), y = cosf(phi), z = sinf(phi)*sinf(th);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        }
    }
    for (int i = 0; i < stacks; i++) for (int j = 0; j < sectors; j++) {
        unsigned a = i*(sectors + 1) + j, b = a + sectors + 1;
        idx.push_back(a); idx.push_back(b); idx.push_back(a + 1);
        idx.push_back(a + 1); idx.push_back(b); idx.push_back(b + 1);
    }
    return makeMesh(verts, idx);
}

// Unit cylinder: radius 1, length 1 along Z (wheels, poles, tanks, trunks)
Mesh buildCylinder(int seg = 28) {
    vector<float> verts; vector<unsigned> idx;
    for (int i = 0; i <= seg; i++) {
        float a = 2 * PI * i / seg, c = cosf(a), sn = sinf(a);
        verts.insert(verts.end(), {c, sn, -0.5f, c, sn, 0.0f}); verts.insert(verts.end(), {c, sn, 0.5f, c, sn, 0.0f});
    }
    for (int i = 0; i < seg; i++) { unsigned a = i * 2; idx.push_back(a); idx.push_back(a + 1); idx.push_back(a + 2); idx.push_back(a + 1); idx.push_back(a + 3); idx.push_back(a + 2); }
    for (int side = 0; side < 2; side++) {
        float z = side ? 0.5f : -0.5f, nz = side ? 1.0f : -1.0f; unsigned base = (unsigned)(verts.size() / 6);
        verts.insert(verts.end(), {0.0f, 0.0f, z, 0.0f, 0.0f, nz});
        for (int i = 0; i <= seg; i++) { float a = 2 * PI * i / seg; verts.insert(verts.end(), {cosf(a), sinf(a), z, 0.0f, 0.0f, nz}); }
        for (int i = 0; i < seg; i++) { idx.push_back(base); idx.push_back(base + 1 + i); idx.push_back(base + 2 + i); }
    }
    return makeMesh(verts, idx);
}

// ============================================================================
//  SEGMENT 5 : DRAW HELPERS  (box / ball / beam draw everything in the scene)
//  Material ids: 0 flat | 1 grass | 2 concrete | 3 plaster | 4 water | 5 bark
//                6 leaves | 7 metal | 8 glass | 9 football turf | 10 glowing
// ============================================================================
GLuint prog, depthProg;
GLint uModel, uView, uProj, uColor, uLight, uCam, uLightVP, uTime, uMat, uRoomOn, uRoomPos, uRoomBox, uGridInfo, uGridDim, uNFlood, uFloodPos, uFloodVP, uNSpots, uSpotPos, uSpotDir, uSpotCol, uSpotCone, uNight, uSky;
GLint dModel, dLightVP, dTime, dMat;
GLint gModel, gColor, gMat;           // uniforms of the program used by the CURRENT pass
bool shadowPass = false; int curMat = -1; float gTime = 0;
Mesh cubeMesh, sphereMesh, cylMesh;

bool lampPass = false; Vec3 cullC = {0, 0, 0}; float cullR = 0;      // used while baking lamp-post shadow maps
void setMat(int m) { if (m != curMat) { curMat = m; glUniform1i(gMat, m); } }
void drawMesh(const Mesh& m, const Mat4& model, Vec3 c, float ext) {
    if (lampPass) {                                         // lamp passes: no glowing parts, no huge slabs, only nearby things
        if (curMat == 10 || ext > 60) return;
        float dx = model.m[12] - cullC.x, dy = model.m[13] - cullC.y, dz = model.m[14] - cullC.z;
        if (sqrtf(dx*dx + dy*dy + dz*dz) - ext > cullR + 4) return;
    }
    glUniformMatrix4fv(gModel, 1, GL_FALSE, model.m);
    if (!shadowPass) glUniform3f(gColor, c.x, c.y, c.z);
    glBindVertexArray(m.vao);
    glDrawElements(GL_TRIANGLES, m.count, GL_UNSIGNED_INT, 0);
}
void box(const Mat4& parent, Vec3 p, Vec3 s, Vec3 c, int m = 0) {
    if (lampPass && (m == 2 || m == 9) && s.y < 0.5f) return;
    setMat(m); drawMesh(cubeMesh, parent * translateM(p.x, p.y, p.z) * scaleM(s.x, s.y, s.z), c, 0.5f * sqrtf(s.x*s.x + s.y*s.y + s.z*s.z));
}
void ball(const Mat4& parent, Vec3 p, Vec3 s, Vec3 c, int m = 0) {
    setMat(m); drawMesh(sphereMesh, parent * translateM(p.x, p.y, p.z) * scaleM(s.x, s.y, s.z), c, max(s.x, max(s.y, s.z)));
}
// cylinder along local Z: s = (radiusX, radiusY, length)
void cyl(const Mat4& parent, Vec3 p, Vec3 s, Vec3 c, int m = 0) {
    setMat(m); drawMesh(cylMesh, parent * translateM(p.x, p.y, p.z) * scaleM(s.x, s.y, s.z), c, sqrtf(s.x*s.x + s.y*s.y + 0.25f*s.z*s.z));
}
// a bar between two points in the local XY plane (used for frames, limbs, spokes)
void beam(const Mat4& P, float x1, float y1, float x2, float y2, float z, float th, float dep, Vec3 c, int m = 0) {
    float dx = x2 - x1, dy = y2 - y1, len = sqrtf(dx*dx + dy*dy);
    box(P * translateM((x1 + x2)/2, (y1 + y2)/2, z) * rotateZ(atan2f(dy, dx)), V(0,0,0), V(len, th, dep), c, m);
}

// ---- colour palette ----
const Vec3 GRASS = {0.30f,0.62f,0.25f}, HILL = {0.24f,0.55f,0.22f}, FIELDC = {0.28f,0.62f,0.24f};
const Vec3 CONCRETE = {0.66f,0.66f,0.64f}, JOINT = {0.46f,0.46f,0.45f}, WHITE = {0.96f,0.96f,0.96f};
const Vec3 PINK = {0.86f,0.63f,0.56f}, CREAM = {0.93f,0.90f,0.84f}, GLASS = {0.18f,0.30f,0.42f};
const Vec3 BOARD = {0.05f,0.36f,0.22f}, DARK = {0.10f,0.10f,0.11f}, STEEL = {0.72f,0.74f,0.78f};
const Vec3 WOOD = {0.50f,0.34f,0.18f}, YELLOW = {0.95f,0.75f,0.10f};

// ============================================================================
//  SEGMENT 6 : BLOCK FONT  (5x7 pixel letters, built from boxes)
// ============================================================================
struct Glyph { char ch; const char* rows[7]; };
const Glyph FONT[] = {
 {'A',{".###.","#...#","#...#","#####","#...#","#...#","#...#"}},
 {'B',{"####.","#...#","#...#","####.","#...#","#...#","####."}},
 {'C',{".####","#....","#....","#....","#....","#....",".####"}},
 {'D',{"####.","#...#","#...#","#...#","#...#","#...#","####."}},
 {'E',{"#####","#....","#....","####.","#....","#....","#####"}},
 {'F',{"#####","#....","#....","####.","#....","#....","#...."}},
 {'G',{".####","#....","#....","#..##","#...#","#...#",".####"}},
 {'H',{"#...#","#...#","#...#","#####","#...#","#...#","#...#"}},
 {'I',{"#####","..#..","..#..","..#..","..#..","..#..","#####"}},
 {'J',{"..###","...#.","...#.","...#.","...#.","#..#.",".##.."}},
 {'K',{"#...#","#..#.","#.#..","##...","#.#..","#..#.","#...#"}},
 {'L',{"#....","#....","#....","#....","#....","#....","#####"}},
 {'M',{"#...#","##.##","#.#.#","#.#.#","#...#","#...#","#...#"}},
 {'N',{"#...#","##..#","#.#.#","#..##","#...#","#...#","#...#"}},
 {'O',{".###.","#...#","#...#","#...#","#...#","#...#",".###."}},
 {'P',{"####.","#...#","#...#","####.","#....","#....","#...."}},
 {'Q',{".###.","#...#","#...#","#...#","#.#.#","#..#.",".##.#"}},
 {'R',{"####.","#...#","#...#","####.","#.#..","#..#.","#...#"}},
 {'S',{".####","#....","#....",".###.","....#","....#","####."}},
 {'T',{"#####","..#..","..#..","..#..","..#..","..#..","..#.."}},
 {'U',{"#...#","#...#","#...#","#...#","#...#","#...#",".###."}},
 {'V',{"#...#","#...#","#...#","#...#","#...#",".#.#.","..#.."}},
 {'W',{"#...#","#...#","#...#","#.#.#","#.#.#","##.##","#...#"}},
 {'Y',{"#...#","#...#",".#.#.","..#..","..#..","..#..","..#.."}},
 {'Z',{"#####","....#","...#.","..#..",".#...","#....","#####"}},
};
const char* const* getGlyph(char c) { for (const Glyph& g : FONT) if (g.ch == c) return g.rows; return nullptr; }
float textWidth(const string& t, float px) { return (t.size() * 6 - 1) * px; }

// origin = bottom-left of first letter. Text runs along +x, up is +y. (small text is skipped in the shadow pass)
void drawText(const Mat4& parent, const string& t, Vec3 origin, float px, float depth, Vec3 col, int m = 0) {
    if (shadowPass && px < 0.3f) return;
    float x0 = origin.x;
    for (char ch : t) {
        const char* const* g = getGlyph(ch);
        if (g) for (int row = 0; row < 7; row++) {
            int c = 0;
            while (c < 5) {
                if (g[row][c] == '#') {
                    int e = c; while (e < 5 && g[row][e] == '#') e++;
                    float w = (e - c) * px;
                    box(parent, V(x0 + c*px + w/2, origin.y + (6 - row)*px + px/2, origin.z), V(w, px, depth), col, m);
                    c = e;
                } else c++;
            }
        }
        x0 += 6 * px;
    }
}
void drawTextCentered(const Mat4& parent, const string& t, float cx, float y, float z, float px, float depth, Vec3 col, int m = 0) {
    drawText(parent, t, V(cx - textWidth(t, px)/2, y, z), px, depth, col, m);
}

// ---- switchable machine groups: w ramps smoothly 0..1, t only advances while running ----
struct Rig { bool on = false; float w = 0, t = 0;
    void update(float dt) { w += ((on ? 1.f : 0.f) - w) * min(1.f, dt * 1.2f); t += w * dt; } };
Rig rigFan, rigEEE, rigMech, rigCivil;
bool lightsOn = true, nightMode = false;

// ============================================================================
//  SEGMENT 7 : GROUND + POND + WALLS
// ============================================================================
const float PX0 = -82, PX1 = -50, PZ0 = -24, PZ1 = 6;      // pond rectangle
void groundSlab(float x0, float x1, float z0, float z1) {
    box(I, V((x0 + x1)/2, -1.5f, (z0 + z1)/2), V(x1 - x0, 3, z1 - z0), GRASS, 1);
}
void drawGround() {                       // the ground has a hole where the pond is
    groundSlab(-260, 260, PZ1, 260); groundSlab(-260, 260, -260, PZ0);
    groundSlab(-260, PX0, PZ0, PZ1); groundSlab(PX1, 260, PZ0, PZ1);
    Vec3 STONE = V(0.48f,0.45f,0.40f); float cx = (PX0 + PX1)/2, cz = (PZ0 + PZ1)/2, w = PX1 - PX0, d = PZ1 - PZ0;
    box(I, V(cx, -2.55f, cz), V(w, 0.2f, d), V(0.25f,0.20f,0.14f), 2);                 // mud bottom
    box(I, V(PX0 + 0.15f, -1.3f, cz), V(0.3f, 2.6f, d), STONE, 2);                      // stone lining
    box(I, V(PX1 - 0.15f, -1.3f, cz), V(0.3f, 2.6f, d), STONE, 2);
    box(I, V(cx, -1.3f, PZ0 + 0.15f), V(w, 2.6f, 0.3f), STONE, 2);
    box(I, V(cx, -1.3f, PZ1 - 0.15f), V(w, 2.6f, 0.3f), STONE, 2);
    box(I, V(cx, -0.5f, cz), V(w - 0.6f, 0.1f, d - 0.6f), V(0.1f,0.3f,0.3f), 4);        // flowing water
    box(I, V(cx, 0.2f, PZ1 + 0.5f), V(w + 2, 0.4f, 1), CONCRETE, 2);                   // curb around the pond
    box(I, V(cx, 0.2f, PZ0 - 0.5f), V(w + 2, 0.4f, 1), CONCRETE, 2);
    box(I, V(PX0 - 0.5f, 0.2f, cz), V(1, 0.4f, d), CONCRETE, 2);
    box(I, V(PX1 + 0.5f, 0.2f, cz), V(1, 0.4f, d), CONCRETE, 2);
}
void wallLine(float x0, float z0, float x1, float z1) {
    float dx = fabsf(x1 - x0), dz = fabsf(z1 - z0);
    box(I, V((x0 + x1)/2, 1.3f, (z0 + z1)/2), V(dx + 0.7f, 2.6f, dz + 0.7f), V(0.78f,0.76f,0.70f), 2);
    int n = (int)((dx + dz) / 14);
    for (int i = 0; i <= n; i++) { float t = n ? (float)i / n : 0;
        box(I, V(x0 + (x1 - x0)*t, 1.6f, z0 + (z1 - z0)*t), V(1.3f, 3.2f, 1.3f), V(0.62f,0.60f,0.55f), 2); }
}
void drawWalls() {
    wallLine(-95, 100, -24, 100); wallLine(30, 100, 85, 100);
    wallLine(-95, 100, -95, -152); wallLine(85, 100, 85, -152); wallLine(-95, -152, 85, -152);
}

// ============================================================================
//  SEGMENT 8 : CONCRETE ROADS (list used for drawing AND for planting trees)
// ============================================================================
struct Road { bool alongZ; float c, a0, a1, w, top; bool mark; };
vector<Road> roads;
void addRoad(bool alongZ, float c, float a0, float a1, float w, bool mark, float top) { roads.push_back({alongZ, c, a0, a1, w, top, mark}); }
void initRoads() {
    addRoad(true,   0,   30, 140, 10, true, 0.100f);      // MAIN ROAD from outside the gate to the plaza road
    addRoad(false, 30, -38, 38, 8, true, 0.092f);          // road in front of the mosque plaza
    addRoad(false, 88, -38, 38, 8, true, 0.092f);
    addRoad(false, -48, -38, 38, 8, true, 0.092f);         // road behind the mosque
    addRoad(true, -38, -140, 88, 8, true, 0.096f);         // LEFT road (hill, halls)
    addRoad(true,  38, -150, 88, 8, true, 0.096f);         // RIGHT road (academic buildings, Rokeya Hall)
    addRoad(false, -118, 44, 60, 4, false, 0.085f);        // path into Rokeya Hall
    addRoad(true,   0,  -48, -23, 6, false, 0.094f);       // behind mosque
    float hz[6] = {22, -36, -60, -84, -108, -132};
    for (int i = 0; i < 6; i++) addRoad(false, hz[i], -56, -42, 4, false, 0.085f);   // paths to the halls
    addRoad(false, 72, 42, 65, 5, false, 0.085f);          // path to CSE
    addRoad(false,  2, 42, 65, 5, false, 0.085f);          // path to the 5-storey building
}
void roadZ(float x, float width, float z0, float z1, bool markings, float top) {
    box(I, V(x, top/2, (z0 + z1)/2), V(width, top, z1 - z0), CONCRETE, 2);
    for (float z = z0 + 6; z < z1; z += 6) box(I, V(x, top + 0.005f, z), V(width, 0.02f, 0.12f), JOINT);
    if (markings) for (float z = z0 + 2; z < z1 - 2; z += 5) box(I, V(x, top + 0.01f, z), V(0.22f, 0.02f, 2.2f), WHITE);
}
void roadX(float z, float x0, float x1, float width, bool markings, float top) {
    box(I, V((x0 + x1)/2, top/2, z), V(x1 - x0, top, width), CONCRETE, 2);
    for (float x = x0 + 6; x < x1; x += 6) box(I, V(x, top + 0.005f, z), V(0.12f, 0.02f, width), JOINT);
    if (markings) for (float x = x0 + 2; x < x1 - 2; x += 5) box(I, V(x, top + 0.01f, z), V(2.2f, 0.02f, 0.22f), WHITE);
}
void drawRoads() {
    for (const Road& r : roads) { if (r.alongZ) roadZ(r.c, r.w, r.a0, r.a1, r.mark, r.top); else roadX(r.c, r.a0, r.a1, r.w, r.mark, r.top); }
    box(I, V(0, 0.06f, 14), V(52, 0.12f, 24), CONCRETE, 2);            // plaza in front of the mosque (z 2..26)
    for (float x = -24; x <= 24; x += 4) box(I, V(x, 0.125f, 14), V(0.1f, 0.02f, 24), JOINT);
    for (float z = 6; z < 26; z += 4) box(I, V(0, 0.125f, z), V(52, 0.02f, 0.1f), JOINT);
}

// ============================================================================
//  SEGMENT 9 : TREES + LAMP POSTS (one tree, one lamp, one tree ... along streets)
// ============================================================================
void drawTree(float x, float z) {
    float h = frac(sinf(x*12.9898f + z*78.233f) * 43758.5453f), s = 0.85f + 0.35f*h;
    Vec3 l1 = V(0.12f + 0.06f*h, 0.42f + 0.1f*h, 0.14f), l2 = V(0.18f, 0.52f + 0.08f*h, 0.18f);
    cyl(I * translateM(x, 1.3f*s, z) * rotateX(PI/2), V(0,0,0), V(0.3f, 0.3f, 2.6f*s), V(0.40f,0.26f,0.14f), 5);
    ball(I, V(x, 4.0f*s, z), V(2.4f*s, 2.2f*s, 2.4f*s), l1, 6);
    ball(I, V(x + 0.9f*s, 5.0f*s, z + 0.3f*s), V(1.7f*s, 1.6f*s, 1.7f*s), l2, 6);
    ball(I, V(x - 0.8f*s, 4.6f*s, z - 0.6f*s), V(1.6f*s, 1.5f*s, 1.6f*s), l1, 6);
}
void drawLamp(float x, float z, float ax, float az) {
    Vec3 G = V(0.25f,0.27f,0.30f);
    box(I, V(x, 0.25f, z), V(0.5f, 0.5f, 0.5f), G, 7);
    cyl(I * translateM(x, 3.6f, z) * rotateX(PI/2), V(0,0,0), V(0.1f, 0.1f, 7.2f), G, 7);
    if (fabsf(ax) > 0.1f) box(I, V(x + ax*0.9f, 7.3f, z), V(1.8f, 0.14f, 0.14f), G, 7);
    else                  box(I, V(x, 7.3f, z + az*0.9f), V(0.14f, 0.14f, 1.8f), G, 7);
    box(I, V(x + ax*1.7f, 7.15f, z + az*1.7f), V(0.8f, 0.2f, 0.8f), nightMode ? V(1.0f,0.92f,0.65f) : V(0.75f,0.75f,0.7f), 10);   // shines at night
}
struct Item { float x, z; bool tree; float ax, az; };
struct Rect { float x0, x1, z0, z1; };
const Vec3 HILL_SPOTS[4] = {{-46, 0.3f, 56}, {-46, 0.3f, 72}, {-58, 0.3f, 83}, {-70, 0.3f, 83}};
const float POND_POSTS[4][2] = {{PX0 - 2.5f, PZ1 + 2.5f}, {PX1 + 2.5f, PZ1 + 2.5f}, {PX0 - 2.5f, PZ0 - 2.5f}, {PX1 + 2.5f, PZ0 - 2.5f}};
struct BenchSpot { float x, z, rot, y; int kind; };           // kind 0 = couple, 1 = two boys, 2 = empty
vector<Item> scenery; vector<Rect> noPlant; vector<BenchSpot> benches;
bool blocked(float x, float z) {
    for (const Road& r : roads) {
        float m = r.w/2 + 2.2f;
        if (r.alongZ  && fabsf(x - r.c) < m && z > r.a0 - 2 && z < r.a1 + 2) return true;
        if (!r.alongZ && fabsf(z - r.c) < m && x > r.a0 - 2 && x < r.a1 + 2) return true;
    }
    for (const Rect& q : noPlant) if (x > q.x0 && x < q.x1 && z > q.z0 && z < q.z1) return true;
    return false;
}
// trees and lamps alternate along a straight line
void avenue(float x0, float z0, float x1, float z1, float step, bool treeFirst, float ax, float az) {
    float L = sqrtf((x1 - x0)*(x1 - x0) + (z1 - z0)*(z1 - z0)); int n = (int)(L / step);
    for (int i = 0; i <= n; i++) {
        float t = n ? (float)i / n : 0, x = x0 + (x1 - x0)*t, z = z0 + (z1 - z0)*t;
        bool tree = ((i % 2) == 0) == treeFirst;
        if (blocked(x, z)) continue;
        scenery.push_back({x, z, tree, ax, az});
    }
}
void initScenery() {
    noPlant.push_back({-28, 28, 0, 27});      noPlant.push_back({-88, -44, -30, 12});
    noPlant.push_back({-82, -46, 46, 82});    noPlant.push_back({-34, 34, -116, -58});
    noPlant.push_back({-20, 20, -27, 0});     noPlant.push_back({-26, 30, 96, 104});
    noPlant.push_back({42, 66, -92, -18});
    avenue(-7.5f, 36, -7.5f, 94, 8, true, 1, 0);   avenue(7.5f, 36, 7.5f, 94, 8, false, -1, 0);       // main road
    avenue(-43.5f, 84, -43.5f, -136, 8, true, 1, 0); avenue(-32.5f, 84, -32.5f, -44, 8, false, -1, 0); // left road
    avenue(32.5f, 84, 32.5f, -90, 8, false, 1, 0);   avenue(43.5f, 86, 43.5f, 50, 8, true, -1, 0);    // right road
    avenue(-34, 37, 34, 37, 8, true, 0, -1);                                                         // plaza road
    avenue(-34, 95, 34, 95, 8, true, 0, -1);   avenue(-34, 81, 34, 81, 8, false, 0, 1);
    avenue(-34, -41, 34, -41, 8, true, 0, -1); avenue(-34, -55, 34, -55, 8, false, 0, 1);
    avenue(-30, -118, 30, -118, 8, true, 0, 1);                                                      // behind the field
    for (int i = 0; i < 7; i++) scenery.push_back({-88.f + (i % 2)*0.f, -30.f + i*5.f, true, 0, 0}); // trees west of pond
    for (int i = 0; i < 4; i++) { scenery.push_back({-27.f, 4.f + i*7.f, false, 1, 0}); scenery.push_back({27.f, 4.f + i*7.f, false, -1, 0}); }
    float hallZ[6] = {22, -36, -60, -84, -108, -132};            // 2 lamps in front of every hall
    for (int i = 0; i < 6; i++) { scenery.push_back({-51.5f, hallZ[i] - 8.0f, false, -1, 0}); scenery.push_back({-51.5f, hallZ[i] + 8.0f, false, -1, 0}); }
    float deptZ[5] = {72, 36, 2, -34, -70};                      // 2 lamps in front of every department building
    for (int i = 0; i < 5; i++) { scenery.push_back({60.0f, deptZ[i] - 14.5f, false, 1, 0}); scenery.push_back({60.0f, deptZ[i] + 14.5f, false, 1, 0}); }
    scenery.push_back({50, -103, false, 1, 0}); scenery.push_back({50, -134, false, 1, 0});     // inside Rokeya Hall compound
    scenery.push_back({60, -101, false, 0, 1}); scenery.push_back({60, -143, false, 0, -1});
    scenery.push_back({77, -104, true, 0, 0}); scenery.push_back({77, -132, true, 0, 0}); scenery.push_back({77, -142, true, 0, 0});
    benches = {{-18, 24.5f, PI, 0.12f, 0}, {18, 24.5f, PI, 0.12f, 0}, {-86.5f, -9, PI/2, 0, 0}, {-47, -9, -PI/2, 0, 0},
               {-46, 64, -PI/2, 0, 0}, {-34, -80, PI/2, 0, 0}, {34, -80, -PI/2, 0, 1}};
}
void drawBench(const Mat4& P) {
    box(P, V(0, 0.55f, 0), V(2.2f, 0.12f, 0.6f), WOOD, 5);
    box(P, V(0, 0.95f, -0.25f), V(2.2f, 0.5f, 0.1f), WOOD, 5);
    box(P, V(-0.9f, 0.27f, 0), V(0.12f, 0.55f, 0.5f), DARK); box(P, V(0.9f, 0.27f, 0), V(0.12f, 0.55f, 0.5f), DARK);
}
void drawTreesAndLamps() {
    for (const Item& it : scenery) { if (it.tree) drawTree(it.x, it.z); else drawLamp(it.x, it.z, it.ax, it.az); }
    for (const BenchSpot& b : benches) drawBench(frame(b.x, b.z, b.rot, b.y));
    for (int k = 0; k < 4; k++) {                      // light posts around the pond
        float px = POND_POSTS[k][0], pz = POND_POSTS[k][1], dx = ((PX0 + PX1)/2 - px), dz = ((PZ0 + PZ1)/2 - pz), l = sqrtf(dx*dx + dz*dz);
        cyl(I * translateM(px, 3.1f, pz) * rotateX(PI/2), V(0,0,0), V(0.12f, 0.12f, 6.2f), V(0.25f,0.27f,0.3f), 7);
        box(I, V(px + dx/l*0.5f, 6.3f, pz + dz/l*0.5f), V(0.8f, 0.25f, 0.8f), nightMode ? V(0.85f,0.95f,1.0f) : V(0.7f,0.7f,0.72f), 10);
    }
}

// ============================================================================
//  SEGMENT 10 : MAIN GATE (modelled on the photo: low sloped slab + tall leaning spire)
// ============================================================================
void drawGate() {
    Mat4 G = translateM(0, 0, 100);
    Vec3 SL = V(0.62f,0.64f,0.68f), SC = V(0.80f,0.82f,0.86f), SIGN = V(0.08f,0.22f,0.45f);
    // ---- low slab block on the left (only this one has the KUET sign)
    box(G, V(-15, 1.1f, 0), V(14, 2.2f, 5), SL, 7);
    box(G * translateM(-15, 2.35f, 0) * rotateZ(-0.1f), V(0,0,0), V(14.6f, 0.45f, 5.6f), SC, 7);
    box(G, V(-15, 1.2f, 2.55f), V(10, 1.6f, 0.12f), SIGN);
    drawTextCentered(G, "KUET", -15, 0.45f, 2.64f, 0.22f, 0.1f, WHITE);
    for (int i = 0; i < 4; i++) box(G, V(-19.5f + i*3, 1.1f, -2.55f), V(1.6f, 0.9f, 0.1f), DARK);
    // ---- tall leaning spire on the right
    float ax = 26, ay = 40, y0 = 1.2f;
    box(G, V(17, 0.6f, 0), V(24, 1.2f, 5), SL, 2);
    beam(G, 6, y0, ax, ay, 0, 0.6f, 3.6f, SC, 7);                 // solid inclined plane
    auto xl = [&](float y) { return 6 + (ax - 6) * (y - y0) / (ay - y0); };
    auto xr = [&](float y) { return 28 + (ax - 28) * (y - y0) / (ay - y0); };
    for (int s = -1; s <= 1; s += 2) {
        float z = 1.7f * s;
        beam(G, 28, y0, ax, ay, z, 0.45f, 0.45f, SC, 7);
        for (int k = 0; k < 5; k++) {
            float ya = y0 + k*7.5f, yb = ya + 7.5f;
            beam(G, xl(ya), ya, xr(ya), ya, z, 0.3f, 0.3f, SC, 7);
            if (k % 2 == 0) beam(G, xl(ya), ya, xr(yb), yb, z, 0.25f, 0.25f, SC, 7);
            else            beam(G, xr(ya), ya, xl(yb), yb, z, 0.25f, 0.25f, SC, 7);
        }
    }
    for (int k = 0; k < 5; k++) { float y = y0 + k*7.5f;
        box(G, V(xr(y), y, 0), V(0.3f, 0.3f, 3.4f), SC, 7); }
    beam(G, ax, ay, ax + 2.6f, ay + 6.5f, 0, 0.35f, 0.5f, SC, 7);  // spike
}

// ============================================================================
//  SEGMENT 11 : THE HILL WITH "KUET" ON TOP
// ============================================================================
void drawHill() {
    Vec3 C = V(-64, 0, 64);
    ball(I, C, V(16, 9, 16), HILL, 1);
    const float RAD = 16, HGT = 9, PXs = 0.65f;
    const string word = "KUET"; float step = 6 * PXs;
    for (int i = 0; i < 4; i++) {
        float cxl = (i - 1.5f) * step, edge = fabsf(cxl) + 2.5f * PXs;
        float hillY = HGT * sqrtf(1 - (edge*edge) / (RAD*RAD));
        Mat4 P = translateM(C.x, hillY - 0.05f, C.z) * rotateY(0.6f);
        drawTextCentered(P, string(1, word[i]), cxl, 0, 0, PXs, 1.2f, nightMode ? V(1.0f,0.93f,0.7f) : WHITE, nightMode ? 10 : 3);   // letters glow at night
    }
    for (int k = 0; k < 4; k++) {                      // ground spot lights that light the hill
        box(I, HILL_SPOTS[k], V(0.8f, 0.6f, 0.8f), V(0.2f,0.2f,0.22f), 7);
        box(I, HILL_SPOTS[k] + V(0, 0.34f, 0), V(0.55f, 0.1f, 0.55f), nightMode ? V(1.0f,0.95f,0.8f) : V(0.6f,0.6f,0.55f), 10);
    }
}

// ============================================================================
//  SEGMENT 12 : BUILDINGS  (local frame: centred at origin, front faces +Z)
// ============================================================================
void drawBuilding(const Mat4& P, float w, float d, float h, int floors, const vector<string>& lines, float px, bool open = false) {
    float hf = h / floors, zf = d / 2;
    if (!open) box(P, V(0, h/2, 0), V(w, h, d), PINK, 3);
    else {                                                    // open ground floor (EEE)
        box(P, V(0, (h + hf)/2, 0), V(w, h - hf, d), PINK, 3);
        box(P, V(0, hf/2, -d/2 + 0.3f), V(w, hf, 0.6f), PINK, 3);
        box(P, V(-w/2 + 0.3f, hf/2, 0), V(0.6f, hf, d), PINK, 3);
        box(P, V( w/2 - 0.3f, hf/2, 0), V(0.6f, hf, d), PINK, 3);
        box(P, V(0, 0.12f, 0), V(w - 0.8f, 0.24f, d - 0.8f), CONCRETE, 2);
        int nc = (int)(w / 5.5f);
        for (int i = 0; i <= nc; i++) box(P, V(-w/2 + 0.7f + i*(w - 1.4f)/nc, hf/2, zf - 0.5f), V(0.9f, hf, 0.9f), CREAM, 3);
    }
    box(P, V(0, h + 0.2f, 0), V(w + 0.6f, 0.4f, d + 0.6f), CREAM, 3);
    box(P, V(w*0.3f, h + 1.4f, -d*0.2f), V(4, 2, 3), CREAM, 3);
    for (int f = 1; f < floors; f++) box(P, V(0, f*hf, zf + 0.6f), V(w + 0.4f, 0.3f, 1.4f), CREAM, 3);
    int n = (int)(w / 3.2f); float sp = w / n;
    for (int f = open ? 1 : 0; f < floors - 1; f++) for (int i = 0; i < n; i++)
        box(P, V(-w/2 + sp*(i + 0.5f), f*hf + hf*0.55f, zf + 0.05f), V(1.8f, 1.6f, 0.12f), GLASS, 8);
    if (!open) {
        box(P, V(0, 1.4f, zf + 0.12f), V(3.2f, 2.8f, 0.2f), DARK);
        box(P, V(0, 0.15f, zf + 1.3f), V(6, 0.3f, 2.6f), CREAM, 3);
    }
    float gap = px * 1.5f, maxW = 0;
    for (auto& l : lines) maxW = max(maxW, textWidth(l, px));
    float bw = maxW + 1.4f, bh = lines.size() * 7 * px + (lines.size() + 1) * gap;
    float yc = (floors - 1) * hf + hf / 2;
    box(P, V(0, yc, zf + 0.08f), V(bw, bh, 0.16f), BOARD);
    for (size_t i = 0; i < lines.size(); i++) {
        float top = yc + bh/2 - gap - i * (7 * px + gap);
        drawTextCentered(P, lines[i], 0, top - 7 * px, zf + 0.16f, px, 0.12f, WHITE);
    }
}

// EEE ground floor: ceiling lights, fans and the switch panel  (keys L and F)
void drawEEEUnder(const Mat4& P, float w, float d, float hf) {
    for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) {
        float x = -7.5f + i*5.0f, z = j == 0 ? 2.2f : -2.4f;
        if (lightsOn) box(P, V(x, hf - 0.12f, z), V(1.6f, 0.1f, 0.25f), V(1.0f,1.0f,0.92f), 10);
        else          box(P, V(x, hf - 0.12f, z), V(1.6f, 0.1f, 0.25f), V(0.6f,0.6f,0.58f), 0);
    }
    float fa = rigFan.t * 14.0f;
    for (int i = 0; i < 5; i++) {
        float x = -10 + i*5.0f;
        box(P, V(x, hf - 0.3f, 0), V(0.08f, 0.6f, 0.08f), DARK);
        ball(P, V(x, hf - 0.65f, 0), V(0.22f, 0.12f, 0.22f), DARK);
        for (int k = 0; k < 3; k++)
            box(P * translateM(x, hf - 0.7f, 0) * rotateY(fa + k*2.094f), V(0.65f, 0, 0), V(1.3f, 0.04f, 0.24f), V(0.85f,0.85f,0.8f));
    }
    float pz = -d/2 + 0.68f;
    box(P, V(-11, 1.8f, pz), V(2.4f, 1.4f, 0.1f), V(0.3f,0.3f,0.32f), 7);
    box(P, V(-11.6f, 2.15f, pz + 0.07f), V(0.3f, 0.3f, 0.06f), lightsOn ? V(0.2f,1.0f,0.3f) : V(0.9f,0.1f,0.1f), 10);
    box(P, V(-10.4f, 2.15f, pz + 0.07f), V(0.3f, 0.3f, 0.06f), rigFan.on ? V(0.2f,1.0f,0.3f) : V(0.9f,0.1f,0.1f), 10);
    drawText(P, "L", V(-11.75f, 1.45f, pz + 0.07f), 0.05f, 0.03f, WHITE);
    drawText(P, "F", V(-10.55f, 1.45f, pz + 0.07f), 0.05f, 0.03f, WHITE);
    for (int i = -1; i <= 1; i++) { box(P, V(i*6.0f, 0.8f, -2.8f), V(3, 0.9f, 0.9f), WOOD, 5); box(P, V(i*6.0f, 0.4f, -1.6f), V(1.2f, 0.5f, 0.6f), DARK); }
}

// ============================================================================
//  SEGMENT 13 : MOSQUE (straight ahead of the main gate)
// ============================================================================
void drawMosque() {
    Mat4 M = translateM(0, 0, -12);
    Vec3 WALL = V(0.94f,0.93f,0.88f), GREEN = V(0.16f,0.50f,0.42f), GOLD = V(0.85f,0.70f,0.25f), ARCH = V(0.10f,0.20f,0.18f);
    box(M, V(0, 0.35f, 2), V(38, 0.7f, 30), CONCRETE, 2);
    box(M, V(0, 0.2f, 18.5f), V(16, 0.4f, 3), CONCRETE, 2);
    box(M, V(0, 5.2f, 0), V(24, 9, 20), WALL, 3);
    box(M, V(0, 9.9f, 0), V(24.8f, 0.5f, 20.8f), WALL, 3);
    box(M, V(0, 7.4f, 12.2f), V(22, 0.5f, 4.4f), WALL, 3);                       // portico roof
    for (int i = 0; i < 5; i++) box(M, V(-10 + i*5.0f, 3.95f, 14), V(0.9f, 6.5f, 0.9f), WALL, 3);
    for (int i = -1; i <= 1; i++) { box(M, V(i*7.0f, 3.2f, 10.1f), V(3.6f, 5, 0.2f), ARCH); ball(M, V(i*7.0f, 5.7f, 10.1f), V(1.8f, 1.4f, 0.2f), ARCH); }
    box(M, V(0, 8.6f, 10.2f), V(19, 1.8f, 0.2f), BOARD);
    drawTextCentered(M, "KUET CENTRAL MOSQUE", 0, 8.0f, 10.35f, 0.16f, 0.1f, WHITE);
    box(M, V(0, 11.2f, 0), V(11, 2.6f, 11), WALL, 3);                            // main dome
    ball(M, V(0, 12.2f, 0), V(6.2f, 5.2f, 6.2f), GREEN, 7);
    box(M, V(0, 17.9f, 0), V(0.25f, 2.6f, 0.25f), GOLD, 7); ball(M, V(0, 19.4f, 0), V(0.5f, 0.5f, 0.5f), GOLD, 7);
    for (int sx = -1; sx <= 1; sx += 2) for (int sz = -1; sz <= 1; sz += 2) {     // small domes
        box(M, V(sx*8.0f, 10.9f, sz*6.0f), V(3.2f, 1.4f, 3.2f), WALL, 3);
        ball(M, V(sx*8.0f, 11.5f, sz*6.0f), V(1.8f, 1.6f, 1.8f), GREEN, 7);
    }
    for (int sx = -1; sx <= 1; sx += 2) {                                         // minarets
        float x = sx * 16.0f;
        box(M, V(x, 6.7f, 7), V(3.2f, 12, 3.2f), WALL, 3);
        box(M, V(x, 16.2f, 7), V(2.4f, 6.8f, 2.4f), WALL, 3);
        box(M, V(x, 19.8f, 7), V(3.4f, 0.5f, 3.4f), GREEN, 3);
        box(M, V(x, 21.5f, 7), V(1.8f, 3, 1.8f), WALL, 3);
        ball(M, V(x, 23.2f, 7), V(1.2f, 1.5f, 1.2f), GREEN, 7);
        box(M, V(x, 25, 7), V(0.15f, 2, 0.15f), GOLD, 7);
    }
    for (int sx = -1; sx <= 1; sx += 2) for (int i = 0; i < 3; i++) box(M, V(sx*12.05f, 4.5f, -6 + i*5.0f), V(0.15f, 3, 1.6f), GLASS, 8);
}

// ============================================================================
//  SEGMENT 14 : FOOTBALL FIELD (central field, behind the mosque)
// ============================================================================
const float FZ = -87;
void lineBox(float x0, float z0, float x1, float z1) {
    box(I, V((x0 + x1)/2, 0.07f, (z0 + z1)/2), V(fabsf(x1 - x0) + 0.25f, 0.02f, fabsf(z1 - z0) + 0.25f), WHITE);
}
void goal(float sg) {
    float x = sg * 27;
    for (int s = -1; s <= 1; s += 2) { box(I, V(x, 1.2f, FZ + 3.6f*s), V(0.2f, 2.4f, 0.2f), WHITE); box(I, V(x + sg*2, 1.2f, FZ + 3.6f*s), V(0.1f, 2.4f, 0.1f), WHITE);
        box(I, V(x + sg, 1.2f, FZ + 3.6f*s), V(2, 2.4f, 0.04f), V(0.85f,0.85f,0.9f)); }
    box(I, V(x, 2.4f, FZ), V(0.2f, 0.2f, 7.4f), WHITE);
    box(I, V(x + sg*2, 1.2f, FZ), V(0.04f, 2.4f, 7.2f), V(0.85f,0.85f,0.9f));
    box(I, V(x + sg, 2.4f, FZ), V(2, 0.04f, 7.2f), V(0.85f,0.85f,0.9f));
}
void drawFloodMasts() {                              // 4 big stadium light towers
    for (int i = 0; i < 4; i++) {
        float mx = (i % 2) ? 33.f : -33.f, mz = FZ + (i < 2 ? -25.f : 25.f);
        cyl(I * translateM(mx, 12, mz) * rotateX(PI/2), V(0,0,0), V(0.45f, 0.45f, 24), V(0.45f,0.47f,0.5f), 7);
        box(I, V(mx, 0.45f, mz), V(1.8f, 0.9f, 1.8f), CONCRETE, 2);
        Mat4 H = frame(mx, mz, atan2f(-mx, FZ - mz), 24.0f) * rotateX(0.3f);
        box(H, V(0, 0, 0), V(6.4f, 4.8f, 0.5f), DARK, 7);
        for (int r = 0; r < 3; r++) for (int c = 0; c < 4; c++)
            box(H, V(-2.25f + c*1.5f, -1.5f + r*1.5f, 0.3f), V(1.2f, 1.2f, 0.2f), nightMode ? V(1.0f,0.98f,0.88f) : V(0.7f,0.7f,0.72f), 10);
    }
}
void drawField() {
    box(I, V(0, 0.03f, FZ), V(60, 0.06f, 44), FIELDC, 9);
    lineBox(-27, FZ - 19, 27, FZ - 19); lineBox(-27, FZ + 19, 27, FZ + 19);
    lineBox(-27, FZ - 19, -27, FZ + 19); lineBox(27, FZ - 19, 27, FZ + 19); lineBox(0, FZ - 19, 0, FZ + 19);
    for (int s = -1; s <= 1; s += 2) {
        lineBox(s*27.f, FZ - 8, s*15.5f, FZ - 8); lineBox(s*27.f, FZ + 8, s*15.5f, FZ + 8); lineBox(s*15.5f, FZ - 8, s*15.5f, FZ + 8);
    }
    for (int i = 0; i < 24; i++) { float a = i * PI / 12; lineBox(5*cosf(a), FZ + 5*sinf(a), 5*cosf(a) + 0.01f, FZ + 5*sinf(a) + 0.01f); }
    goal(-1); goal(1);
    drawFloodMasts();
}

// ============================================================================
//  SEGMENT 15 : PEOPLE  (person faces local +X)
// ============================================================================
void drawPerson(const Mat4& P0, Vec3 shirt, Vec3 pants, Vec3 skin, Vec3 hair, float phase, float amp, float sc, Vec3 hat = Vec3{-1, 0, 0}) {
    Mat4 P = P0 * scaleM(sc, sc, sc);
    float sw = sinf(phase) * amp;
    for (int s = -1; s <= 1; s += 2) {
        Mat4 Lg = P * translateM(0, 0.9f, 0.12f * s) * rotateZ(sw * s);
        box(Lg, V(0, -0.45f, 0), V(0.2f, 0.9f, 0.2f), pants);
        box(Lg, V(0.05f, -0.88f, 0), V(0.32f, 0.1f, 0.22f), DARK);
        Mat4 Ar = P * translateM(0, 1.55f, 0.32f * s) * rotateZ(-sw * s);
        box(Ar, V(0, -0.35f, 0), V(0.15f, 0.7f, 0.15f), shirt);
        ball(Ar, V(0, -0.72f, 0), V(0.08f, 0.08f, 0.08f), skin);
    }
    box(P, V(0, 1.25f, 0), V(0.3f, 0.7f, 0.5f), shirt);
    ball(P, V(0.02f, 1.75f, 0), V(0.17f, 0.19f, 0.17f), skin);
    ball(P, V(-0.02f, 1.82f, 0), V(0.18f, 0.16f, 0.18f), hair);
    if (hat.x >= 0) box(P, V(0, 1.97f, 0), V(0.4f, 0.1f, 0.4f), hat);
}

void wheelBike(const Mat4& P, float cx, float cy, float ang) {
    Mat4 W = P * translateM(cx, cy, 0);
    for (int i = 0; i < 12; i++) box(W * rotateZ(i * PI / 6), V(0.45f, 0, 0), V(0.06f, 0.2f, 0.06f), DARK);
    for (int k = 0; k < 3; k++) box(W * rotateZ(ang + k * PI / 3), V(0, 0, 0), V(0.9f, 0.03f, 0.03f), STEEL);
}
void legIK(const Mat4& P, float hx, float hy, float fx, float fy, float z, Vec3 c) {
    float dx = fx - hx, dy = fy - hy, D = sqrtf(dx*dx + dy*dy), l = 0.55f, d = min(D, 2*l - 0.01f);
    float vx = dx / D, vy = dy / D, mx = hx + vx*d/2, my = hy + vy*d/2, h = sqrtf(l*l - d*d/4);
    float kx = mx - vy*h, ky = my + vx*h, ex = hx + vx*d, ey = hy + vy*d;
    beam(P, hx, hy, kx, ky, z, 0.12f, 0.12f, c); beam(P, kx, ky, ex, ey, z, 0.1f, 0.1f, c);
    box(P, V(ex + 0.05f, ey, z), V(0.22f, 0.07f, 0.12f), DARK);
}
// the girl on the bicycle (faces +X). dist = distance travelled (wheels + pedals turn with it)
void drawCyclist(const Mat4& P, float dist, float t) {
    Vec3 BIKE = V(0.85f,0.1f,0.25f), SKIN = V(0.82f,0.6f,0.45f), HAIR = V(0.05f,0.04f,0.04f), KURTA = V(0.95f,0.45f,0.65f), LEG = V(0.2f,0.25f,0.5f);
    float wa = -dist / 0.45f;
    wheelBike(P, -0.62f, 0.45f, wa); wheelBike(P, 0.62f, 0.45f, wa);
    beam(P, 0, 0.42f, -0.28f, 0.98f, 0, 0.06f, 0.06f, BIKE); beam(P, 0, 0.42f, 0.42f, 0.95f, 0, 0.06f, 0.06f, BIKE);
    beam(P, -0.28f, 0.98f, 0.42f, 0.95f, 0, 0.06f, 0.06f, BIKE); beam(P, -0.28f, 0.98f, -0.62f, 0.45f, 0, 0.05f, 0.05f, BIKE);
    beam(P, 0, 0.42f, -0.62f, 0.45f, 0, 0.05f, 0.05f, BIKE); beam(P, 0.42f, 0.95f, 0.62f, 0.45f, 0, 0.05f, 0.05f, BIKE);
    beam(P, 0.42f, 0.95f, 0.48f, 1.1f, 0, 0.05f, 0.05f, DARK);
    box(P, V(0.48f, 1.1f, 0), V(0.06f, 0.06f, 0.62f), DARK);
    box(P, V(-0.3f, 1.02f, 0), V(0.3f, 0.06f, 0.16f), DARK);
    box(P, V(0.72f, 1.0f, 0), V(0.3f, 0.22f, 0.4f), WOOD, 5);                    // basket
    float ca = dist * 2.2f;
    float px1 = 0.2f * cosf(ca), py1 = 0.42f + 0.2f * sinf(ca), px2 = -px1, py2 = 0.84f - py1;
    legIK(P, -0.28f, 1.05f, px1, py1, 0.14f, LEG); legIK(P, -0.28f, 1.05f, px2, py2, -0.14f, LEG);
    beam(P, -0.3f, 1.08f, 0.1f, 1.62f, 0, 0.34f, 0.3f, KURTA);
    ball(P, V(0.2f, 1.82f, 0), V(0.17f, 0.19f, 0.16f), SKIN);
    ball(P, V(0.15f, 1.87f, 0), V(0.19f, 0.17f, 0.18f), HAIR);
    box(P * translateM(0.0f, 1.8f, 0) * rotateZ(0.5f + 0.2f * sinf(t * 6)), V(-0.3f, 0, 0), V(0.6f, 0.1f, 0.1f), HAIR);   // ponytail
    for (int s = -1; s <= 1; s += 2) beam(P, 0.1f, 1.55f, 0.48f, 1.1f, 0.28f * s, 0.1f, 0.1f, KURTA);
    box(P * translateM(0.05f, 1.55f, 0) * rotateZ(0.2f + 0.25f * sinf(t * 7)), V(-0.5f, 0, 0), V(1.0f, 0.05f, 0.34f), V(0.95f,0.95f,1.0f));  // scarf
}

// ---- paths that vehicles / walkers follow (rounded rectangles, arc-length parametrised) ----
struct Path { vector<Vec3> pts; vector<float> cum; float len = 0; };
void finishPath(Path& p) {
    p.cum.assign(p.pts.size(), 0);
    for (size_t i = 1; i < p.pts.size(); i++) { Vec3 d = p.pts[i] - p.pts[i-1]; p.cum[i] = p.cum[i-1] + sqrtf(d.x*d.x + d.z*d.z); }
    p.len = p.cum.back();
}
Path makeLoop(float x0, float x1, float z0, float z1, float R) {
    Path p; const int N = 8;
    auto arc = [&](float cx, float cz, float a0, float a1) { for (int i = 0; i <= N; i++) { float a = a0 + (a1 - a0) * i / N; p.pts.push_back(V(cx + R*cosf(a), 0, cz + R*sinf(a))); } };
    arc(x1 - R, z0 + R, -PI/2, 0); arc(x1 - R, z1 - R, 0, PI/2); arc(x0 + R, z1 - R, PI/2, PI); arc(x0 + R, z0 + R, PI, 1.5f*PI);
    p.pts.push_back(p.pts[0]); finishPath(p); return p;
}
Path reversePath(Path p) { reverse(p.pts.begin(), p.pts.end()); finishPath(p); return p; }
void samplePath(const Path& p, float s, Vec3& pos, float& hd) {
    s = fmodf(s, p.len); if (s < 0) s += p.len;
    size_t i = 1; while (i < p.cum.size() - 1 && p.cum[i] < s) i++;
    Vec3 a = p.pts[i-1], b = p.pts[i]; float seg = p.cum[i] - p.cum[i-1], t = seg > 1e-5f ? (s - p.cum[i-1]) / seg : 0;
    pos = a + (b - a) * t; hd = headingOf(b.x - a.x, b.z - a.z);
}

// ============================================================================
//  SEGMENT 16 : CARS + BUSES (keep to the LEFT lane like in Bangladesh)
// ============================================================================
struct Vehicle { bool bus; int loop; float s, speed; Vec3 col; float f = 1, stuck = 0, ghost = 0; };
vector<Path> loops; vector<Vehicle> vehicles;
struct Walker { int loop; float s, speed; Vec3 shirt, pants; };
vector<Path> pedLoops; vector<Walker> walkers;

void initTraffic() {
    loops.push_back(makeLoop(-40.2f, 40.2f, -50.2f, 90.2f, 7.4f));                       // 0 big ring, outer lane (buses + cars)
    loops.push_back(reversePath(makeLoop(-35.8f, 35.8f, -45.8f, 85.8f, 3.0f)));          // 1 big ring, inner lane
    loops.push_back(makeLoop(-2.2f, 2.2f, 34, 128, 2.19f));                              // 2 main road in/out
    loops.push_back(makeLoop(-40.2f, -35.8f, -138, -62, 2.19f));                         // 3 hall road
    loops.push_back(makeLoop(35.8f, 40.2f, -145, -62, 2.19f));                           // 4 right road (down to Rokeya Hall)
    Vec3 red = V(0.8f,0.1f,0.1f), blue = V(0.12f,0.28f,0.75f), yel = V(0.9f,0.75f,0.1f), wht = V(0.9f,0.9f,0.92f), grn = V(0.1f,0.55f,0.25f), org = V(0.9f,0.45f,0.1f);
    vehicles = { {true,0,0,7,wht}, {false,0,70,7,red}, {true,0,140,7,wht}, {false,0,210,7,yel}, {true,0,280,7,wht}, {false,0,350,7,blue},
                 {false,1,20,6,grn}, {false,1,150,6,org}, {false,1,280,6,blue},
                 {false,2,10,6.5f,red}, {false,2,110,6.5f,yel},
                 {false,3,10,6,blue}, {false,3,80,6,grn}, {false,4,20,6,org} };
    pedLoops.push_back(makeLoop(-6.3f, 6.3f, 34, 80, 2.0f));
    pedLoops.push_back(makeLoop(-88, -46, -26, 9, 3.0f));
    pedLoops.push_back(makeLoop(49, 60, -140, -100, 2.5f));                              // girls inside Rokeya Hall compound
    walkers = { {0,0,1.6f,V(0.9f,0.3f,0.3f),V(0.2f,0.2f,0.3f)}, {0,70,1.5f,V(0.3f,0.5f,0.9f),V(0.15f,0.15f,0.15f)}, {0,150,1.7f,V(0.9f,0.8f,0.2f),V(0.3f,0.3f,0.4f)},
                {1,0,1.4f,V(0.3f,0.7f,0.4f),V(0.2f,0.2f,0.3f)}, {1,60,1.5f,V(0.8f,0.4f,0.7f),V(0.25f,0.2f,0.2f)},
                {2,0,1.2f,V(0.95f,0.5f,0.7f),V(0.95f,0.9f,0.8f)}, {2,45,1.1f,V(0.4f,0.7f,0.95f),V(0.95f,0.9f,0.8f)} };
}

// Only ONE vehicle may be inside the big junction (main road x ring road, in front of the gate) at a time.
// Everybody else waits before it. Vehicles also brake for vehicles ahead and for people.
struct Zone { float x0, x1, z0, z1; int owner; };
Zone junction = {-7, 7, 82, 95, -1};
bool inZone(Vec3 p, float m) { return p.x > junction.x0 - m && p.x < junction.x1 + m && p.z > junction.z0 - m && p.z < junction.z1 + m; }
void updateTraffic(float dt) {
    int n = (int)vehicles.size(); vector<Vec3> P(n); vector<float> H(n), Wd;
    for (int i = 0; i < n; i++) samplePath(loops[vehicles[i].loop], vehicles[i].s, P[i], H[i]);
    vector<Vec3> W;
    for (const Walker& w : walkers) { Vec3 p; float hd; samplePath(pedLoops[w.loop], w.s, p, hd); W.push_back(p); }
    if (junction.owner >= 0 && !inZone(P[junction.owner], vehicles[junction.owner].bus ? 7.0f : 3.4f)) junction.owner = -1;
    for (int i = 0; i < n; i++) {
        Vehicle& a = vehicles[i];
        float fx = cosf(H[i]), fz = -sinf(H[i]), rx = sinf(H[i]), rz = cosf(H[i]);
        float ra = a.bus ? 5.9f : 2.2f, target = 1;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            const Vehicle& b = vehicles[j];
            if (!(j < i || b.loop == a.loop)) continue;
            float dx = P[j].x - P[i].x, dz = P[j].z - P[i].z;
            float along = dx*fx + dz*fz, lat = fabsf(dx*rx + dz*rz), dist = sqrtf(dx*dx + dz*dz);
            float rb = b.bus ? 5.9f : 2.2f, cd = cosf(H[i] - H[j]), t = 1;
            if (cd > 0.8f) { if (along > 0 && lat < 2.6f) t = (along - (ra + rb + 1.5f)) / 6.0f; }
            else if (cd > -0.8f) { float rr = (ra + rb) * 0.75f + 2.5f; if (along > -1.0f && dist < rr + 4.0f) t = (dist - rr) / 4.0f; }
            target = min(target, max(0.0f, min(1.0f, t)));
        }
        for (const Vec3& q : W) {                                            // never run over people
            float dx = q.x - P[i].x, dz = q.z - P[i].z, al = dx*fx + dz*fz, lt = fabsf(dx*rx + dz*rz);
            if (al > 0 && lt < 2.6f) target = min(target, max(0.0f, min(1.0f, (al - (ra + 3.0f)) / 5.0f)));
        }
        Vec3 pf; float hf2; samplePath(loops[a.loop], a.s + (a.bus ? 12.0f : 8.0f), pf, hf2);
        bool nowIn = inZone(P[i], 0.0f), aheadIn = inZone(pf, 0.0f);
        if (junction.owner != i) {
            if (!nowIn && aheadIn) { if (junction.owner < 0) junction.owner = i; else target = 0; }
            else if (nowIn && junction.owner < 0) junction.owner = i;
        }
        if (a.stuck > 8) { a.ghost = 3; a.stuck = 0; }                    // never stay deadlocked
        if (a.ghost > 0) { target = (target < 0.05f && junction.owner != i && aheadIn && !nowIn) ? 0.0f : 1.0f; a.ghost -= dt; }
        a.f += max(-dt * 9.0f, min(dt * 2.5f, target - a.f));
        a.stuck = a.f < 0.05f ? a.stuck + dt : 0;
    }
    for (Vehicle& v : vehicles) v.s += v.speed * v.f * dt;
    for (Walker& w : walkers) w.s += w.speed * dt;
}

// round tyre (cylinder) + metal hub with spokes; rolls with the distance driven
void drawWheel(const Mat4& P, float x, float y, float z, float r, float wd, float roll) {
    Mat4 W = P * translateM(x, y, z) * rotateZ(-roll / r);
    cyl(W, V(0, 0, 0), V(r, r, wd), V(0.05f,0.05f,0.06f));
    cyl(W, V(0, 0, 0), V(r*0.55f, r*0.55f, wd + 0.05f), V(0.72f,0.73f,0.76f), 7);
    box(W, V(0, 0, 0), V(r*1.0f, r*0.12f, wd + 0.08f), V(0.35f,0.35f,0.38f), 7);
    box(W * rotateZ(PI/2), V(0, 0, 0), V(r*1.0f, r*0.12f, wd + 0.08f), V(0.35f,0.35f,0.38f), 7);
}
void drawCar(const Mat4& P, Vec3 col, float roll) {
    box(P, V(0, 0.75f, 0), V(4.2f, 0.8f, 1.9f), col, 7);
    box(P, V(-0.3f, 1.45f, 0), V(2.2f, 0.75f, 1.7f), col, 7);
    box(P, V(-0.3f, 1.5f, 0), V(2.3f, 0.45f, 1.75f), V(0.15f,0.22f,0.30f), 8);
    for (int sx = -1; sx <= 1; sx += 2) for (int sz = -1; sz <= 1; sz += 2) drawWheel(P, 1.3f*sx, 0.4f, 0.95f*sz, 0.4f, 0.3f, roll);
    for (int s = -1; s <= 1; s += 2) { box(P, V(2.12f, 0.8f, 0.6f*s), V(0.1f, 0.25f, 0.4f), V(1.0f,0.95f,0.6f), 10); box(P, V(-2.12f, 0.8f, 0.6f*s), V(0.1f, 0.25f, 0.4f), V(0.8f,0.05f,0.05f)); }
}
void drawBus(const Mat4& P, float roll) {
    Vec3 CHOC = V(0.27f,0.14f,0.07f), BRN = V(0.58f,0.35f,0.16f), TXT = V(0.97f,0.93f,0.80f);
    box(P, V(0, 0.8f, 0), V(11.5f, 0.9f, 2.6f), CHOC, 7);
    box(P, V(0, 1.9f, 0), V(11.5f, 1.3f, 2.6f), BRN, 7);
    box(P, V(0, 2.9f, 0), V(11.5f, 0.8f, 2.55f), CHOC, 7);
    box(P, V(0, 3.4f, 0), V(11.6f, 0.15f, 2.7f), CHOC, 7);
    for (int i = 0; i < 8; i++) for (int s = -1; s <= 1; s += 2) box(P, V(-4.8f + i*1.37f, 2.92f, 1.29f*s), V(1.0f, 0.65f, 0.06f), GLASS, 8);
    box(P, V(5.78f, 2.1f, 0), V(0.1f, 2.4f, 2.3f), GLASS, 8);
    for (int s = -1; s <= 1; s += 2) { box(P, V(5.78f, 0.9f, 0.9f*s), V(0.1f, 0.25f, 0.4f), V(1.0f,0.95f,0.6f), 10); box(P, V(-5.78f, 0.9f, 0.9f*s), V(0.1f, 0.25f, 0.4f), V(0.8f,0.05f,0.05f)); }
    for (float ax : {3.8f, -2.8f, -4.2f}) for (int sz = -1; sz <= 1; sz += 2) drawWheel(P, ax, 0.55f, 1.12f*sz, 0.55f, 0.4f, roll);
    for (int side = 0; side < 2; side++) {          // the name on both sides
        Mat4 Q = side == 0 ? P : P * rotateY(PI);
        drawTextCentered(Q, "KHULNA UNIVERSITY OF", -0.3f, 1.95f, 1.31f, 0.07f, 0.04f, TXT);
        drawTextCentered(Q, "ENGINEERING AND TECHNOLOGY", -0.3f, 1.36f, 1.31f, 0.07f, 0.04f, TXT);
    }
}
void drawVehicles() {
    for (const Vehicle& v : vehicles) {
        Vec3 pos; float hd; samplePath(loops[v.loop], v.s, pos, hd);
        Mat4 P = translateM(pos.x, 0.1f, pos.z) * rotateY(hd);
        if (v.bus) drawBus(P, v.s); else drawCar(P, v.col, v.s);
    }
    for (const Walker& w : walkers) {
        Vec3 pos; float hd; samplePath(pedLoops[w.loop], w.s, pos, hd);
        drawPerson(frame(pos.x, pos.z, hd, 0.09f), w.shirt, w.pants, V(0.8f,0.58f,0.42f), V(0.06f,0.05f,0.05f), w.s * 3.5f, 0.55f, 0.95f);
    }
    Path cyc = makeLoop(-23, 23, 10, 24, 4.0f);                    // girl cycling around the plaza
    Vec3 pos; float hd; float dist = gTime * 4.5f; samplePath(cyc, dist, pos, hd);
    drawCyclist(frame(pos.x, pos.z, hd, 0.12f), dist, gTime);
}

// ============================================================================
//  SEGMENT 17 : KIDS ON THE PLAZA + FOOTBALL MATCH
// ============================================================================
void drawKids() {
    float t = gTime;
    Vec3 sh[6] = {V(0.9f,0.2f,0.2f), V(0.2f,0.5f,0.9f), V(0.95f,0.8f,0.1f), V(0.2f,0.7f,0.35f), V(0.85f,0.4f,0.7f), V(0.95f,0.5f,0.15f)};
    Vec3 skin = V(0.8f,0.58f,0.42f), hairc = V(0.08f,0.06f,0.05f), shorts = V(0.2f,0.25f,0.4f);
    float cxs[3] = {-6, 6, 0}, czs[3] = {18, 19, 15}, rs[3] = {3.2f, 2.6f, 2.0f}, ws[3] = {0.9f, -1.1f, 1.3f};
    for (int i = 0; i < 3; i++) {                                   // kids chasing each other
        float a = t*ws[i] + i*2.0f, sg = ws[i] > 0 ? 1.f : -1.f;
        drawPerson(frame(cxs[i] + rs[i]*cosf(a), czs[i] + rs[i]*sinf(a), headingOf(-sinf(a)*sg, cosf(a)*sg), 0.12f), sh[i], shorts, skin, hairc, t*9 + i, 0.8f, 0.62f);
    }
    float bump = fabsf(sinf(t * 8)) * 0.06f;                        // two kids playing catch
    drawPerson(frame(-3, 12, 0, 0.12f + bump), sh[3], shorts, skin, hairc, 0, 0, 0.62f);
    drawPerson(frame(3, 12, PI, 0.12f + bump), sh[4], shorts, skin, hairc, 0, 0, 0.62f);
    float tt = t * 0.4f, ph = tt - floorf(tt); int dir = (int)floorf(tt) % 2;
    float bx = dir ? 3 - 6*ph : -3 + 6*ph;
    ball(I, V(bx, 0.12f + 1.1f + 2.2f*sinf(ph*PI), 12), V(0.18f, 0.18f, 0.18f), V(0.9f,0.15f,0.15f));
    drawPerson(frame(9, 13, -PI/2, 0.12f + fabsf(sinf(t*5)) * 0.5f), sh[5], shorts, skin, hairc, t*10, 1.2f, 0.62f);   // hopping kid
}
Vec3 playerPos(int i, float t) {
    static const float hx[10] = {-23,-13,-13,-5,-5, 23,13,13,5,5}, hz[10] = {0,-10,10,-5,6, 0,10,-10,5,-6};
    float gk = (i % 5 == 0), ax = gk ? 1.5f : 7.0f, az = gk ? 3.0f : 6.0f, ph = i * 1.7f;
    return V(hx[i] + ax*sinf(t*0.45f + ph), 0, FZ + hz[i] + az*sinf(t*0.37f + ph*1.3f));
}
void drawFootball() {
    float t = gTime;
    for (int i = 0; i < 10; i++) {
        Vec3 p = playerPos(i, t), q = playerPos(i, t + 0.05f);
        Vec3 shirt = i < 5 ? V(0.85f,0.1f,0.1f) : V(0.1f,0.25f,0.85f);
        drawPerson(frame(p.x, p.z, headingOf(q.x - p.x, q.z - p.z), 0.06f), shirt, WHITE, V(0.78f,0.56f,0.4f), V(0.08f,0.06f,0.05f), t*9 + i, 0.9f, 0.9f);
    }
    float tt = t / 2.4f; int k = (int)floorf(tt); float f = tt - k, s = f*f*(3 - 2*f);
    int a = k % 10, b = (a + 3 + (k % 3)) % 10; Vec3 pa = playerPos(a, t), pb = playerPos(b, t);
    ball(I, V(pa.x + (pb.x - pa.x)*s, 0.35f + 1.3f*sinf(f*PI), pa.z + (pb.z - pa.z)*s), V(0.3f, 0.3f, 0.3f), WHITE);
}

// ============================================================================
//  SEGMENT 18 : BIG MACHINES  (EEE: G | MECHANICAL: M | CIVIL: C)
// ============================================================================
void drawGear(const Mat4& R, float r, float ang, Vec3 c) {
    box(R, V(0, 0, 0), V(r*0.35f, r*0.35f, 0.5f), c, 7);
    for (int k = 0; k < 4; k++) box(R * rotateZ(ang + k*PI/4), V(0, 0, 0), V(r*1.8f, 0.22f, 0.35f), c, 7);
    for (int k = 0; k < 12; k++) box(R * rotateZ(ang + k*PI/6), V(r, 0, 0), V(0.35f, 0.5f, 0.4f), c, 7);
}
void drawEEEYard(const Mat4& F) {
    float t = rigEEE.t, w = rigEEE.w;
    box(F, V(0, 0.04f, 14), V(26, 0.08f, 15), CONCRETE, 2);
    Mat4 T = F * translateM(-8, 0, 15);                                       // wind turbine
    box(T, V(0, 6, 0), V(0.7f, 12, 0.7f), WHITE, 7);
    box(T, V(0, 12.4f, 0.3f), V(1.2f, 1.1f, 2.6f), WHITE, 7);
    Mat4 R = T * translateM(0, 12.4f, 1.7f) * rotateZ(t * 2.0f);
    ball(R, V(0, 0, 0.1f), V(0.5f, 0.5f, 0.6f), WHITE, 7);
    for (int k = 0; k < 3; k++) box(R * rotateZ(k * 2.094f), V(0, 3.2f, 0), V(0.55f, 6.4f, 0.12f), WHITE, 7);
    float shake = sinf(t * 60) * 0.015f * w;                                  // generator set
    Mat4 Gm = F * translateM(2, 0, 14);
    box(Gm, V(0, 0.3f, 0), V(8, 0.6f, 3.2f), V(0.2f,0.2f,0.22f));
    box(Gm, V(-2, 1.6f + shake, 0), V(3.2f, 2.0f, 2.2f), V(0.85f,0.55f,0.08f), 7);
    box(Gm, V(1.6f, 1.5f + shake, 0), V(2.6f, 1.8f, 2.0f), V(0.25f,0.35f,0.6f), 7);
    drawGear(Gm * translateM(-2, 2.2f, 1.3f), 1.2f, t * 5.0f, V(0.3f,0.3f,0.32f));
    box(Gm, V(-3, 3.2f, 0.6f), V(0.3f, 2.4f, 0.3f), DARK);
    if (w > 0.05f) for (int k = 0; k < 4; k++) { float u = fmodf(t*1.2f + k*0.25f, 1.0f);
        ball(Gm, V(-3 + u*1.5f, 4.6f + u*3, 0.6f), V(0.3f + u*0.9f, 0.3f + u*0.9f, 0.3f + u*0.9f), V(0.55f,0.55f,0.55f) * (1 - u*0.4f)); }
    Mat4 Tr = F * translateM(9, 0, 14);                                       // transformer
    box(Tr, V(0, 1.3f, 0), V(2.6f, 2.6f, 1.8f), V(0.5f,0.55f,0.5f), 7);
    for (int i = -1; i <= 1; i++) box(Tr, V(i*0.8f, 3.05f, 0), V(0.25f, 0.9f, 0.25f), V(0.9f,0.85f,0.7f));
    box(Tr, V(0, 1.6f, 0.95f), V(1.0f, 0.7f, 0.1f), YELLOW);
    box(Tr, V(0.9f, 0.6f, 0.95f), V(0.3f, 0.3f, 0.1f), rigEEE.on ? V(0.2f,1.0f,0.3f) : V(0.9f,0.1f,0.1f), 10);
}
void drawMechYard(const Mat4& F) {
    float mt = rigMech.t; Vec3 SB = V(0.2f,0.45f,0.7f), SI = V(0.8f,0.82f,0.85f);
    box(F, V(0, 0.04f, 15), V(26, 0.08f, 17), CONCRETE, 2);
    for (int ix = -1; ix <= 1; ix++) for (int iz = 0; iz < 2; iz++) box(F, V(ix*11.0f, 3.5f, 7.5f + iz*15.0f), V(0.5f, 7, 0.5f), STEEL, 7);
    box(F * translateM(0, 7.2f, 15) * rotateX(-0.05f), V(0, 0, 0), V(25, 0.3f, 17.5f), V(0.45f,0.5f,0.58f), 7);    // roof (chawni)
    for (int i = 0; i < 17; i++) box(F * translateM(0, 7.2f, 15) * rotateX(-0.05f), V(-12 + i*1.5f, 0.2f, 0), V(0.15f, 0.15f, 17.6f), V(0.35f,0.4f,0.48f), 7);
    Mat4 A = F * translateM(-8, 0, 15);                                       // lathe
    box(A, V(0, 1.0f, 0), V(7, 0.5f, 1.6f), V(0.2f,0.45f,0.35f), 7);
    box(A, V(-2.8f, 0.5f, 0), V(0.8f, 1.0f, 1.4f), DARK); box(A, V(2.8f, 0.5f, 0), V(0.8f, 1.0f, 1.4f), DARK);
    box(A, V(-3, 1.8f, 0), V(1.6f, 1.4f, 1.7f), SB, 7);
    box(A * translateM(-2.1f, 1.9f, 0) * rotateX(mt * 14), V(0, 0, 0), V(0.3f, 1.0f, 1.0f), DARK, 7);
    box(A * translateM(-0.4f, 1.9f, 0) * rotateX(mt * 14), V(0, 0, 0), V(3, 0.25f, 0.25f), SI, 7);
    box(A, V(3, 1.7f, 0), V(1, 1.2f, 1.2f), SB, 7);
    Mat4 Pr = F * translateM(0, 0, 15);                                       // hydraulic press
    float py = 4.2f - 1.4f * (0.5f + 0.5f * sinf(mt * 3.0f));
    box(Pr, V(0, 0.35f, 0), V(3.6f, 0.7f, 2), DARK, 7);
    box(Pr, V(-1.4f, 3.3f, 0), V(0.6f, 5.5f, 0.6f), SB, 7); box(Pr, V(1.4f, 3.3f, 0), V(0.6f, 5.5f, 0.6f), SB, 7);
    box(Pr, V(0, 6.3f, 0), V(3.6f, 1.0f, 1.4f), SB, 7);
    box(Pr, V(0, py + 1.0f, 0), V(0.5f, 2.0f, 0.5f), SI, 7); box(Pr, V(0, py, 0), V(2.2f, 0.5f, 1.2f), V(0.6f,0.15f,0.1f), 7);
    box(F, V(8, 0.7f, 14.5f), V(2.4f, 1.4f, 1.4f), V(0.7f,0.2f,0.2f), 7);       // motor + big gears
    drawGear(F * translateM(8, 3.4f, 15.5f), 2.2f, mt * 3.0f, V(0.75f,0.55f,0.1f));
    drawGear(F * translateM(12.1f, 3.4f, 15.5f), 1.0f, -mt * 3.0f * 2.2f, V(0.3f,0.5f,0.75f));
    box(F, V(0, 0.95f, 20.5f), V(18, 0.35f, 1.4f), V(0.2f,0.2f,0.22f), 7);      // conveyor
    for (int i = -4; i <= 4; i += 2) { box(F, V(i*2.2f, 0.45f, 20.5f), V(0.4f, 0.9f, 1.2f), DARK); }
    for (int i = 0; i < 5; i++) box(F, V(-8.5f + fmodf(mt*2.2f + i*3.6f, 18.0f), 1.5f, 20.5f), V(0.9f, 0.7f, 0.9f), WOOD, 5);
    box(F, V(-11.2f, 1.2f, 21.5f), V(0.8f, 2.4f, 0.4f), V(0.3f,0.3f,0.32f), 7);  // control box + lamp
    box(F, V(-11.2f, 2.2f, 21.75f), V(0.3f, 0.3f, 0.1f), rigMech.on ? V(0.2f,1.0f,0.3f) : V(0.9f,0.1f,0.1f), 10);
}
void drawCivilYard(const Mat4& F) {
    float ct = rigCivil.t; Vec3 OR = V(0.95f,0.45f,0.1f);
    box(F, V(0, 0.04f, 15), V(26, 0.08f, 17), V(0.55f,0.5f,0.42f), 2);
    for (int i = 0; i < 9; i++) { box(F, V(-12 + i*3.0f, 0.5f, 23.4f), V(2.4f, 0.6f, 0.15f), i % 2 ? WHITE : OR); box(F, V(-12 + i*3.0f, 0.25f, 23.4f), V(0.15f, 0.5f, 0.4f), DARK); }
    Mat4 C = F * translateM(-9, 0, 15);                                       // tower crane
    box(C, V(0, 0.5f, 0), V(3, 1, 3), V(0.5f,0.5f,0.5f), 2);
    for (int sx = -1; sx <= 1; sx += 2) for (int sz = -1; sz <= 1; sz += 2) box(C, V(sx*0.9f, 11.5f, sz*0.9f), V(0.25f, 22, 0.25f), YELLOW, 7);
    for (int k = 1; k <= 7; k++) { float y = k*3.0f; box(C, V(0, y, 0.9f), V(1.8f, 0.15f, 0.15f), YELLOW, 7); box(C, V(0, y, -0.9f), V(1.8f, 0.15f, 0.15f), YELLOW, 7); box(C, V(0.9f, y, 0), V(0.15f, 0.15f, 1.8f), YELLOW, 7); box(C, V(-0.9f, y, 0), V(0.15f, 0.15f, 1.8f), YELLOW, 7); }
    Mat4 J = C * translateM(0, 22.4f, 0) * rotateY(sinf(ct * 0.35f) * 1.0f);
    box(J, V(0, 0, 0), V(2, 1.2f, 2), V(0.9f,0.9f,0.9f), 7);
    box(J, V(7, 0.6f, 0), V(22, 0.5f, 0.8f), YELLOW, 7); box(J, V(-4, 0.6f, 0), V(8, 0.5f, 0.8f), YELLOW, 7);
    box(J, V(-7, 0, 0), V(2.5f, 1.8f, 1.8f), V(0.4f,0.4f,0.4f), 2);
    box(J, V(0, 2.2f, 0), V(0.3f, 3, 0.3f), YELLOW, 7);
    float tx = 8 + 4 * sinf(ct * 0.8f), len = 8 + 4 * (0.5f + 0.5f * sinf(ct * 1.1f));
    box(J, V(tx, 0.1f, 0), V(1.2f, 0.7f, 1.2f), DARK, 7);
    box(J, V(tx, 0.1f - len/2, 0), V(0.06f, len, 0.06f), DARK);
    box(J, V(tx, 0.1f - len - 0.4f, 0), V(1.8f, 0.5f, 0.9f), V(0.6f,0.6f,0.65f), 7);
    Mat4 S = F * translateM(9, 0, 13);                                        // building under construction
    for (int sx = -1; sx <= 1; sx += 2) for (int sz = -1; sz <= 1; sz += 2) box(S, V(sx*3.0f, 4.5f, sz*2.5f), V(0.5f, 9, 0.5f), V(0.6f,0.6f,0.58f), 2);
    for (int f = 1; f <= 3; f++) box(S, V(0, f*3.0f, 0), V(6.5f, 0.3f, 5.5f), V(0.6f,0.6f,0.58f), 2);
    Mat4 E = F * translateM(-1, 0, 10);                                       // excavator
    box(E, V(0, 0.45f, 1.2f), V(4.6f, 0.9f, 0.9f), DARK); box(E, V(0, 0.45f, -1.2f), V(4.6f, 0.9f, 0.9f), DARK);
    Mat4 Bd = E * translateM(0, 0.9f, 0) * rotateY(sinf(ct * 0.7f) * 0.8f);
    box(Bd, V(0, 0.4f, 0), V(3, 0.8f, 2.4f), YELLOW, 7); box(Bd, V(-0.5f, 1.6f, 0), V(1.8f, 1.6f, 1.8f), YELLOW, 7);
    box(Bd, V(-0.3f, 1.8f, 0.92f), V(1.2f, 0.9f, 0.05f), GLASS, 8);
    Mat4 Bm = Bd * translateM(1.0f, 1.0f, 0) * rotateZ(0.7f + 0.25f * sinf(ct * 1.2f));
    box(Bm, V(1.7f, 0, 0), V(3.4f, 0.5f, 0.6f), YELLOW, 7);
    Mat4 St = Bm * translateM(3.4f, 0, 0) * rotateZ(-1.4f + 0.35f * sinf(ct * 1.5f));
    box(St, V(1.2f, 0, 0), V(2.4f, 0.4f, 0.5f), YELLOW, 7);
    Mat4 Bk = St * translateM(2.4f, 0, 0) * rotateZ(-0.9f + 0.4f * sinf(ct * 1.9f));
    box(Bk, V(0.4f, -0.2f, 0), V(0.9f, 0.6f, 0.9f), V(0.3f,0.3f,0.3f), 7);
    float rx = -6 + 6 * sinf(ct * 0.5f), rh = cosf(ct * 0.5f) >= 0 ? 0.f : PI;  // road roller
    Mat4 Rl = frame(rx, 19.5f, rh, 0.08f); Rl = F * Rl;
    box(Rl, V(0, 1.4f, 0), V(1.8f, 1.2f, 1.8f), OR, 7); box(Rl, V(-0.6f, 2.5f, 0), V(1.2f, 1.2f, 1.6f), OR, 7);
    cyl(Rl, V(1.4f, 0.8f, 0), V(0.8f, 0.8f, 2.5f), V(0.35f,0.35f,0.38f), 7); cyl(Rl, V(-1.4f, 0.8f, 0), V(0.8f, 0.8f, 2.5f), V(0.35f,0.35f,0.38f), 7);
    ball(F, V(3, 0.4f, 8.5f), V(2.2f, 1.1f, 2.2f), V(0.85f,0.75f,0.5f));         // sand + bricks + gravel
    ball(F, V(-3, 0.4f, 20.5f), V(1.8f, 0.9f, 1.8f), V(0.5f,0.5f,0.52f));
    for (int i = 0; i < 12; i++) box(F, V(5.5f + (i % 4)*0.8f, 0.2f + (i / 4)*0.4f, 8.5f), V(0.75f, 0.35f, 0.4f), V(0.65f,0.25f,0.15f));
    for (int i = 0; i < 2; i++) {                                              // workers
        float a = ct*0.6f + i*3.0f; Vec3 hat = i ? V(1,1,1) : YELLOW;
        float x = (i ? 4.0f : -5.0f) + 2.5f*cosf(a), z = 17 + 1.5f*sinf(a), sg = 1.0f;
        Mat4 Pw = F * frame(x, z, headingOf(-sinf(a)*sg, cosf(a)*sg), 0.08f);
        drawPerson(Pw, V(0.95f,0.5f,0.1f), V(0.2f,0.25f,0.45f), V(0.7f,0.5f,0.38f), V(0.05f,0.05f,0.05f), rigCivil.on ? ct*8 + i : 0, rigCivil.on ? 0.7f : 0, 0.95f, hat);
    }
}

// ============================================================================
//  SEGMENT 19 : NATURE  (ducks, lily pads, birds, clouds)
// ============================================================================
void drawPond() {
    float cx = (PX0 + PX1)/2, cz = (PZ0 + PZ1)/2;
    for (int i = 0; i < 14; i++) {
        float x = cx + sinf(i*12.9898f) * 13, z = cz + sinf(i*78.233f) * 11;
        ball(I, V(x + 0.4f*sinf(gTime*0.4f + i), -0.43f, z + 0.3f*cosf(gTime*0.3f + i)), V(0.7f, 0.03f, 0.7f), V(0.15f,0.5f,0.15f), 6);
    }
    for (int i = 0; i < 3; i++) {
        float a = gTime * (0.25f + 0.05f*i) + i*2.1f, rx = 11 - i*2, rz = 8 - i*1.5f;
        float x = cx + rx*cosf(a), z = cz + rz*sinf(a);
        Mat4 P = translateM(x, -0.42f + 0.04f*sinf(gTime*2 + i), z) * rotateY(headingOf(-rx*sinf(a), rz*cosf(a)));
        ball(P, V(0, 0.2f, 0), V(0.5f, 0.28f, 0.32f), WHITE);
        ball(P, V(0.4f, 0.55f, 0), V(0.2f, 0.22f, 0.2f), V(0.1f,0.4f,0.2f));
        box(P, V(0.65f, 0.5f, 0), V(0.22f, 0.06f, 0.12f), V(0.95f,0.5f,0.1f));
        ball(P, V(-0.5f, 0.3f, 0), V(0.2f, 0.15f, 0.15f), WHITE);
    }
}
void drawBirds() {
    Vec3 dk = V(0.12f,0.12f,0.14f);
    for (int i = 0; i < 6; i++) {
        float w = 0.35f + 0.05f*i, a = gTime*w + i*1.1f, R = 40 + i*9, cx = (i % 2) ? -20.f : 15.f, cz = (i % 3) * -30.f;
        Mat4 P = translateM(cx + R*cosf(a), 22 + 4*sinf(gTime*0.7f + i), cz + R*sinf(a)) * rotateY(headingOf(-sinf(a), cosf(a)));
        float f = sinf(gTime*9 + i) * 0.7f;
        box(P, V(0, 0, 0), V(0.7f, 0.18f, 0.25f), dk); ball(P, V(0.4f, 0.05f, 0), V(0.14f, 0.12f, 0.12f), dk);
        box(P * translateM(0, 0.05f, 0.12f) * rotateX(f), V(0, 0, 0.5f), V(0.4f, 0.04f, 1.0f), dk);
        box(P * translateM(0, 0.05f, -0.12f) * rotateX(-f), V(0, 0, -0.5f), V(0.4f, 0.04f, 1.0f), dk);
    }
}
void drawClouds() {
    for (int i = 0; i < 9; i++) {
        float x = fmodf(i*47.0f + gTime*(1.5f + 0.2f*i) + 300.0f, 600.0f) - 300.0f, z = -150.0f + fmodf(i*83.0f, 300.0f), y = 75.0f + (i % 3)*8.0f;
        Vec3 c = V(0.98f,0.98f,1.0f);
        ball(I, V(x, y, z), V(14, 4.5f, 8), c); ball(I, V(x + 9, y + 1, z + 2), V(10, 4, 7), c); ball(I, V(x - 9, y - 0.5f, z - 1), V(9, 3.5f, 6), c);
    }
}

// ============================================================================
//  SEGMENT 20 : ALL BUILDINGS + WHOLE SCENE
//  Left of the gate (-x): hill, Khan Jahan Ali, pond, 5 halls | Right (+x): CSE, EEE, 5-storey, ME, CE
// ============================================================================

// ---------- boys' clothes drying on a roof line (matId 11 is bent by the wind in the vertex shader)
void drawClothesLine(const Mat4& P, float x0, float x1, float y, float z, int seed) {
    float ry = y + 1.5f;
    box(P, V(x0, y + 0.75f, z), V(0.08f, 1.5f, 0.08f), STEEL, 7); box(P, V(x1, y + 0.75f, z), V(0.08f, 1.5f, 0.08f), STEEL, 7);
    box(P, V((x0 + x1)/2, ry, z), V(x1 - x0, 0.03f, 0.03f), V(0.8f,0.8f,0.7f));
    Vec3 pal[8] = {V(0.95f,0.95f,0.95f), V(0.2f,0.4f,0.85f), V(0.85f,0.15f,0.15f), V(0.2f,0.65f,0.3f), V(0.95f,0.8f,0.2f), V(0.15f,0.15f,0.2f), V(0.9f,0.5f,0.15f), V(0.6f,0.3f,0.7f)};
    int n = (int)((x1 - x0) / 1.1f);
    for (int i = 0; i < n; i++) {
        float x = x0 + (i + 0.5f) * (x1 - x0) / n;
        float h1 = frac(sinf(seed*12.9898f + i*78.233f) * 43758.5453f), h2 = frac(sinf(seed*3.7f + i*19.19f) * 9631.7f);
        Vec3 c = pal[(int)(h2 * 8) % 8]; int type = (int)(h1 * 4) % 4;
        if (type == 0)      { box(P, V(x, ry - 0.4f, z), V(0.5f, 0.7f, 0.04f), c, 11); box(P, V(x, ry - 0.15f, z), V(0.95f, 0.22f, 0.04f), c, 11); }        // shirt
        else if (type == 1) { box(P, V(x - 0.13f, ry - 0.5f, z), V(0.22f, 0.95f, 0.04f), c, 11); box(P, V(x + 0.13f, ry - 0.5f, z), V(0.22f, 0.95f, 0.04f), c, 11); }  // trousers
        else if (type == 2) box(P, V(x, ry - 0.55f, z), V(0.6f, 1.1f, 0.04f), c, 11);                                                                                // lungi / towel
        else                { box(P, V(x, ry - 0.32f, z), V(0.45f, 0.55f, 0.04f), c, 11); box(P, V(x, ry - 0.12f, z), V(0.75f, 0.18f, 0.04f), c, 11); }          // t-shirt
    }
}

// ---------- the six halls: each one looks different. st: 0 KJA | 1 Rashid | 2 Shaheed Smriti | 3 Fazlul Haque | 4 Amar Ekushey | 5 Lalan Shah
void drawHall(const Mat4& P, const string& name, int floors, int st) {
    const float w = 18, d = 8, hf = 4; float h = floors * hf, zf = d / 2;
    Vec3 wallc[7] = {V(0.93f,0.84f,0.55f), V(0.62f,0.78f,0.86f), V(0.70f,0.38f,0.30f), V(0.95f,0.94f,0.88f), V(0.95f,0.70f,0.45f), V(0.60f,0.74f,0.58f), V(0.82f,0.72f,0.90f)};
    Vec3 trimc[7] = {WHITE, V(0.65f,0.30f,0.25f), CREAM, V(0.2f,0.5f,0.35f), V(0.45f,0.25f,0.15f), V(0.95f,0.93f,0.85f), V(0.97f,0.88f,0.93f)};
    Vec3 W = wallc[st], T = trimc[st];
    float wUp = w, dUp = d, zUp = zf, zc = 0, hLow = h;                 // size of the highest block
    if (st == 5) {                                                      // terraced: top two floors are set back
        hLow = h - 2*hf; wUp = w - 4; dUp = d - 2; zc = -1; zUp = zc + dUp/2;
        box(P, V(0, hLow/2, 0), V(w, hLow, d), W, 3);
        box(P, V(0, hLow + 0.15f, 0), V(w + 0.5f, 0.3f, d + 0.5f), T, 3);
        box(P, V(0, hLow + hf, zc), V(wUp, 2*hf, dUp), W, 3);
    } else box(P, V(0, h/2, 0), V(w, h, d), W, 3);
    box(P, V(0, h + 0.2f, zc), V(wUp + 0.6f, 0.4f, dUp + 0.6f), T, 3);   // roof slab

    int cols = (st == 1) ? 6 : (st == 0 || st == 5 || st == 6) ? 5 : 4;
    float ww = st == 1 ? 1.2f : st == 2 ? 1.4f : 1.8f, wh = st == 2 ? 2.2f : 1.6f;
    for (int f = 0; f < floors - 1; f++) {
        bool up = (st == 5 && f >= floors - 2);
        float wi = up ? wUp : w, zfr = up ? zUp : zf, sp = wi / (up ? 4 : cols);
        for (int i = 0; i < (up ? 4 : cols); i++)
            box(P, V(-wi/2 + sp*(i + 0.5f), f*hf + hf*0.55f, zfr + 0.05f), V(ww, wh, 0.12f), GLASS, 8);
    }
    box(P, V(0, 1.4f, zf + 0.12f), V(3.2f, 2.8f, 0.2f), DARK);                    // door
    box(P, V(0, 0.15f, zf + 1.3f), V(6, 0.3f, 2.6f), CREAM, 3);                   // steps

    float ry0 = h + 0.4f;                                                         // roof walking level
    if (st == 0) {                                                                // KJA: full-width balconies + water tanks
        for (int f = 1; f < floors; f++) { box(P, V(0, f*hf, zf + 0.6f), V(w + 0.4f, 0.3f, 1.4f), T, 3); box(P, V(0, f*hf + 0.45f, zf + 1.25f), V(w + 0.4f, 0.7f, 0.08f), T, 3); }
        for (int sx = -1; sx <= 1; sx += 2) cyl(P * translateM(sx*5.0f, ry0 + 1.0f, -2.6f) * rotateX(PI/2), V(0,0,0), V(1.1f, 1.1f, 2.0f), V(0.3f,0.4f,0.55f), 7);
        drawClothesLine(P, -8, 8, ry0, 1.8f, 0); drawClothesLine(P, -8, 8, ry0, 0.2f, 1);
    } else if (st == 1) {                                                         // Rashid: brick band + tin shed on the roof
        box(P, V(0, hf/2, zf + 0.03f), V(w + 0.1f, hf, 0.06f), T, 3);
        for (int sx = -1; sx <= 1; sx += 2) for (int sz = 0; sz < 2; sz++) box(P, V(sx*6.0f, ry0 + 0.9f, -3.5f + sz*2.3f), V(0.15f, 1.8f, 0.15f), STEEL, 7);
        box(P * translateM(0, ry0 + 1.9f, -2.35f) * rotateX(0.12f), V(0, 0, 0), V(13.5f, 0.12f, 3.6f), V(0.45f,0.5f,0.55f), 7);
        drawClothesLine(P, -8, 8, ry0, 0.8f, 2); drawClothesLine(P, -8, 8, ry0, 2.4f, 3);
    } else if (st == 2) {                                                         // Shaheed Smriti: brick with cream piers + stair penthouse
        for (int i = 0; i < 5; i++) if (i != 2) box(P, V(-9 + i*4.5f, h/2, zf + 0.15f), V(0.8f, h, 0.5f), T, 3);
        box(P, V(6, ry0 + 1.2f, -1.5f), V(3.5f, 2.4f, 3.2f), W, 3); box(P, V(6, ry0 + 1.0f, 0.12f), V(1.2f, 2.0f, 0.1f), DARK);
        drawClothesLine(P, -8, 3, ry0, 1.8f, 4); drawClothesLine(P, -8, 3, ry0, 0.2f, 5);
    } else if (st == 3) {                                                         // Fazlul Haque: green bands, porch, crenellated roof
        for (int f = 1; f < floors; f++) box(P, V(0, f*hf, zf + 0.08f), V(w + 0.2f, 0.25f, 0.2f), T, 3);
        box(P, V(0, 3.4f, zf + 1.3f), V(7, 0.3f, 2.6f), T, 3);
        for (int sx = -1; sx <= 1; sx += 2) box(P, V(sx*3.0f, 1.7f, zf + 2.3f), V(0.4f, 3.4f, 0.4f), CREAM, 3);
        for (float x = -8.5f; x <= 8.6f; x += 1.7f) box(P, V(x, ry0 + 0.25f, zf - 0.15f), V(0.8f, 0.5f, 0.3f), T, 3);
        drawClothesLine(P, -8, 8, ry0, 1.0f, 6); drawClothesLine(P, -8, 8, ry0, -1.0f, 7);
    } else if (st == 4) {                                                         // Amar Ekushey: glass strip, solar panels, Shaheed Minar
        box(P, V(0, (hf + (floors - 1)*hf)/2, zf + 0.07f), V(2.4f, (floors - 2)*hf, 0.1f), GLASS, 8);
        for (int i = 0; i < 4; i++) box(P * translateM(-6.75f + i*4.5f, ry0 + 0.9f, -2.6f) * rotateX(-0.5f), V(0, 0, 0), V(3.6f, 0.08f, 1.6f), V(0.1f,0.15f,0.4f), 8);
        drawClothesLine(P, -8, 8, ry0, 1.8f, 8); drawClothesLine(P, -8, 8, ry0, 0.2f, 9);
        Mat4 Mn = P * translateM(6, 0, zf + 5.5f);
        box(Mn, V(0, 0.15f, 0), V(6, 0.3f, 2), CONCRETE, 2);
        ball(Mn, V(0, 2.8f, -0.5f), V(1.3f, 1.3f, 0.2f), V(0.8f,0.1f,0.1f));
        box(Mn, V(0, 2.0f, 0), V(0.5f, 3.4f, 0.3f), WHITE, 3);
        for (int sx = -1; sx <= 1; sx += 2) { box(Mn, V(sx*1.1f, 1.5f, 0), V(0.4f, 2.4f, 0.3f), WHITE, 3); box(Mn, V(sx*2.0f, 1.2f, 0), V(0.3f, 1.8f, 0.3f), WHITE, 3); }
    } else if (st == 5) {                                                         // Lalan Shah: terraces with planters, clothes on both roofs
        for (int f = 1; f < floors - 2; f++) { box(P, V(0, f*hf + 0.35f, zf + 0.5f), V(w - 2, 0.5f, 0.9f), T, 3); box(P, V(0, f*hf + 0.8f, zf + 0.5f), V(w - 2.4f, 0.4f, 0.7f), V(0.2f,0.5f,0.2f), 6); }
        drawClothesLine(P, -8, 8, hLow + 0.3f, 3.0f, 10); drawClothesLine(P, -5, 5, ry0, -0.5f, 11);
    }
    else {                                                                        // Rokeya Hall (girls): lavender, flower balconies, porch
        Vec3 fl[4] = {V(0.95f,0.3f,0.5f), V(0.95f,0.85f,0.2f), V(0.9f,0.9f,0.95f), V(0.85f,0.2f,0.2f)};
        for (int f = 1; f < floors; f++) {
            box(P, V(0, f*hf, zf + 0.6f), V(w + 0.4f, 0.3f, 1.4f), T, 3); box(P, V(0, f*hf + 0.4f, zf + 1.25f), V(w + 0.4f, 0.6f, 0.08f), T, 3);
            for (int i = 0; i < 9; i++) ball(P, V(-8 + i*2.0f, f*hf + 0.95f, zf + 1.05f), V(0.35f, 0.3f, 0.35f), fl[(i + f) % 4], 6);
        }
        box(P, V(0, 3.4f, zf + 1.3f), V(7, 0.3f, 2.6f), T, 3);
        for (int sx = -1; sx <= 1; sx += 2) box(P, V(sx*3.0f, 1.7f, zf + 2.3f), V(0.4f, 3.4f, 0.4f), CREAM, 3);
        drawClothesLine(P, -8, 8, ry0, 1.8f, 12); drawClothesLine(P, -8, 8, ry0, 0.2f, 13);
    }
    // name board on the top band of the highest block
    vector<string> lines = {name, "HALL"}; float px = 0.12f, gap = px*1.5f, maxW = 0;
    for (auto& l : lines) maxW = max(maxW, textWidth(l, px));
    float bw = maxW + 1.4f, bh = lines.size()*7*px + (lines.size() + 1)*gap, yc = (floors - 1)*hf + hf/2;
    box(P, V(0, yc, zUp + 0.08f), V(bw, bh, 0.16f), BOARD);
    for (size_t i = 0; i < lines.size(); i++) {
        float top = yc + bh/2 - gap - i*(7*px + gap);
        drawTextCentered(P, lines[i], 0, top - 7*px, zUp + 0.16f, px, 0.12f, WHITE);
    }
}


// ---------- flowering trees in front of every hall (each hall has its own kind)
void drawFlowerTree(float x, float z, int kind) {
    Vec3 fc[7] = {V(0.90f,0.25f,0.10f), V(0.95f,0.80f,0.12f), V(0.55f,0.30f,0.75f), V(0.95f,0.60f,0.70f), V(0.95f,0.45f,0.15f), V(0.98f,0.72f,0.20f), V(0.95f,0.95f,0.90f)};
    Vec3 c = fc[kind];
    cyl(I * translateM(x, 1.2f, z) * rotateX(PI/2), V(0,0,0), V(0.28f, 0.28f, 2.4f), V(0.38f,0.26f,0.16f), 5);
    if (kind == 0 || kind == 1 || kind == 4) {          // umbrella shaped (Krishnachura, Radhachura, Palash)
        ball(I, V(x, 3.4f, z), V(3.0f, 1.5f, 3.0f), c * 0.85f, 6); ball(I, V(x + 0.8f, 3.9f, z - 0.4f), V(2.0f, 1.1f, 2.0f), c, 6);
        ball(I, V(x - 1.0f, 3.7f, z + 0.6f), V(1.8f, 1.0f, 1.8f), c * 0.95f, 6);
    } else {                                           // round crown (Jarul, Kanchan, Kadam, Jasmine)
        ball(I, V(x, 3.7f, z), V(2.3f, 2.3f, 2.3f), c * 0.9f, 6); ball(I, V(x + 0.9f, 4.5f, z + 0.3f), V(1.6f, 1.5f, 1.6f), c, 6);
        ball(I, V(x - 0.9f, 4.2f, z - 0.5f), V(1.5f, 1.4f, 1.5f), c * 0.95f, 6);
    }
    for (int k = 0; k < 10; k++) {                      // flower clusters
        float a = k*2.4f + x, r = 1.3f + 0.9f*frac(sinf(k*7.1f + z)*43.5f), y = 3.3f + 1.1f*frac(sinf(k*3.3f + x)*91.7f);
        ball(I, V(x + r*cosf(a), y, z + r*sinf(a)), V(0.45f, 0.35f, 0.45f), c * 1.05f, 6);
    }
    for (int k = 0; k < 14; k++) {                      // fallen petals on the ground
        float a = k*1.9f, r = 3.0f*frac(sinf(k*5.7f + x)*77.7f) + 0.5f;
        ball(I, V(x + r*cosf(a), 0.05f, z + r*sinf(a)), V(0.18f, 0.03f, 0.18f), c * 0.9f);
    }
}
void drawHallFlowers() {
    float hz[6] = {22, -36, -60, -84, -108, -132};
    Vec3 bush[6] = {V(0.9f,0.25f,0.1f), V(0.95f,0.8f,0.15f), V(0.6f,0.35f,0.8f), V(0.95f,0.6f,0.7f), V(0.95f,0.45f,0.15f), V(0.98f,0.72f,0.2f)};
    for (int i = 0; i < 6; i++) {
        drawTree(-50.5f, hz[i] - 5.5f); drawTree(-50.5f, hz[i] + 5.5f);
        for (int k = 0; k < 4; k++) ball(I, V(-52.8f, 0.5f, hz[i] + (k < 2 ? -3.4f - k*1.1f : 3.4f + (k - 2)*1.1f)), V(0.6f, 0.5f, 0.6f), bush[i], 6);
    }
}
// ---------- Rokeya Hall: 3-storey girls' hall inside its own boundary wall, beside the Civil department
void drawRokeya() {
    drawHall(frame(66, -118, -PI/2), "ROKEYA", 3, 6);
    wallLine(46, -96, 85, -96); wallLine(46, -148, 85, -148); wallLine(46, -96, 46, -115); wallLine(46, -121, 46, -148);
    Mat4 Gt = frame(46, -118, -PI/2);
    for (int sx = -1; sx <= 1; sx += 2) box(Gt, V(sx*3.3f, 2.1f, 0), V(1.2f, 4.2f, 1.2f), V(0.94f,0.90f,0.82f), 3);
    box(Gt, V(0, 4.6f, 0), V(8.0f, 1.3f, 0.5f), BOARD);
    drawTextCentered(Gt, "ROKEYA HALL", 0, 4.0f, 0.3f, 0.12f, 0.1f, WHITE);
    Vec3 fl[4] = {V(0.95f,0.3f,0.5f), V(0.95f,0.85f,0.2f), V(0.9f,0.9f,0.95f), V(0.85f,0.2f,0.2f)};
    for (int i = 0; i < 8; i++) for (int sz = -1; sz <= 1; sz += 2) ball(I, V(50 + i*1.4f, 0.45f, -118 + sz*2.8f), V(0.5f, 0.45f, 0.5f), fl[(i + (sz > 0)) % 4], 6);
    drawTree(54, -106); drawTree(54, -130); drawTree(56, -142); drawTree(56, -100);
}
// ---------- couples chatting on the benches
void drawSeated(const Mat4& Q0, Vec3 shirt, Vec3 pants, Vec3 skin, Vec3 hair, float t, float ph, bool girl, float turn) {
    Mat4 Q = Q0 * scaleM(0.95f, 0.95f, 0.95f);
    for (int s = -1; s <= 1; s += 2) {
        box(Q, V(0.28f, 0.1f, 0.12f*s), V(0.55f, 0.18f, 0.18f), pants);
        box(Q, V(0.52f, -0.28f, 0.12f*s), V(0.18f, 0.62f, 0.18f), pants);
        box(Q, V(0.6f, -0.57f, 0.12f*s), V(0.3f, 0.1f, 0.2f), DARK);
    }
    box(Q, V(0, 0.5f, 0), V(0.3f, 0.7f, 0.5f), shirt);
    if (girl) box(Q, V(0.1f, 0.12f, 0), V(0.42f, 0.16f, 0.52f), pants);
    float nod = sinf(t*2.0f + ph) * 0.03f;
    ball(Q, V(0.02f, 1.0f + nod, turn*0.04f), V(0.17f, 0.19f, 0.17f), skin);
    ball(Q, V(-0.02f, 1.07f + nod, turn*0.04f), V(0.18f, 0.16f, 0.18f), hair);
    if (girl) box(Q, V(-0.14f, 0.75f, 0), V(0.1f, 0.6f, 0.3f), hair);
    Mat4 A1 = Q * translateM(0, 0.8f, 0.32f * turn) * rotateZ(0.9f + 0.55f*sinf(t*2.6f + ph));   // the arm that talks (towards the partner)
    box(A1, V(0, -0.32f, 0), V(0.15f, 0.64f, 0.15f), shirt); ball(A1, V(0, -0.66f, 0), V(0.08f, 0.08f, 0.08f), skin);
    Mat4 A2 = Q * translateM(0, 0.8f, -0.32f * turn) * rotateZ(1.25f);
    box(A2, V(0, -0.32f, 0), V(0.15f, 0.64f, 0.15f), shirt); ball(A2, V(0, -0.66f, 0), V(0.08f, 0.08f, 0.08f), skin);
}
void drawBenchPeople() {
    float t = gTime;
    for (const BenchSpot& b : benches) {
        if (b.kind > 1) continue;
        Mat4 B = frame(b.x, b.z, b.rot, b.y); float ph = b.x*0.37f + b.z*0.21f;
        Mat4 Q1 = B * translateM(-0.5f, 0.61f, -0.2f) * rotateY(-PI/2), Q2 = B * translateM(0.5f, 0.61f, -0.2f) * rotateY(-PI/2);
        Vec3 sk = V(0.8f,0.58f,0.42f), hr = V(0.06f,0.05f,0.05f);
        drawSeated(Q1, V(0.25f,0.45f,0.85f), V(0.15f,0.15f,0.25f), sk, hr, t, ph, false, -1);
        if (b.kind == 0) drawSeated(Q2, V(0.95f,0.45f,0.65f), V(0.95f,0.9f,0.8f), V(0.84f,0.62f,0.48f), hr, t, ph + 1.7f, true, 1);
        else             drawSeated(Q2, V(0.85f,0.75f,0.2f), V(0.3f,0.3f,0.35f), sk, hr, t, ph + 1.7f, false, 1);
    }
}
// ---------- people bathing in the pond (the part under the water is hidden by the water surface)
void drawBathers() {
    float t = gTime, cx = (PX0 + PX1)/2, cz = (PZ0 + PZ1)/2;
    Vec3 sk[3] = {V(0.78f,0.55f,0.4f), V(0.7f,0.5f,0.36f), V(0.85f,0.62f,0.46f)}, hr = V(0.05f,0.04f,0.04f);
    for (int i = 0; i < 3; i++) {
        float a = t*0.15f + i*2.1f, x = cx + 6*cosf(a) + i*1.5f, z = cz + 5*sinf(a*0.8f);
        float dunk = max(0.0f, sinf(t*0.5f + i*2.0f)) * 0.9f;
        drawPerson(frame(x, z, headingOf(-sinf(a), cosf(a)*0.8f), -1.3f - dunk), sk[i], V(0.1f,0.1f,0.3f), sk[i], hr, t*3.5f + i, 1.4f, 1.0f);
        for (int k = 0; k < 5; k++) {                   // splashes
            float u = frac(t*1.3f + k*0.2f + i*0.31f), r = 0.2f + 0.7f*u;
            ball(I, V(x + r*cosf(k*2.1f + i), -0.4f + 1.1f*sinf(u*PI), z + r*sinf(k*2.1f + i)), V(0.07f, 0.07f, 0.07f)*(1.0f - 0.5f*u), V(0.85f,0.95f,1.0f));
        }
    }
    float u = frac(t / 8.0f), lx = PX1 + 0.5f, lz = cz + 6;                                  // a boy jumps in from the edge
    Vec3 skin = V(0.76f,0.54f,0.4f);
    if (u > 0.9f)      drawPerson(frame(lx, lz, PI, 0.4f), skin, V(0.8f,0.2f,0.2f), skin, hr, 0, 0, 1.0f);
    else if (u < 0.15f) { float v = u / 0.15f; drawPerson(frame(lx - 5*v, lz, PI, 0.4f + 3.0f*sinf(PI*v) - 1.6f*v), skin, V(0.8f,0.2f,0.2f), skin, hr, t*8, 1.3f, 1.0f); }
    else if (u < 0.4f) for (int k = 0; k < 8; k++) { float w = (u - 0.15f) / 0.25f, r = 0.4f + 1.2f*w; ball(I, V(lx - 5 + r*cosf(k*0.8f), -0.4f + 1.6f*sinf(w*PI)*(0.4f + 0.6f*frac(k*0.37f)), lz + r*sinf(k*0.8f)), V(0.1f, 0.1f, 0.1f), V(0.85f,0.95f,1.0f)); }
    else if (u < 0.75f) drawPerson(frame(lx - 5, lz, PI, -1.35f), skin, V(0.8f,0.2f,0.2f), skin, hr, t*4, 1.5f, 1.0f);
}
Mat4 RF(float z) { return frame(70, z, -PI/2); }          // right-side buildings face -x (towards the road)
void drawAllBuildings() {
    struct H { const char* n; float z; int floors; } halls[6] = {{"KHAN JAHAN ALI", 22, 5}, {"DR M A RASHID", -36, 3}, {"SHAHEED SMRITI", -60, 5},
                                                              {"FAZLUL HAQUE", -84, 3}, {"AMAR EKUSHEY", -108, 5}, {"LALAN SHAH", -132, 5}};
    for (int i = 0; i < 6; i++) drawHall(frame(-60, halls[i].z, PI/2), halls[i].n, halls[i].floors, i);
    drawBuilding(RF(72), 26, 9, 12, 3, {"CSE BUILDING"}, 0.16f);
    drawBuilding(RF(36), 26, 9, 12, 3, {"EEE BUILDING"}, 0.16f, true);
    drawEEEUnder(RF(36), 26, 9, 4);
    drawBuilding(RF(2), 24, 10, 20, 5, {"CENTRAL ACADEMIC", "BUILDING"}, 0.16f);
    drawBuilding(RF(-34), 26, 9, 12, 3, {"MECHANICAL ENGINEERING", "BUILDING"}, 0.16f);
    drawBuilding(RF(-70), 26, 9, 12, 3, {"CIVIL ENGINEERING", "BUILDING"}, 0.16f);
    drawEEEYard(RF(36)); drawMechYard(RF(-34)); drawCivilYard(RF(-70));
    drawHallFlowers(); drawRokeya();
}

// ============================================================================
//  SEGMENT 20b : LAMP-POST SHADOW MAPS (baked once; the light of every lamp is blocked by trees / buildings)
// ============================================================================
struct Lamp { Vec3 pos; Mat4 vp; };
vector<Lamp> lamps; GLuint lampTex = 0, lampFBO = 0; const int LAMP_RES = 384;
GLuint lampPosTex = 0, lampMatTex = 0, lampGridTex = 0, floodTex = 0, floodFBO = 0; const int FLOOD_RES = 1024;
const float GX0 = -110, GZ0 = -165, GCELL = 12; const int GCOLS = 18, GROWS = 26, GK = 10;
struct Flood { Vec3 pos; Mat4 vp; } floods[4];
float spotPosA[24], spotDirA[24], spotColA[24], spotConeA[16]; int nSpotsTotal = 0;
void addSpot(Vec3 p, Vec3 target, Vec3 col, float cosOuter, float cosInner) {
    Vec3 d = normalize(target - p); int i = nSpotsTotal++;
    spotPosA[i*3] = p.x; spotPosA[i*3 + 1] = p.y; spotPosA[i*3 + 2] = p.z;
    spotDirA[i*3] = d.x; spotDirA[i*3 + 1] = d.y; spotDirA[i*3 + 2] = d.z;
    spotColA[i*3] = col.x; spotColA[i*3 + 1] = col.y; spotColA[i*3 + 2] = col.z;
    spotConeA[i*2] = cosOuter; spotConeA[i*2 + 1] = cosInner;
}
void renderFloodOccluders() { drawFootball(); goal(-1); goal(1); drawFloodMasts(); }
void initFloods() {
    for (int i = 0; i < 4; i++) {
        float mx = (i % 2) ? 33.f : -33.f, mz = FZ + (i < 2 ? -25.f : 25.f);
        Vec3 toC = normalize(V(-mx, 0, FZ - mz)), eye = V(mx, 23.7f, mz) + toC * 1.5f;
        floods[i].pos = eye; floods[i].vp = perspectiveM(75.0f*PI/180, 1.0f, 2.0f, 160.0f) * lookAtM(eye, V(0, 0, FZ), V(0, 1, 0));
    }
    glGenTextures(1, &floodTex); glBindTexture(GL_TEXTURE_2D_ARRAY, floodTex);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT24, FLOOD_RES, FLOOD_RES, 4, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER); glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float b[4] = {1, 1, 1, 1}; glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, b);
    glGenFramebuffers(1, &floodFBO); glBindFramebuffer(GL_FRAMEBUFFER, floodFBO); glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
void renderFloodShadows() {                        // live every frame at night (players, ball, goals)
    glBindFramebuffer(GL_FRAMEBUFFER, floodFBO); glViewport(0, 0, FLOOD_RES, FLOOD_RES);
    glUseProgram(depthProg); glUniform1f(dTime, gTime);
    gModel = dModel; gMat = dMat; gColor = -1; shadowPass = true; lampPass = true;
    glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(2.0f, 4.0f);
    for (int i = 0; i < 4; i++) {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, floodTex, 0, i); glClear(GL_DEPTH_BUFFER_BIT);
        glUniformMatrix4fv(dLightVP, 1, GL_FALSE, floods[i].vp.m);
        cullC = V(0, 0, FZ); cullR = 60; curMat = -1; renderFloodOccluders();
    }
    glDisable(GL_POLYGON_OFFSET_FILL); lampPass = false;
}
void renderLampOccluders() { drawTreesAndLamps(); drawWalls(); drawGate(); drawHill(); drawAllBuildings(); drawMosque(); }
void initLampShadows() {
    for (const Item& it : scenery) if (!it.tree) {
        Vec3 p = V(it.x + it.ax*1.7f, 7.1f, it.z + it.az*1.7f);
        lamps.push_back({p, perspectiveM(150.0f*PI/180, 1.0f, 0.3f, 45.0f) * lookAtM(p, p + V(0, -1, 0), V(0, 0, 1))});
    }
    int n = (int)lamps.size();
    glGenTextures(1, &lampTex); glBindTexture(GL_TEXTURE_2D_ARRAY, lampTex);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT24, LAMP_RES, LAMP_RES, n, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER); glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float b[4] = {1, 1, 1, 1}; glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, b);
    glGenFramebuffers(1, &lampFBO); glBindFramebuffer(GL_FRAMEBUFFER, lampFBO);
    glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE);
    glUseProgram(depthProg); glUniform1f(dTime, 0.0f);
    gModel = dModel; gMat = dMat; gColor = -1; shadowPass = true; lampPass = true;
    glViewport(0, 0, LAMP_RES, LAMP_RES); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(2.0f, 4.0f);
    for (int i = 0; i < n; i++) {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, lampTex, 0, i);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUniformMatrix4fv(dLightVP, 1, GL_FALSE, lamps[i].vp.m);
        cullC = lamps[i].pos; cullR = 30; curMat = -1;
        renderLampOccluders();
    }
    glDisable(GL_POLYGON_OFFSET_FILL); lampPass = false; shadowPass = false;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // ---- lamp data textures: position, shadow matrix and a grid (each 12x12 cell -> its 10 nearest lamps) so ALL lamps light up everywhere
    vector<float> lpos, lmat, grid((size_t)GCOLS * GROWS * GK, -1.0f);
    for (const Lamp& L : lamps) { lpos.push_back(L.pos.x); lpos.push_back(L.pos.y); lpos.push_back(L.pos.z); for (int k = 0; k < 16; k++) lmat.push_back(L.vp.m[k]); }
    for (int cz = 0; cz < GROWS; cz++) for (int cx = 0; cx < GCOLS; cx++) {
        float px = GX0 + (cx + 0.5f) * GCELL, pz = GZ0 + (cz + 0.5f) * GCELL;
        vector<pair<float, int>> sc;
        for (int i = 0; i < n; i++) { float dx = lamps[i].pos.x - px, dz = lamps[i].pos.z - pz, d2 = dx*dx + dz*dz; if (d2 < 46.0f*46.0f) sc.push_back({d2, i}); }
        sort(sc.begin(), sc.end());
        for (int k = 0; k < GK && k < (int)sc.size(); k++) grid[((size_t)cz * GCOLS + cx) * GK + k] = (float)sc[k].second;
    }
    auto mk = [&](GLuint& t, GLint fmt, GLenum fm, int w, int h, const float* d) {
        glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, fm, GL_FLOAT, d);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    };
    mk(lampPosTex, GL_RGB32F, GL_RGB, n, 1, lpos.data()); mk(lampMatTex, GL_RGBA32F, GL_RGBA, n * 4, 1, lmat.data());
    mk(lampGridTex, GL_R32F, GL_RED, GK, GCOLS * GROWS, grid.data());
    initFloods();
    for (int k = 0; k < 4; k++) addSpot(HILL_SPOTS[k] + V(0, 0.5f, 0), V(-64, 6, 64), V(1.0f, 0.9f, 0.7f) * 3.2f, 0.72f, 0.93f);          // KUET hill
    for (int k = 0; k < 4; k++) addSpot(V(POND_POSTS[k][0], 6.4f, POND_POSTS[k][1]), V((PX0 + PX1)/2, -0.4f, (PZ0 + PZ1)/2), V(0.8f, 0.95f, 1.0f) * 6.0f, 0.55f, 0.85f);   // pond
    printf("Baked %d lamp-post shadow maps, %d ground spots, 4 stadium floodlights\n", n, nSpotsTotal);
}

void renderScene() {
    drawGround(); drawRoads(); drawWalls(); drawTreesAndLamps(); drawGate(); drawHill();
    drawAllBuildings(); drawMosque(); drawField(); drawPond();
    drawVehicles(); drawKids(); drawFootball(); drawBenchPeople(); drawBathers();
    if (!shadowPass) { drawBirds(); drawClouds(); }
}

// ============================================================================
//  SEGMENT 21 : CAMERA + INPUT
// ============================================================================
Vec3 camPos = {0, 6, 128};
float yawDeg = -90, pitchDeg = -4;
double lastX = 640, lastY = 360; bool firstMouse = true;
int winW = 1280, winH = 720;

Vec3 camFront() {
    float y = yawDeg * PI / 180, p = pitchDeg * PI / 180;
    return normalize(V(cosf(p) * cosf(y), sinf(p), cosf(p) * sinf(y)));
}
void mouseCallback(GLFWwindow*, double xp, double yp) {
    if (firstMouse) { lastX = xp; lastY = yp; firstMouse = false; }
    yawDeg += (float)(xp - lastX) * 0.12f; pitchDeg -= (float)(yp - lastY) * 0.12f;
    pitchDeg = max(-89.f, min(89.f, pitchDeg));
    lastX = xp; lastY = yp;
}
void resizeCallback(GLFWwindow*, int w, int h) { winW = w; winH = h; }
void keyCallback(GLFWwindow*, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_L) { lightsOn = !lightsOn; printf("EEE lights: %s\n", lightsOn ? "ON" : "OFF"); }
    if (key == GLFW_KEY_F) { rigFan.on = !rigFan.on; printf("EEE fans: %s\n", rigFan.on ? "ON" : "OFF"); }
    if (key == GLFW_KEY_G) { rigEEE.on = !rigEEE.on; printf("EEE generator + wind turbine: %s\n", rigEEE.on ? "ON" : "OFF"); }
    if (key == GLFW_KEY_M) { rigMech.on = !rigMech.on; printf("Mechanical machines: %s\n", rigMech.on ? "ON" : "OFF"); }
    if (key == GLFW_KEY_C) { rigCivil.on = !rigCivil.on; printf("Construction machines: %s\n", rigCivil.on ? "ON" : "OFF"); }
    if (key == GLFW_KEY_N) { nightMode = !nightMode; printf("%s\n", nightMode ? "NIGHT: lamp posts are shining" : "DAY"); }
}
void processInput(GLFWwindow* win, float dt) {
    if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, true);
    float speed = 18.0f * dt * (glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 3.0f : 1.0f);
    Vec3 f = camFront(), r = normalize(cross(f, V(0, 1, 0)));
    if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS) camPos = camPos + f * speed;
    if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS) camPos = camPos - f * speed;
    if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS) camPos = camPos + r * speed;
    if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS) camPos = camPos - r * speed;
    if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS) camPos.y += speed;
    if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS) camPos.y -= speed;
    if (camPos.y < 1) camPos.y = 1;
}

// ============================================================================
//  SEGMENT 22 : MAIN  (window + shadow pass + render pass)
// ============================================================================
Vec3 moonDir() { return normalize(V(-0.3f, 0.75f, 0.5f)); }
void drawSky() {                                  // moon + stars (night only, follow the camera)
    ball(I, camPos + moonDir() * 420, V(16, 16, 16), V(0.95f,0.95f,0.85f), 10);
    for (int i = 0; i < 150; i++) {
        float a = frac(sinf(i*12.9898f) * 43758.5f), b = frac(sinf(i*78.233f) * 24634.6f), c = frac(sinf(i*39.346f) * 91827.3f);
        box(I, camPos + normalize(V(a - 0.5f, 0.12f + 0.88f*b, c - 0.5f)) * 480, V(1.2f, 1.2f, 1.2f), V(0.9f,0.9f,1.0f), 10);
    }
}

int main() {
    if (!glfwInit()) { printf("GLFW init failed\n"); return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    const GLFWvidmode* vm = glfwGetVideoMode(mon);
    winW = vm->width; winH = vm->height;
    GLFWwindow* window = glfwCreateWindow(winW, winH, "KUET Campus - OpenGL (v4)", mon, nullptr);
    if (!window) { printf("Window creation failed\n"); glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetFramebufferSizeCallback(window, resizeCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSwapInterval(1);
    if (!gladLoadGL(glfwGetProcAddress)) { printf("GLAD failed\n"); return -1; }
    glfwGetFramebufferSize(window, &winW, &winH);

    glEnable(GL_DEPTH_TEST); glEnable(GL_MULTISAMPLE);
    prog = createProgram(VS_SRC, FS_SRC);
    depthProg = createProgram(DVS_SRC, DFS_SRC);
    uModel = glGetUniformLocation(prog, "model");   uView = glGetUniformLocation(prog, "view");
    uProj = glGetUniformLocation(prog, "proj");     uColor = glGetUniformLocation(prog, "color");
    uLight = glGetUniformLocation(prog, "lightDir"); uCam = glGetUniformLocation(prog, "camPos");
    uLightVP = glGetUniformLocation(prog, "lightVP"); uTime = glGetUniformLocation(prog, "time");
    uMat = glGetUniformLocation(prog, "matId");
    dModel = glGetUniformLocation(depthProg, "model"); dLightVP = glGetUniformLocation(depthProg, "lightVP");
    dTime = glGetUniformLocation(depthProg, "time");   dMat = glGetUniformLocation(depthProg, "matId");
    uRoomOn = glGetUniformLocation(prog, "roomOn"); uRoomPos = glGetUniformLocation(prog, "roomPos"); uRoomBox = glGetUniformLocation(prog, "roomBox");
    uGridInfo = glGetUniformLocation(prog, "gridInfo"); uGridDim = glGetUniformLocation(prog, "gridDim");
    uNFlood = glGetUniformLocation(prog, "nFlood"); uFloodPos = glGetUniformLocation(prog, "floodPos"); uFloodVP = glGetUniformLocation(prog, "floodVP");
    uNSpots = glGetUniformLocation(prog, "nSpots"); uSpotPos = glGetUniformLocation(prog, "spotPos"); uSpotDir = glGetUniformLocation(prog, "spotDir");
    uSpotCol = glGetUniformLocation(prog, "spotCol"); uSpotCone = glGetUniformLocation(prog, "spotCone");
    uNight = glGetUniformLocation(prog, "night"); uSky = glGetUniformLocation(prog, "skyColor");

    cubeMesh = buildCube(); sphereMesh = buildSphere(); cylMesh = buildCylinder();
    initShadow(); initRoads(); initScenery(); initTraffic(); initLampShadows();
    rigFan.on = true; rigCivil.on = true;
    printf("L = EEE lights | F = EEE fans | G = EEE generator/turbine | M = Mechanical machines | C = Construction machines | N = day/night\n");

    Vec3 sunDir = normalize(V(0.2f, 0.7f, 0.55f));                    // direction TOWARDS the sun
    Vec3 lc = V(0, 0, -10);
    Mat4 lproj = orthoM(-175, 175, -175, 175, 10, 400);

    double last = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime(); float dt = (float)(now - last); last = now; gTime = (float)now;
        processInput(window, dt);
        updateTraffic(dt);
        rigFan.update(dt); rigEEE.update(dt); rigMech.update(dt); rigCivil.update(dt);
        Vec3 sun = nightMode ? moonDir() : sunDir;                       // the moon is the directional light at night
        Mat4 lightVP = lproj * lookAtM(lc + sun * 200, lc, V(0, 1, 0));
        Vec3 skyC = nightMode ? V(0.02f, 0.03f, 0.09f) : V(0.53f, 0.81f, 0.92f);

        // ---------- pass 1 : shadow map ----------
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
        glViewport(0, 0, SHADOW_RES, SHADOW_RES);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(depthProg);
        glUniformMatrix4fv(dLightVP, 1, GL_FALSE, lightVP.m); glUniform1f(dTime, gTime);
        gModel = dModel; gMat = dMat; gColor = -1; shadowPass = true; curMat = -1;
        glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(2.0f, 4.0f);
        renderScene();
        glDisable(GL_POLYGON_OFFSET_FILL);
        if (nightMode) renderFloodShadows();

        // ---------- pass 2 : real picture ----------
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, winW, winH);
        glClearColor(skyC.x, skyC.y, skyC.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(prog);
        Mat4 view = lookAtM(camPos, camPos + camFront(), V(0, 1, 0));
        Mat4 proj = perspectiveM(60.0f * PI / 180, (float)winW / (float)winH, 0.3f, 600.0f);
        glUniformMatrix4fv(uView, 1, GL_FALSE, view.m); glUniformMatrix4fv(uProj, 1, GL_FALSE, proj.m);
        glUniformMatrix4fv(uLightVP, 1, GL_FALSE, lightVP.m);
        glUniform3f(uLight, sun.x, sun.y, sun.z); glUniform3f(uCam, camPos.x, camPos.y, camPos.z);
        glUniform1f(uTime, gTime);
        glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, lampGridTex);
        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, lampMatTex);
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, lampPosTex);
        glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D_ARRAY, floodTex);
        glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D_ARRAY, lampTex);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, shadowTex);
        glUniform1i(glGetUniformLocation(prog, "shadowMap"), 0);   glUniform1i(glGetUniformLocation(prog, "lampShadow"), 1);
        glUniform1i(glGetUniformLocation(prog, "floodShadow"), 2); glUniform1i(glGetUniformLocation(prog, "lampPosTex"), 3);
        glUniform1i(glGetUniformLocation(prog, "lampMatTex"), 4);  glUniform1i(glGetUniformLocation(prog, "lampGridTex"), 5);
        glUniform1f(uNight, nightMode ? 1.0f : 0.0f); glUniform3f(uSky, skyC.x, skyC.y, skyC.z);
        glUniform4f(uGridInfo, GX0, GZ0, GCELL, 0.0f); glUniform2i(uGridDim, GCOLS, GROWS);
        {   // EEE tube lights really light the ground floor (EEE frame: world = (70 - zl, y, 36 + xl))
            float rp[24]; int k = 0;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 2; j++) { float xl = -7.5f + i*5.0f, zl = j == 0 ? 2.2f : -2.4f; rp[k*3] = 70 - zl; rp[k*3 + 1] = 3.8f; rp[k*3 + 2] = 36 + xl; k++; }
            glUniform1f(uRoomOn, lightsOn ? 1.0f : 0.0f); glUniform3fv(uRoomPos, 8, rp); glUniform4f(uRoomBox, 63.5f, 74.5f, 22.5f, 49.5f);
        }
        glUniform1i(uNFlood, nightMode ? 4 : 0); glUniform1i(uNSpots, nightMode ? nSpotsTotal : 0);
        if (nightMode) {
            float fp[12], fv[64];
            for (int i = 0; i < 4; i++) { fp[i*3] = floods[i].pos.x; fp[i*3 + 1] = floods[i].pos.y; fp[i*3 + 2] = floods[i].pos.z; for (int k = 0; k < 16; k++) fv[i*16 + k] = floods[i].vp.m[k]; }
            glUniform3fv(uFloodPos, 4, fp); glUniformMatrix4fv(uFloodVP, 4, GL_FALSE, fv);
            glUniform3fv(uSpotPos, nSpotsTotal, spotPosA); glUniform3fv(uSpotDir, nSpotsTotal, spotDirA);
            glUniform3fv(uSpotCol, nSpotsTotal, spotColA); glUniform2fv(uSpotCone, nSpotsTotal, spotConeA);
        }
        gModel = uModel; gMat = uMat; gColor = uColor; shadowPass = false; curMat = -1;
        renderScene();
        if (nightMode) drawSky();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}