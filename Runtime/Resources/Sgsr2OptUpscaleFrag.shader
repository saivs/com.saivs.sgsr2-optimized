// Snapdragon Game Super Resolution 2 — fully fused fragment variant.
// The upscale IS the final blit: one fullscreen pass reads raw scene color
// and depth, does the convert-pass math inline (tonemap+YCoCg per tap,
// dilated-depth motion reconstruction), rectifies against history and
// writes MRT0 = backbuffer, MRT1 = new history. No intermediate buffers.
//
// Derived from github.com/SnapdragonGameStudios/snapdragon-gsr (BSD-3):
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
Shader "Hidden/Sgsr2Optimized/UpscaleFrag"
{
    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" }
        Pass
        {
            Name "SGSR2 Fused Upscale"
            ZTest Always ZWrite Off Cull Off Blend Off

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma multi_compile_fragment __ SGSR_FP16
            #pragma multi_compile_fragment __ SGSR_LITE
            #pragma multi_compile_fragment __ SGSR_PACKED
            #pragma multi_compile_fragment __ SGSR_ULTRA
            #pragma multi_compile_fragment __ SGSR_PASSTHROUGH
            // Stillness as a compile variant, not a uniform branch: branches
            // around fetches measurably cost registers and fetch scheduling
            // on Adreno. The C# side flips this keyword only when the camera
            // starts or stops moving.
            #pragma multi_compile_fragment __ SGSR_STILL
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"

            #if defined(SGSR_FP16)
            #define hfloat  min16float
            #define hfloat2 min16float2
            #define hfloat3 min16float3
            #define hfloat4 min16float4
            #else
            #define hfloat  float
            #define hfloat2 float2
            #define hfloat3 float3
            #define hfloat4 float4
            #endif

            Texture2D<float4> _SgsrColor;
            Texture2D<float>  _SgsrDepth;
            Texture2D<float4> _SgsrPrevHistory;
            Texture2D<uint>   _SgsrPacked;
            // Ultra guide texture, one point Load per display pixel:
            // x = motion (f16 pair), y = depthclip | avgLuma (f16 pair),
            // z = rectbox min, w = rectbox max (11/11/10 biased).
            // A split RG32+R32 layout with branched fetches measured SLOWER
            // on Adreno 610: branch-free scheduling and fetch count beat
            // fetch width there.
            Texture2D<uint4>  _SgsrGuideBox;

            SamplerState sgsr_linear_clamp_sampler;
            SamplerState sgsr_point_clamp_sampler;

            float4 _RenderSizeInfo;   // xy = render size, zw = 1 / render size
            float4 _DisplaySizeInfo;  // xy = display size, zw = 1 / display size
            float4 _JitterOffset;     // xy = jitter in render pixels
            float4 _SgsrParams;       // x = preExposure, y = cameraFovAngleHor, z = minLerpContribution, w = unused
            float4 _ClipToPrevClip0;
            float4 _ClipToPrevClip1;
            float4 _ClipToPrevClip2;
            float4 _ClipToPrevClip3;
            float4 _BlitScaleBias;    // output-UV -> input-UV (final-blit convention)
            float  _SameCamera;
            float  _Reset;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(uint vertexID : SV_VertexID)
            {
                Varyings o;
                o.positionCS = GetFullScreenTriangleVertexPosition(vertexID);
                o.uv = GetFullScreenTriangleTexCoord(vertexID);
                return o;
            }

            struct FragOut
            {
                half4 color : SV_Target0;   // backbuffer
                float4 history : SV_Target1; // new history (ycocg + Wfactor)
            };

            hfloat FastLanczos(hfloat base)
            {
                hfloat y = base - hfloat(1.0);
                hfloat y2 = y * y;
                hfloat y_temp = hfloat(0.75) * y + y2;
                return y_temp * y2;
            }

            // Convert-pass color math, inline per tap: tonemap + YCoCg. The
            // saturate+bias pair mirrors the convert pass's unsigned packing,
            // then the 0.5 biases are removed — the upscale math (and the
            // history) work in the centered space that DecodeColor produces
            // on the compute path.
            hfloat3 FetchYCoCg(int2 pos, out float colorMax)
            {
                pos = clamp(pos, int2(0, 0), int2(_RenderSizeInfo.xy) - 1);
                float3 c = _SgsrColor.Load(int3(pos, 0)).xyz;
                colorMax = max(max(c.x, c.y), c.z) + _SgsrParams.x;
                hfloat3 rgb = (hfloat3)(c / colorMax);
                hfloat y = hfloat(0.25) * (rgb.x + hfloat(2.0) * rgb.y + rgb.z);
                hfloat coBiased = saturate(hfloat(0.5) * rgb.x + hfloat(0.5) - hfloat(0.5) * rgb.z);
                hfloat cgBiased = saturate(y + coBiased - rgb.x);
                return hfloat3(y, coBiased - hfloat(0.5), cgBiased - hfloat(0.5));
            }

            // Decode of the pack-pass output (matches the compute-path layout).
            hfloat3 DecodeColor(uint sample32)
            {
                uint x11 = sample32 >> 21u;
                uint y11 = sample32 & (2047u << 10u);
                uint z10 = sample32 & 1023u;
                hfloat3 samplecolor;
                samplecolor.x = hfloat(float(x11) * (1.0 / 2047.5));
                samplecolor.y = hfloat((float(y11) * (4.76953602e-7)) - 0.5);
                samplecolor.z = hfloat((float(z10) * (1.0 / 1023.5)) - 0.5);
                return samplecolor;
            }

            #define BOX_SAMPLE(colorexpr, offsx, offsy) \
                { \
                    hfloat3 samplecolor = colorexpr; \
                    hfloat2 baseoffset = srcDelta + hfloat2(offsx, offsy); \
                    hfloat baseoffset_dot = dot(baseoffset, baseoffset); \
                    hfloat base = saturate(baseoffset_dot * hKernelbias2); \
                    hfloat weight = FastLanczos(base); \
                    Upsampledcw += hfloat4(samplecolor * weight, weight); \
                    hfloat boxweight = exp(baseoffset_dot * hCurvebias); \
                    rectboxmin = min(rectboxmin, samplecolor); \
                    rectboxmax = max(rectboxmax, samplecolor); \
                    hfloat3 wsample = samplecolor * boxweight; \
                    rectboxcenter += wsample; \
                    rectboxvar += (samplecolor * wsample); \
                    rectboxweight += boxweight; \
                }

            // Ultra moving-camera tap: Lanczos accumulation only. The box
            // statistics BOX_SAMPLE gathers per tap already sit precomputed
            // in the guide, so accumulating them here would be paying twice.
            #define ULTRA_TAP(colorexpr, offsx, offsy) \
                { \
                    hfloat3 samplecolor = colorexpr; \
                    hfloat2 baseoffset = srcDelta + hfloat2(offsx, offsy); \
                    hfloat baseoffset_dot = dot(baseoffset, baseoffset); \
                    hfloat base = saturate(baseoffset_dot * hKernelbias2); \
                    hfloat weight = FastLanczos(base); \
                    Upsampledcw += hfloat4(samplecolor * weight, weight); \
                }

            FragOut Frag(Varyings i)
            {
                float2 renderSize = _RenderSizeInfo.xy;
                float2 renderSizeRcp = _RenderSizeInfo.zw;
                float2 displaySize = _DisplaySizeInfo.xy;

                // Output-orientation UV vs input-orientation UV: the inputs
                // (color/depth) may be flipped relative to the backbuffer we
                // render into; history lives in output orientation.
                float2 HruvOut = i.uv;
                float2 HruvIn = i.uv * _BlitScaleBias.xy + float2(0.0, _BlitScaleBias.w);
                float motionFlipY = _BlitScaleBias.y < 0.0 ? -1.0 : 1.0;

                float2 InputJitter = _JitterOffset.xy;
                float2 Jitteruv;
                Jitteruv.x = saturate(HruvIn.x + (InputJitter.x * renderSizeRcp.x));
                Jitteruv.y = saturate(HruvIn.y + (InputJitter.y * renderSizeRcp.y));

#if defined(SGSR_PASSTHROUGH)
                // Floor measurement: identical pass structure (both MRTs
                // attached and stored), but the shader collapses to the
                // theoretical minimum of any upscaler — one jitter-compensated
                // bilinear tap into the backbuffer. Ultra minus this number
                // is everything a smarter kernel could still win back.
                FragOut po;
                po.color = half4(_SgsrColor.SampleLevel(sgsr_linear_clamp_sampler, Jitteruv, 0.0).xyz, 1.0);
                po.history = float4(0.0, 0.0, 0.0, 0.0);
                return po;
#endif
                // Clamped: at the right/bottom edge saturate(Jitteruv) hits
                // exactly 1.0 and the truncated index lands one texel out of
                // range — an OOB Load returns 0, which decodes to a blue-ish
                // YCoCg box (the "purple edge line").
                int2 InputPos = min(int2(Jitteruv * renderSize), int2(renderSize) - 1);

                // ---- convert-pass math, inline (per display pixel) --------
                float2 gatherCoord = float2(InputPos) * renderSizeRcp;
                float2 ViewportUV = gatherCoord + 0.5 * renderSizeRcp;

#if defined(SGSR_ULTRA)
                // Ultra: everything precomputed per RENDER pixel arrives in
                // ONE point Load — motion, depthclip, avg luma and the box.
                uint4 gb = _SgsrGuideBox.Load(int3(InputPos, 0));
#if defined(SGSR_STILL)
                // Still: motion is identically zero — its guide slot carries
                // the center color instead (see the tap section below).
                float2 Motion = float2(0.0, 0.0);
#else
                float2 Motion = float2(f16tof32(gb.x & 0xFFFFu), f16tof32(gb.x >> 16u));
#endif
                float depthclip = f16tof32(gb.y & 0xFFFFu);
                hfloat ultraAvgY = (hfloat)f16tof32(gb.y >> 16u);
#elif defined(SGSR_LITE)
                // Lite (low-tier GPUs): 2x2 dilation from one gather, no
                // depth-clip disocclusion weighting. 4 depth texels vs 16.
                float4 topleftD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord);
                float topLeftMax9 = max(max(max(topleftD.y, topleftD.x), topleftD.z), topleftD.w);
                float depthclip = 0.0;
#else
                float4 topleftD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord);
                float4 topRightD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(renderSizeRcp.x * 2.0, 0.0));
                float4 bottomLeftD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(0.0, renderSizeRcp.y * 2.0));
                float4 bottomRightD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(renderSizeRcp.x * 2.0, renderSizeRcp.y * 2.0));

                // Reversed-Z: nearest = max, far plane = 0, nearness = raw depth.
                float maxC = max(max(max(topleftD.y, topRightD.x), bottomLeftD.z), bottomRightD.w);
                float topleft4 = max(max(max(topleftD.y, topleftD.x), topleftD.z), topleftD.w);
                float topLeftMax9 = max(bottomLeftD.w, max(max(maxC, topleft4), topRightD.w));

                float depthclip = 0.0;
                if (maxC > 1.0e-05)
                {
                    float topRight4 = max(max(max(topRightD.y, topRightD.x), topRightD.z), topRightD.w);
                    float bottomLeft4 = max(max(max(bottomLeftD.y, bottomLeftD.x), bottomLeftD.z), bottomLeftD.w);
                    float bottomRight4 = max(max(max(bottomRightD.y, bottomRightD.x), bottomRightD.z), bottomRightD.w);

                    float Wdepth = 0.0;
                    float Ksep_Kfov_diagonal = 1.37e-05 * _SgsrParams.y * length(renderSize);
                    float Depthsep = Ksep_Kfov_diagonal * maxC;
                    float EPSILON = 1.19e-07;
                    Wdepth += saturate(Depthsep / (abs(maxC - topleft4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - topRight4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - bottomLeft4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - bottomRight4) + EPSILON));
                    depthclip = saturate(1.0 - Wdepth * 0.25);
                }
#endif

#if !defined(SGSR_ULTRA)
#if defined(SGSR_STILL)
                // Still: zero motion; the reprojection math (and with Lite
                // the depth gather feeding it) dead-code strips.
                float2 Motion = float2(0.0, 0.0);
#else
                float2 ScreenPos = 2.0 * ViewportUV - 1.0;
                float4 PreClip = _ClipToPrevClip3 + ((_ClipToPrevClip2 * topLeftMax9) +
                    ((_ClipToPrevClip1 * ScreenPos.y) + (_ClipToPrevClip0 * ScreenPos.x)));
                float2 PreScreen = PreClip.xy / PreClip.w;
                float2 Motion = ScreenPos - PreScreen;
#endif
#endif

                // ---- upscale-pass math ------------------------------------
                hfloat depthfactor = (hfloat)depthclip;

#if defined(SGSR_STILL)
                // Zero motion: history sits exactly under this pixel — point
                // Load with no branch (the branched Load-vs-Sample version
                // measured slower than either).
                float4 History = _SgsrPrevHistory.Load(int3(int2(i.positionCS.xy), 0));
#else
                float2 PrevUV;
                PrevUV.x = saturate(-0.5 * Motion.x + HruvOut.x);
                PrevUV.y = saturate(-0.5 * Motion.y * motionFlipY + HruvOut.y);
                float4 History = _SgsrPrevHistory.SampleLevel(sgsr_linear_clamp_sampler, PrevUV, 0.0);
#endif
                hfloat3 HistoryColor = (hfloat3)History.xyz;
                // History.w has been identically 0 since the port (the
                // original's activation flag is unused here). Folding it to a
                // constant breaks the history-fetch -> kernel-bias dependency
                // chain: the Lanczos weight no longer waits on the slowest
                // fetch in the shader.
                const hfloat Wfactor = hfloat(0.0);

                float Biasmax_viewportXScale = min(displaySize.x / renderSize.x, 1.99);
                hfloat scalefactor = (hfloat)min(20.0, pow((displaySize.x / renderSize.x) * (displaySize.y / renderSize.y), 3.0));

                hfloat4 Upsampledcw = hfloat4(0.0, 0.0, 0.0, 0.0);
                hfloat kernelfactor = saturate(Wfactor + hfloat(_Reset));
                hfloat biasmax = hfloat(Biasmax_viewportXScale) - hfloat(Biasmax_viewportXScale) * kernelfactor;
                hfloat biasmin = max(hfloat(1.0), hfloat(0.3) + hfloat(0.3) * biasmax);
                float motion_viewport_len = length(Motion * displaySize);
                // While panning the history contribution collapses (see the
                // startLerpValue note below), so the output is mostly this
                // spatial kernel — at full sharpness it stair-steps. Fast
                // motion widens the kernel instead, which reads as slight
                // motion blur; a dolly moves only a few pixels per frame and
                // never triggers it.
                hfloat hMotionScaled = (hfloat)saturate(motion_viewport_len * 0.02);
                hfloat biasfactor = max(max(hfloat(0.25) * depthfactor, kernelfactor),
                    hMotionScaled);
                // The kernel-bias chain stays in half registers end to end:
                // the old hfloat -> float -> hfloat round trip spent
                // conversions and full registers on values that never
                // exceed |4|.
                hfloat kernelbias = lerp(biasmax, biasmin, biasfactor) * hfloat(0.5);
                hfloat hCurvebias = lerp(hfloat(-2.0), hfloat(-3.0), hMotionScaled);

                hfloat3 rectboxcenter = hfloat3(0.0, 0.0, 0.0);
                hfloat3 rectboxvar = hfloat3(0.0, 0.0, 0.0);
                hfloat rectboxweight = hfloat(0.0);
                float2 srcpos = float2(InputPos) + 0.5 - InputJitter;
                float2 srcOutputPos = HruvIn * renderSize;

                hfloat2 srcDelta = (hfloat2)(srcpos - srcOutputPos);
                hfloat hKernelbias2 = kernelbias * kernelbias;

                float centerColorMax = 0.0;
                hfloat3 rectboxmin;
                hfloat3 rectboxmax;

#if defined(SGSR_ULTRA)
                // Ultra current-frame taps. Still camera: NO color fetch at
                // all — the guide's motion slot carries this texel's color
                // (motion is identically zero), and the Lanczos-weighted EMA
                // converges to the same sharp reconstruction as the multi-tap
                // path because the jitter scans the texel phase by phase.
                // In motion there is no accumulation to refine anything: a
                // single bilinear tap reconstructs edges as the HW tent
                // filter — visibly softer than every multi-tap variant, and
                // unsharp cannot recover an edge the reconstruction lost. So
                // the moving variant runs the same 5-tap Lanczos pattern as
                // the reference moving path; only the box statistics stay
                // precomputed in the guide.
#if defined(SGSR_STILL)
                hfloat3 tapCenter = DecodeColor(gb.x);
#else
                float cmUltra;
                hfloat3 tapCenter = FetchYCoCg(InputPos, centerColorMax);
                hfloat3 tapUp     = FetchYCoCg(InputPos + int2( 0,  1), cmUltra);
                hfloat3 tapRight  = FetchYCoCg(InputPos + int2( 1,  0), cmUltra);
                hfloat3 tapLeft   = FetchYCoCg(InputPos + int2(-1,  0), cmUltra);
                hfloat3 tapDown   = FetchYCoCg(InputPos + int2( 0, -1), cmUltra);
#endif
#elif defined(SGSR_PACKED)
                // Packed input: the whole 5-tap plus pattern arrives in two
                // gathers of the pack-pass output (lane mapping mirrors the
                // reference moving-camera fetch). ColorMax is not stored in
                // the packed buffer, so the >4000 HDR branch never fires.
                float2 packGatherCoord = float2(InputPos) * renderSizeRcp;
                uint4 packTL = _SgsrPacked.GatherRed(sgsr_point_clamp_sampler, packGatherCoord);
                uint2 packBR = _SgsrPacked.GatherRed(sgsr_point_clamp_sampler, packGatherCoord + renderSizeRcp).xz;
                hfloat3 tapCenter = DecodeColor(packTL.y);
                hfloat3 tapUp     = DecodeColor(packBR.x);
                hfloat3 tapRight  = DecodeColor(packBR.y);
                hfloat3 tapLeft   = DecodeColor(packTL.x);
                hfloat3 tapDown   = DecodeColor(packTL.z);
#else
                float cmTmp;
                hfloat3 tapCenter = FetchYCoCg(InputPos, centerColorMax);
                hfloat3 tapUp     = FetchYCoCg(InputPos + int2( 0,  1), cmTmp);
                hfloat3 tapRight  = FetchYCoCg(InputPos + int2( 1,  0), cmTmp);
                hfloat3 tapLeft   = FetchYCoCg(InputPos + int2(-1,  0), cmTmp);
                hfloat3 tapDown   = FetchYCoCg(InputPos + int2( 0, -1), cmTmp);
#endif

#if defined(SGSR_ULTRA)
                // Ultra: the rectification box was precomputed per render
                // pixel (3x3, full quality) — one point tap of the guide.
                rectboxmin = DecodeColor(gb.z);
                rectboxmax = DecodeColor(gb.w);

                hfloat ultraSharpen;
                {
                    hfloat baseoffset_dot = dot(srcDelta, srcDelta);
                    hfloat base = saturate(baseoffset_dot * hKernelbias2);
                    hfloat weight = max(FastLanczos(base), hfloat(0.05));

                    // Unsharp strength only — applied AFTER the temporal
                    // blend (see the output stage). Sharpening the tap here
                    // fed the sharpened value back through history, so the
                    // unsharp reapplied itself every frame: a recursive
                    // amplifier for jitter noise. Kernel-weight modulation
                    // still gates it, and fast motion fades it out — the
                    // kernel already widens with motion, and sharpening a
                    // widened kernel only accentuates stair-stepping.
                    ultraSharpen = (hfloat)_SgsrParams.w * saturate(weight * hfloat(4.0))
                                 * (hfloat(1.0) - (hfloat)saturate(motion_viewport_len * 0.05));

#if defined(SGSR_STILL)
                    // Still: single guide-carried tap, no reconstruction
                    // needed — reprojection is exact, so let the EMA run
                    // long (weight * 0.3): the tap's jitter wobble averages
                    // out over ~14 frames instead of showing. The weight
                    // floor keeps the accumulator alive when the output
                    // pixel lands far from the render texel center.
                    weight *= hfloat(0.3);
                    Upsampledcw = hfloat4(
                        clamp(tapCenter, rectboxmin - hfloat(0.05), rectboxmax + hfloat(0.05)),
                        weight);
#else
                    // Moving: 5-tap Lanczos reconstruction — same kernel as
                    // the reference moving path, so edges in motion match the
                    // multi-tap variants instead of the HW tent filter. The
                    // negative Lanczos lobe needs the true (unfloored) center
                    // weight, and the guide box clamps the normalized sum.
                    hfloat wCenter = FastLanczos(base);
                    Upsampledcw = hfloat4(tapCenter * wCenter, wCenter);
                    ULTRA_TAP(tapUp,     0,  1)
                    ULTRA_TAP(tapRight,  1,  0)
                    ULTRA_TAP(tapLeft,  -1,  0)
                    ULTRA_TAP(tapDown,   0, -1)
                    Upsampledcw.xyz = clamp(Upsampledcw.xyz / Upsampledcw.w,
                        rectboxmin - hfloat(0.05), rectboxmax + hfloat(0.05));
                    Upsampledcw.w = Upsampledcw.w * hfloat(1.0 / 3.0);
#endif
                }
#else
                {
                    hfloat3 samplecolor = tapCenter;
                    hfloat2 baseoffset = srcDelta;
                    hfloat baseoffset_dot = dot(baseoffset, baseoffset);
                    hfloat base = saturate(baseoffset_dot * hKernelbias2);
                    hfloat weight = FastLanczos(base);
                    Upsampledcw += hfloat4(samplecolor * weight, weight);
                    hfloat boxweight = exp(baseoffset_dot * hCurvebias);
                    rectboxmin = samplecolor;
                    rectboxmax = samplecolor;
                    hfloat3 wsample = samplecolor * boxweight;
                    rectboxcenter += wsample;
                    rectboxvar += (samplecolor * wsample);
                    rectboxweight += boxweight;
                }
                BOX_SAMPLE(tapUp,     0,  1)
                BOX_SAMPLE(tapRight,  1,  0)
                BOX_SAMPLE(tapLeft,  -1,  0)
                BOX_SAMPLE(tapDown,   0, -1)

#if !defined(SGSR_LITE) && !defined(SGSR_PACKED) && defined(SGSR_STILL)
                // Lite always runs the 5-tap kernel; the 4 corner taps are
                // the still-camera quality path and the single biggest cost.
                // The packed fetch is 5-tap by construction (two gathers).
                // Compile-variant now: the moving variant never carries this
                // code, the still variant runs it branch-free.
                {
                    float cmCorner;
                    BOX_SAMPLE(FetchYCoCg(InputPos + int2( 1,  1), cmCorner),  1,  1)
                    BOX_SAMPLE(FetchYCoCg(InputPos + int2(-1,  1), cmCorner), -1,  1)
                    BOX_SAMPLE(FetchYCoCg(InputPos + int2( 1, -1), cmCorner),  1, -1)
                    BOX_SAMPLE(FetchYCoCg(InputPos + int2(-1, -1), cmCorner), -1, -1)
                }
#endif

                hfloat rectboxweightRcp = hfloat(1.0) / rectboxweight;
                rectboxcenter *= rectboxweightRcp;
                rectboxvar *= rectboxweightRcp;
                rectboxvar = sqrt(abs(rectboxvar - rectboxcenter * rectboxcenter));

                Upsampledcw.xyz = clamp(Upsampledcw.xyz / Upsampledcw.w, rectboxmin - hfloat(0.05), rectboxmax + hfloat(0.05));
                Upsampledcw.w = Upsampledcw.w * hfloat(1.0 / 3.0);
#endif // !SGSR_ULTRA

                hfloat OneMinusWfactor = hfloat(1.0) - Wfactor;

                hfloat baseupdate = OneMinusWfactor - OneMinusWfactor * depthfactor;
                baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w * hfloat(10.0), (hfloat)saturate(10.0 * motion_viewport_len)));
                baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w, (hfloat)saturate(motion_viewport_len * 0.05)));
                hfloat basealpha = baseupdate;

                const hfloat EPSILON = hfloat(1.192e-07);
#if !defined(SGSR_ULTRA)
                // Variance-scaled box shrink: needs the per-tap statistics,
                // which the ultra path does not accumulate — its precomputed
                // min/max box is used as-is.
                hfloat boxscale = max(depthfactor, (hfloat)saturate(motion_viewport_len * 0.05));
                hfloat boxsize = lerp(scalefactor, hfloat(1.0), boxscale);
                hfloat3 sboxvar = rectboxvar * boxsize;
                hfloat3 boxmin = rectboxcenter - sboxvar;
                hfloat3 boxmax = rectboxcenter + sboxvar;
                rectboxmax = min(rectboxmax, boxmax);
                rectboxmin = max(rectboxmin, boxmin);
#endif

                hfloat3 clampedcolor = clamp(HistoryColor, rectboxmin, rectboxmax);
                hfloat startLerpValue = (hfloat)_SgsrParams.z;
                // The reference zeroes this under any motion: history outside
                // the box is discarded outright while panning, so edges
                // restart every pan frame from the low-res spatial kernel —
                // that is the visible edge pixelation. Our motion vectors are
                // camera-only and exact for static geometry, so keeping a
                // sliver of history is safe; disocclusions are still handled
                // by depthclip. Dynamic objects pay with slightly longer
                // trails.
                if ((abs(Motion.x) + abs(Motion.y)) > 0.000001) startLerpValue = hfloat(0.15);
                hfloat lerpcontribution = (any(rectboxmin > HistoryColor) || any(HistoryColor > rectboxmax)) ? startLerpValue : hfloat(1.0);

                HistoryColor = lerp(clampedcolor, HistoryColor, saturate(lerpcontribution));
                hfloat basemin = min(basealpha, hfloat(0.1));
                basealpha = lerp(basemin, basealpha, saturate(lerpcontribution));

                hfloat alphasum = max(EPSILON, basealpha + Upsampledcw.w);
                hfloat alpha = saturate(Upsampledcw.w / alphasum + hfloat(_Reset));
                Upsampledcw.xyz = lerp(HistoryColor, Upsampledcw.xyz, alpha);

                FragOut o;
                o.history = float4(Upsampledcw.xyz, Wfactor);

#if defined(SGSR_ULTRA)
                // Output-only unsharp, after the temporal blend: the input
                // here is the converged (stable) signal, so the sharpen no
                // longer amplifies per-frame jitter — and the history written
                // above stays clean, breaking the feedback loop.
                Upsampledcw.x = saturate(Upsampledcw.x
                    + (Upsampledcw.x - ultraAvgY) * ultraSharpen);
#endif

                hfloat x_z = Upsampledcw.x - Upsampledcw.z;
                hfloat3 rgb = hfloat3(
                    saturate(x_z + Upsampledcw.y),
                    saturate(Upsampledcw.x + Upsampledcw.z),
                    saturate(x_z - Upsampledcw.y));

                float compMax = saturate(max(max((float)rgb.x, (float)rgb.y), (float)rgb.z));
                float scale = _SgsrParams.x / ((1.0 + 600.0 / 65504.0) - compMax);
                if (centerColorMax > 4000.0) scale = centerColorMax;

                o.color = half4(float3(rgb) * scale, 1.0);
                return o;
            }
            ENDHLSL
        }

        Pass
        {
            Name "SGSR2 Pack"
            ZTest Always ZWrite Off Cull Off Blend Off

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment FragPack
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"

            Texture2D<float4> _SgsrColor;
            float4 _RenderSizeInfo;   // xy = render size
            float4 _SgsrParams;       // x = preExposure

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(uint vertexID : SV_VertexID)
            {
                Varyings o;
                o.positionCS = GetFullScreenTriangleVertexPosition(vertexID);
                o.uv = GetFullScreenTriangleTexCoord(vertexID);
                return o;
            }

            // Tonemap + YCoCg + 11/11/10 packing, identical to the convert
            // pass — runs after transparents, so they are included.
            uint FragPack(Varyings i) : SV_Target
            {
                int2 pos = int2(i.uv * _RenderSizeInfo.xy);
                float3 c = _SgsrColor.Load(int3(pos, 0)).xyz;
                float colorMax = max(max(c.x, c.y), c.z) + _SgsrParams.x;
                float3 rgb = c / colorMax;
                float y = 0.25 * (rgb.x + 2.0 * rgb.y + rgb.z);
                float co = saturate(0.5 * rgb.x + 0.5 - 0.5 * rgb.z);
                float cg = saturate(y + co - rgb.x);
                uint x11 = (uint)(y * 2047.5);
                uint y11 = (uint)(co * 2047.5);
                uint z10 = (uint)(cg * 1023.5);
                return ((x11 << 21u) | (y11 << 10u)) | z10;
            }
            ENDHLSL
        }

        Pass
        {
            Name "SGSR2 Guides"
            ZTest Always ZWrite Off Cull Off Blend Off

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment FragGuides
            #pragma multi_compile_fragment __ SGSR_FP16
            #pragma multi_compile_fragment __ SGSR_LITE
            #pragma multi_compile_fragment __ SGSR_STILL
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"

            // Fat kernel (9x tonemap+YCoCg plus the min/max chain), so fp16
            // pays for its boundary conversions here; depth and motion stay
            // fp32. SGSR_LITE trims the depth work: 2x2 dilation from one
            // gather and no depth-clip, the same trade the Lite kernel makes.
            #if defined(SGSR_FP16)
            #define hfloat  min16float
            #define hfloat3 min16float3
            #else
            #define hfloat  float
            #define hfloat3 float3
            #endif

            // Ultra prep, once per RENDER pixel: full-quality dilated depth,
            // depth-clip, camera motion (MRT0) and the 3x3 YCoCg rectbox
            // (MRT1, 11/11/10 biased min/max). At fractional render scales
            // this math would otherwise repeat for every display pixel that
            // maps to the same render texel.

            Texture2D<float4> _SgsrColor;
            Texture2D<float>  _SgsrDepth;
            SamplerState sgsr_point_clamp_sampler;

            float4 _RenderSizeInfo;   // xy = render size, zw = 1 / render size
            float4 _SgsrParams;       // x = preExposure, y = cameraFovAngleHor
            float4 _ClipToPrevClip0;
            float4 _ClipToPrevClip1;
            float4 _ClipToPrevClip2;
            float4 _ClipToPrevClip3;

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(uint vertexID : SV_VertexID)
            {
                Varyings o;
                o.positionCS = GetFullScreenTriangleVertexPosition(vertexID);
                o.uv = GetFullScreenTriangleTexCoord(vertexID);
                return o;
            }

            struct GuidesOut
            {
                // x = motion (f16 pair), y = depthclip | avgLuma (f16 pair),
                // z = rectbox min, w = rectbox max (11/11/10 biased).
                uint4 gb : SV_Target0;
            };

            hfloat3 ToYCoCgBiased(float3 c)
            {
                float colorMax = max(max(c.x, c.y), c.z) + _SgsrParams.x;
                hfloat3 rgb = (hfloat3)(c / colorMax);
                hfloat y = hfloat(0.25) * (rgb.x + hfloat(2.0) * rgb.y + rgb.z);
                hfloat co = saturate(hfloat(0.5) * rgb.x + hfloat(0.5) - hfloat(0.5) * rgb.z);
                hfloat cg = saturate(y + co - rgb.x);
                return hfloat3(y, co, cg);
            }

            uint PackYCoCg(float3 biased)
            {
                uint x11 = (uint)(biased.x * 2047.5);
                uint y11 = (uint)(biased.y * 2047.5);
                uint z10 = (uint)(biased.z * 1023.5);
                return ((x11 << 21u) | (y11 << 10u)) | z10;
            }

            GuidesOut FragGuides(Varyings i)
            {
                float2 renderSize = _RenderSizeInfo.xy;
                float2 renderSizeRcp = _RenderSizeInfo.zw;
                int2 pos = int2(i.uv * renderSize);

                // ---- full-quality dilated depth + depth-clip --------------
                float2 gatherCoord = float2(pos) * renderSizeRcp;
                float2 ViewportUV = gatherCoord + 0.5 * renderSizeRcp;

                float4 topleftD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord);
#if defined(SGSR_LITE)
                // Lite guides: 2x2 dilation from the one gather above, no
                // depth-clip. 4 depth texels instead of 16 — disocclusion
                // weighting is lost, the rectbox clamp carries that alone.
                float topLeftMax9 = max(max(max(topleftD.y, topleftD.x), topleftD.z), topleftD.w);
                float depthclip = 0.0;
#else
                float4 topRightD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(renderSizeRcp.x * 2.0, 0.0));
                float4 bottomLeftD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(0.0, renderSizeRcp.y * 2.0));
                float4 bottomRightD = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(renderSizeRcp.x * 2.0, renderSizeRcp.y * 2.0));

                float maxC = max(max(max(topleftD.y, topRightD.x), bottomLeftD.z), bottomRightD.w);
                float topleft4 = max(max(max(topleftD.y, topleftD.x), topleftD.z), topleftD.w);
                float topLeftMax9 = max(bottomLeftD.w, max(max(maxC, topleft4), topRightD.w));

                float depthclip = 0.0;
                if (maxC > 1.0e-05)
                {
                    float topRight4 = max(max(max(topRightD.y, topRightD.x), topRightD.z), topRightD.w);
                    float bottomLeft4 = max(max(max(bottomLeftD.y, bottomLeftD.x), bottomLeftD.z), bottomLeftD.w);
                    float bottomRight4 = max(max(max(bottomRightD.y, bottomRightD.x), bottomRightD.z), bottomRightD.w);

                    float Wdepth = 0.0;
                    float Ksep_Kfov_diagonal = 1.37e-05 * _SgsrParams.y * length(renderSize);
                    float Depthsep = Ksep_Kfov_diagonal * maxC;
                    float EPSILON = 1.19e-07;
                    Wdepth += saturate(Depthsep / (abs(maxC - topleft4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - topRight4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - bottomLeft4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - bottomRight4) + EPSILON));
                    depthclip = saturate(1.0 - Wdepth * 0.25);
                }
#endif

                float2 ScreenPos = 2.0 * ViewportUV - 1.0;
                float4 PreClip = _ClipToPrevClip3 + ((_ClipToPrevClip2 * topLeftMax9) +
                    ((_ClipToPrevClip1 * ScreenPos.y) + (_ClipToPrevClip0 * ScreenPos.x)));
                float2 Motion = ScreenPos - PreClip.xy / PreClip.w;

                // ---- 3x3 rectbox ------------------------------------------
                int2 maxPos = int2(renderSize) - 1;
                hfloat3 c0 = ToYCoCgBiased(_SgsrColor.Load(int3(pos, 0)).xyz);
                hfloat3 boxMin = c0;
                hfloat3 boxMax = c0;
                hfloat lumaSum = c0.x;

                [unroll]
                for (int dy = -1; dy <= 1; ++dy)
                {
                    [unroll]
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dx == 0 && dy == 0)
                            continue;
                        int2 p = clamp(pos + int2(dx, dy), int2(0, 0), maxPos);
                        hfloat3 c = ToYCoCgBiased(_SgsrColor.Load(int3(p, 0)).xyz);
                        boxMin = min(boxMin, c);
                        boxMax = max(boxMax, c);
                        lumaSum += c.x;
                    }
                }

                GuidesOut o;
                // Packing runs in fp32: near 2047 the fp16 ulp is 1.0, which
                // would cost the 11-bit quantization its last bit.
                o.gb = uint4(
#if defined(SGSR_STILL)
                    // Still: motion is identically zero, so its slot carries
                    // the center color — the display pass drops its color
                    // fetch. The reprojection math above dead-code strips.
                    PackYCoCg((float3)c0),
#else
                    f32tof16(Motion.x) | (f32tof16(Motion.y) << 16u),
#endif
                    f32tof16(depthclip) | (f32tof16(float(lumaSum) * (1.0 / 9.0)) << 16u),
                    PackYCoCg((float3)boxMin),
                    PackYCoCg((float3)boxMax));
                return o;
            }
            ENDHLSL
        }
    }
    Fallback Off
}
