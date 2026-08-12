// Faithful port of the reference 2-pass FRAGMENT variant (their fastest
// mobile path): glsl_2_pass_fs/sgsr2_convert.fs + sgsr2_upscale.fs.
// Works in raw RGB (no tonemap, no YCoCg), 5 color taps, and the upscale
// output doubles as next frame's history. Adaptations, consistent with the
// rest of this package: reversed-Z (nearest = max, far plane = 0, nearness
// = raw depth), velocity input removed (camera motion reconstructed from
// depth), out-of-range taps clamped (the reference relies on UB there).
//
// Derived from github.com/SnapdragonGameStudios/snapdragon-gsr (BSD-3):
//                  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
Shader "Hidden/Sgsr2Optimized/OriginalFs"
{
    SubShader
    {
        Tags { "RenderPipeline" = "UniversalPipeline" }

        HLSLINCLUDE
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

        float4 _RenderSizeInfo;   // xy = render size, zw = 1 / render size
        float4 _DisplaySizeInfo;  // xy = display size, zw = 1 / display size
        float4 _JitterOffset;     // xy = jitter in render pixels
        float4 _SgsrParams;       // x = preExposure, y = cameraFovAngleHor, z = minLerpContribution
        float4 _ClipToPrevClip0;
        float4 _ClipToPrevClip1;
        float4 _ClipToPrevClip2;
        float4 _ClipToPrevClip3;
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
        ENDHLSL

        Pass
        {
            Name "SGSR2 FS Convert"
            ZTest Always ZWrite Off Cull Off Blend Off

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment FragConvert

            Texture2D<float> _SgsrDepth;
            SamplerState sgsr_point_clamp_sampler;

            // Reference: MotionDepthClipAlphaBuffer = (motion, depthclip, 0)
            float4 FragConvert(Varyings i) : SV_Target
            {
                float2 renderSize = _RenderSizeInfo.xy;
                float2 renderSizeRcp = _RenderSizeInfo.zw;
                float2 texCoord = i.uv;
                float2 gatherCoord = texCoord - 0.5 * renderSizeRcp;

                float4 btmLeft = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord);
                float4 btmRight = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(renderSizeRcp.x * 2.0, 0.0));
                float4 topLeft = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(0.0, renderSizeRcp.y * 2.0));
                float4 topRight = _SgsrDepth.GatherRed(sgsr_point_clamp_sampler, gatherCoord + float2(renderSizeRcp.x * 2.0, renderSizeRcp.y * 2.0));

                // Reversed-Z: the reference takes min (nearest in standard Z).
                float maxC = max(max(max(btmLeft.z, btmRight.w), topLeft.y), topRight.x);
                float btmLeft4 = max(max(max(btmLeft.y, btmLeft.x), btmLeft.z), btmLeft.w);
                float btmLeftMax9 = max(topLeft.x, max(max(maxC, btmLeft4), btmRight.x));

                float depthclip = 0.0;
                if (maxC > 1.0e-05)
                {
                    float btmRight4 = max(max(max(btmRight.y, btmRight.x), btmRight.z), btmRight.w);
                    float topLeft4 = max(max(max(topLeft.y, topLeft.x), topLeft.z), topLeft.w);
                    float topRight4 = max(max(max(topRight.y, topRight.x), topRight.z), topRight.w);

                    float Wdepth = 0.0;
                    float Ksep_Kfov_diagonal = 1.37e-05 * _SgsrParams.y * length(renderSize);
                    // Reversed-Z nearness is the raw depth (reference: 1 - maxC).
                    float Depthsep = Ksep_Kfov_diagonal * maxC;
                    float EPSILON = 1.19e-07;
                    Wdepth += saturate(Depthsep / (abs(maxC - btmLeft4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - btmRight4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - topLeft4) + EPSILON));
                    Wdepth += saturate(Depthsep / (abs(maxC - topRight4) + EPSILON));
                    depthclip = saturate(1.0 - Wdepth * 0.25);
                }

                float2 ScreenPos = 2.0 * texCoord - 1.0;
                float4 PreClip = _ClipToPrevClip3 + ((_ClipToPrevClip2 * btmLeftMax9) +
                    ((_ClipToPrevClip1 * ScreenPos.y) + (_ClipToPrevClip0 * ScreenPos.x)));
                float2 PreScreen = PreClip.xy / PreClip.w;
                float2 Motion = ScreenPos - PreScreen;

                return float4(Motion, depthclip, 0.0);
            }
            ENDHLSL
        }

        Pass
        {
            Name "SGSR2 FS Upscale"
            ZTest Always ZWrite Off Cull Off Blend Off

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment FragUpscale
            #pragma multi_compile_fragment __ SGSR_FP16

            Texture2D<float4> _SgsrColor;
            Texture2D<float4> _SgsrPrevHistory;   // previous frame's output
            Texture2D<float4> _SgsrMda;           // motion, depthclip
            SamplerState sgsr_linear_clamp_sampler;

            hfloat FastLanczos(hfloat base)
            {
                hfloat y = base - hfloat(1.0);
                hfloat y2 = y * y;
                hfloat y_temp = hfloat(0.75) * y + y2;
                return y_temp * y2;
            }

            hfloat3 FetchColor(int2 pos)
            {
                pos = clamp(pos, int2(0, 0), int2(_RenderSizeInfo.xy) - 1);
                return (hfloat3)_SgsrColor.Load(int3(pos, 0)).xyz;
            }

            #define FS_BOX_SAMPLE(colorexpr, offsx, offsy) \
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

            float4 FragUpscale(Varyings i) : SV_Target
            {
                float2 renderSize = _RenderSizeInfo.xy;
                float2 renderSizeRcp = _RenderSizeInfo.zw;
                float2 displaySize = _DisplaySizeInfo.xy;

                float Biasmax_viewportXScale = min(displaySize.x / renderSize.x, 1.99);
                hfloat scalefactor = (hfloat)min(20.0, pow((displaySize.x / renderSize.x) * (displaySize.y / renderSize.y), 3.0));

                float2 Hruv = i.uv;
                float2 InputJitter = _JitterOffset.xy;
                float2 Jitteruv;
                Jitteruv.x = saturate(Hruv.x + (InputJitter.x * renderSizeRcp.x));
                Jitteruv.y = saturate(Hruv.y + (InputJitter.y * renderSizeRcp.y));
                int2 InputPos = min(int2(Jitteruv * renderSize), int2(renderSize) - 1);

                float3 mda = _SgsrMda.SampleLevel(sgsr_linear_clamp_sampler, Jitteruv, 0.0).xyz;
                float2 Motion = mda.xy;

                float2 PrevUV;
                PrevUV.x = saturate(-0.5 * Motion.x + Hruv.x);
                PrevUV.y = saturate(-0.5 * Motion.y + Hruv.y);

                hfloat depthfactor = (hfloat)mda.z;
                hfloat3 HistoryColor = (hfloat3)_SgsrPrevHistory.SampleLevel(sgsr_linear_clamp_sampler, PrevUV, 0.0).xyz;

                hfloat4 Upsampledcw = hfloat4(0.0, 0.0, 0.0, 0.0);
                hfloat biasmax = (hfloat)Biasmax_viewportXScale;
                hfloat biasmin = max(hfloat(1.0), hfloat(0.3) + hfloat(0.3) * biasmax);
                hfloat biasfactor = hfloat(0.25) * depthfactor;
                hfloat kernelbias = lerp(biasmax, biasmin, biasfactor) * hfloat(0.5);
                float motion_viewport_len = length(Motion * displaySize);
                hfloat hMotionScaled = (hfloat)saturate(motion_viewport_len * 0.02);
                hfloat hCurvebias = lerp(hfloat(-2.0), hfloat(-3.0), hMotionScaled);
                hfloat hKernelbias2 = kernelbias * kernelbias;

                hfloat3 rectboxcenter = hfloat3(0.0, 0.0, 0.0);
                hfloat3 rectboxvar = hfloat3(0.0, 0.0, 0.0);
                hfloat rectboxweight = hfloat(0.0);
                float2 srcpos = float2(InputPos) + 0.5 - InputJitter;
                hfloat2 srcDelta = (hfloat2)(srcpos - Hruv * renderSize);

                hfloat3 rectboxmin;
                hfloat3 rectboxmax;
                {
                    hfloat3 samplecolor = FetchColor(InputPos + int2(0, 1));
                    hfloat2 baseoffset = srcDelta + hfloat2(0.0, 1.0);
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
                FS_BOX_SAMPLE(FetchColor(InputPos + int2( 1,  0)),  1,  0)
                FS_BOX_SAMPLE(FetchColor(InputPos + int2(-1,  0)), -1,  0)
                FS_BOX_SAMPLE(FetchColor(InputPos + int2( 0,  0)),  0,  0)
                FS_BOX_SAMPLE(FetchColor(InputPos + int2( 0, -1)),  0, -1)
                // The reference ships the 4 corner taps behind if(false)
                // ("maybe disable this for ultra performance") - kept off.

                hfloat rectboxweightRcp = hfloat(1.0) / rectboxweight;
                rectboxcenter *= rectboxweightRcp;
                rectboxvar *= rectboxweightRcp;
                rectboxvar = sqrt(abs(rectboxvar - rectboxcenter * rectboxcenter));

                Upsampledcw.xyz = clamp(Upsampledcw.xyz / Upsampledcw.w, rectboxmin - hfloat(0.075), rectboxmax + hfloat(0.075));
                Upsampledcw.w = Upsampledcw.w * hfloat(1.0 / 3.0);

                hfloat baseupdate = hfloat(1.0) - depthfactor;
                baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w * hfloat(10.0), (hfloat)saturate(10.0 * motion_viewport_len)));
                baseupdate = min(baseupdate, lerp(baseupdate, Upsampledcw.w, (hfloat)saturate(motion_viewport_len * 0.05)));
                hfloat basealpha = baseupdate;

                const hfloat EPSILON = hfloat(1.192e-07);
                hfloat boxscale = max(depthfactor, (hfloat)saturate(motion_viewport_len * 0.05));
                hfloat boxsize = lerp(scalefactor, hfloat(1.0), boxscale);
                hfloat3 sboxvar = rectboxvar * boxsize;
                hfloat3 boxmin = rectboxcenter - sboxvar;
                hfloat3 boxmax = rectboxcenter + sboxvar;
                rectboxmax = min(rectboxmax, boxmax);
                rectboxmin = max(rectboxmin, boxmin);

                hfloat3 clampedcolor = clamp(HistoryColor, rectboxmin, rectboxmax);
                hfloat startLerpValue = (hfloat)_SgsrParams.z;
                if ((abs(Motion.x) + abs(Motion.y)) > 0.000001) startLerpValue = hfloat(0.0);
                hfloat lerpcontribution = (any(rectboxmin > HistoryColor) || any(HistoryColor > rectboxmax)) ? startLerpValue : hfloat(1.0);

                HistoryColor = lerp(clampedcolor, HistoryColor, saturate(lerpcontribution));
                hfloat basemin = min(basealpha, hfloat(0.1));
                basealpha = lerp(basemin, basealpha, saturate(lerpcontribution));

                hfloat alphasum = max(EPSILON, basealpha + Upsampledcw.w);
                hfloat alpha = saturate(Upsampledcw.w / alphasum + hfloat(_Reset));
                Upsampledcw.xyz = lerp(HistoryColor, Upsampledcw.xyz, alpha);

                return float4(Upsampledcw.xyz, 0.0);
            }
            ENDHLSL
        }
    }
    Fallback Off
}
