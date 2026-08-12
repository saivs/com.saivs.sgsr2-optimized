using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>Which pass pipeline the feature builds this frame.</summary>
    public enum Sgsr2Variant
    {
        /// <summary>Reference port: 2-pass compute (convert + upscale), YCoCg.</summary>
        OriginalCompute2Pass = 0,
        /// <summary>Reference port: 2-pass fragment, raw RGB, output doubles as history. The reference's fastest mobile variant.</summary>
        OriginalFragment2Pass = 1,
        /// <summary>Reference port: 3-pass compute (convert + activate + upscale) with reactivity mask and luma stability history. The reference's quality variant.</summary>
        OriginalCompute3Pass = 2,
        /// <summary>Optimized: single fused fragment pass, the upscale is the final blit. Full kernel.</summary>
        FusedFull = 3,
        /// <summary>Optimized: fused, 5-tap kernel, 2x2 depth dilation, no depth clip. Best at native scale and on faster GPUs.</summary>
        FusedLite = 4,
        /// <summary>Optimized: fused, packed YCoCg input fetched with two gathers. Wins or loses by GPU generation - measure.</summary>
        FusedPacked = 5,
        /// <summary>Optimized: render-resolution guides prepass + 2-fetch display pass. Best at aggressive downscale on low-tier GPUs.</summary>
        FusedUltra = 6,
    }

    /// <summary>Per-camera persistent state: history ping-pong, jitter phase, stillness tracking.</summary>
    public sealed class Sgsr2CameraHistory
    {
        public RTHandle HistoryA;
        public RTHandle HistoryB;
        public RTHandle LumaA;    // 3-pass only: luma stability history
        public RTHandle LumaB;
        public int FrameIndex;
        public int Width;
        public int Height;
        public int LumaWidth;
        public int LumaHeight;
        public bool HasHistory;
        public bool RandomWriteAlloc;
        public Matrix4x4 PrevViewProj = Matrix4x4.identity;
        public int StillFrames;
        public Vector2 JitterPixels;

        public RTHandle PrevHistory => (FrameIndex & 1) == 0 ? HistoryA : HistoryB;
        public RTHandle NextHistory => (FrameIndex & 1) == 0 ? HistoryB : HistoryA;
        public RTHandle PrevLuma => (FrameIndex & 1) == 0 ? LumaA : LumaB;
        public RTHandle NextLuma => (FrameIndex & 1) == 0 ? LumaB : LumaA;

        public void Release()
        {
            HistoryA?.Release();
            HistoryB?.Release();
            HistoryA = null;
            HistoryB = null;
            ReleaseLuma();
            HasHistory = false;
        }

        public void ReleaseLuma()
        {
            LumaA?.Release();
            LumaB?.Release();
            LumaA = null;
            LumaB = null;
        }
    }

    /// <summary>
    /// Per-frame data shared between the passes of the active pipeline.
    /// Created by the first pass that runs, consumed by the rest; producer
    /// passes publish their intermediate texture handles here.
    /// </summary>
    public sealed class Sgsr2FrameData : ContextItem
    {
        public Sgsr2UpscaleFeature Feature;
        public Sgsr2CameraHistory History;
        public int RenderW, RenderH, DisplayW, DisplayH;
        public Vector4 RenderSizeInfo, DisplaySizeInfo, Jitter, SgsrParams;
        public Matrix4x4 ClipToPrevClip;
        public bool IsStill, ResetHistory;
        public TextureHandle PrevHistory, NextHistory;
        public TextureHandle PrevLuma, NextLuma;

        // Published by producer passes:
        public TextureHandle Mda;       // convert output (motion, depth or depthclip, alpha)
        public TextureHandle YCoCg;     // packed color (compute variants)
        public TextureHandle MdcaClip;  // 3-pass activate output
        public TextureHandle GuideBox;  // Ultra guides
        public TextureHandle Packed;    // FusedPacked input

        public override void Reset()
        {
            Feature = null;
            History = null;
            PrevHistory = NextHistory = default;
            PrevLuma = NextLuma = default;
            Mda = YCoCg = MdcaClip = GuideBox = Packed = default;
        }

        /// <summary>Called by the terminal pass of the active pipeline.</summary>
        public void FinishFrame()
        {
            History.HasHistory = true;
            History.FrameIndex++;
        }
    }

    internal static class Sgsr2ShaderIDs
    {
        public static readonly int RenderSizeInfo = Shader.PropertyToID("_RenderSizeInfo");
        public static readonly int DisplaySizeInfo = Shader.PropertyToID("_DisplaySizeInfo");
        public static readonly int JitterOffset = Shader.PropertyToID("_JitterOffset");
        public static readonly int ClipToPrevClip0 = Shader.PropertyToID("_ClipToPrevClip0");
        public static readonly int ClipToPrevClip1 = Shader.PropertyToID("_ClipToPrevClip1");
        public static readonly int ClipToPrevClip2 = Shader.PropertyToID("_ClipToPrevClip2");
        public static readonly int ClipToPrevClip3 = Shader.PropertyToID("_ClipToPrevClip3");
        public static readonly int SgsrParams = Shader.PropertyToID("_SgsrParams");
        public static readonly int SameCameraFrmNum = Shader.PropertyToID("_SameCameraFrmNum");
        public static readonly int Reset = Shader.PropertyToID("_Reset");
        public static readonly int SameCamera = Shader.PropertyToID("_SameCamera");
        public static readonly int BlitScaleBias = Shader.PropertyToID("_BlitScaleBias");

        public static readonly int SgsrColor = Shader.PropertyToID("_SgsrColor");
        public static readonly int SgsrDepth = Shader.PropertyToID("_SgsrDepth");
        public static readonly int SgsrPrevHistory = Shader.PropertyToID("_SgsrPrevHistory");
        public static readonly int SgsrMda = Shader.PropertyToID("_SgsrMda");
        public static readonly int SgsrPacked = Shader.PropertyToID("_SgsrPacked");
        public static readonly int SgsrGuideBox = Shader.PropertyToID("_SgsrGuideBox");

        public static readonly int InputColor = Shader.PropertyToID("_InputColor");
        public static readonly int InputOpaqueColor = Shader.PropertyToID("_InputOpaqueColor");
        public static readonly int InputDepth = Shader.PropertyToID("_InputDepth");
        public static readonly int Mda = Shader.PropertyToID("_MotionDepthAlphaBuffer");
        public static readonly int MdcaClip = Shader.PropertyToID("_MotionDepthClipAlphaBuffer");
        public static readonly int YCoCg = Shader.PropertyToID("_YCoCgColor");
        public static readonly int PrevHistoryOutput = Shader.PropertyToID("_PrevHistoryOutput");
        public static readonly int HistoryOutput = Shader.PropertyToID("_HistoryOutput");
        public static readonly int SceneColorOutput = Shader.PropertyToID("_SceneColorOutput");
        public static readonly int PrevLumaHistory = Shader.PropertyToID("_PrevLumaHistory");
        public static readonly int LumaHistory = Shader.PropertyToID("_LumaHistory");
    }
}
