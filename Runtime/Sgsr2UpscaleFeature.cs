using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// Snapdragon Game Super Resolution 2 for URP with a ladder of
    /// implementations selected per frame by this feature: the three
    /// original reference pipelines (2-pass compute, 2-pass fragment,
    /// 3-pass compute) and the optimized fused family (full, Lite, Packed,
    /// Ultra). Renders at the URP Render Scale, upscales to the display
    /// resolution. Applies its own Halton jitter, so TAA stays off; motion
    /// is camera-only (reconstructed from depth), exact for static scenes.
    ///
    /// Every pipeline pass carries its own Render Graph profiling sampler
    /// (named after the pass), so timings are visible in the Unity Profiler,
    /// in RenderDoc and through UnityEngine.Profiling.Recorder.
    ///
    /// Derived from github.com/SnapdragonGameStudios/snapdragon-gsr (BSD-3).
    /// </summary>
    public sealed class Sgsr2UpscaleFeature : ScriptableRendererFeature
    {
        [Tooltip("Which pipeline runs. Originals are faithful ports for baseline comparison; the Fused family is the optimized path. Optimum depends on hardware tier and render scale.")]
        public Sgsr2Variant variant = Sgsr2Variant.FusedUltra;

        [Tooltip("Per-axis jitter sign/scale. Flip an axis if the image refuses to converge on a static camera.")]
        public Vector2 jitterScale = new Vector2(1f, 1f);

        [Tooltip("Half precision (min16float) where the variant supports it. Measured rule of thumb: helps large kernels, loses in tiny ones where boundary conversions outweigh the savings.")]
        public bool halfPrecision = false;

        [Tooltip("FusedUltra only: guides use 2x2 depth dilation and skip depth clip. 4 depth texels instead of 16 in the prepass.")]
        public bool liteGuides = false;

        [Tooltip("Fused family debug: floor measurement. Identical pass structure, but the shader collapses to one bilinear tap - the theoretical minimum of any upscaler. History zeroes while on.")]
        public bool passthroughFloor = false;

        [Range(0f, 1.5f)]
        [Tooltip("FusedUltra only: luma unsharp strength, applied to the output after the temporal blend (never fed back into history).")]
        public float ultraSharpness = 0.5f;

        /// <summary>Active feature instance, for runtime debug panels.</summary>
        public static Sgsr2UpscaleFeature Instance { get; private set; }

        static readonly Dictionary<Camera, Sgsr2CameraHistory> s_histories = new Dictionary<Camera, Sgsr2CameraHistory>();

        ComputeShader m_Convert;
        ComputeShader m_Upscale;
        ComputeShader m_Convert3;
        ComputeShader m_Activate3;
        ComputeShader m_Upscale3;
        Material m_FusedMaterial;
        Material m_OriginalFsMaterial;

        Sgsr2ConvertComputePass m_ConvertComputePass;
        Sgsr2UpscaleComputePass m_UpscaleComputePass;
        Sgsr2FsConvertPass m_FsConvertPass;
        Sgsr2FsUpscalePass m_FsUpscalePass;
        Sgsr2Convert3Pass m_Convert3Pass;
        Sgsr2Activate3Pass m_Activate3Pass;
        Sgsr2Upscale3Pass m_Upscale3Pass;
        Sgsr2GuidesPass m_GuidesPass;
        Sgsr2PackPass m_PackPass;
        Sgsr2FusedUpscalePass m_FusedPass;

        public override void Create()
        {
            Instance = this;
            m_Convert = Resources.Load<ComputeShader>("Sgsr2OptConvert");
            m_Upscale = Resources.Load<ComputeShader>("Sgsr2OptUpscale");
            m_Convert3 = Resources.Load<ComputeShader>("Sgsr2OptConvert3");
            m_Activate3 = Resources.Load<ComputeShader>("Sgsr2OptActivate3");
            m_Upscale3 = Resources.Load<ComputeShader>("Sgsr2OptUpscale3");

            Shader fusedShader = Resources.Load<Shader>("Sgsr2OptUpscaleFrag");
            if (fusedShader != null && m_FusedMaterial == null)
                m_FusedMaterial = CoreUtils.CreateEngineMaterial(fusedShader);
            Shader fsShader = Resources.Load<Shader>("Sgsr2OptOriginalFs");
            if (fsShader != null && m_OriginalFsMaterial == null)
                m_OriginalFsMaterial = CoreUtils.CreateEngineMaterial(fsShader);

            m_ConvertComputePass = new Sgsr2ConvertComputePass(this, m_Convert) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_UpscaleComputePass = new Sgsr2UpscaleComputePass(this, m_Upscale) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_FsConvertPass = new Sgsr2FsConvertPass(this, m_OriginalFsMaterial) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_FsUpscalePass = new Sgsr2FsUpscalePass(this, m_OriginalFsMaterial) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_Convert3Pass = new Sgsr2Convert3Pass(this, m_Convert3) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_Activate3Pass = new Sgsr2Activate3Pass(this, m_Activate3) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_Upscale3Pass = new Sgsr2Upscale3Pass(this, m_Upscale3) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_GuidesPass = new Sgsr2GuidesPass(this, m_FusedMaterial) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_PackPass = new Sgsr2PackPass(this, m_FusedMaterial) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };
            m_FusedPass = new Sgsr2FusedUpscalePass(this, m_FusedMaterial) { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };

            RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
            RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
            RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
        }

        protected override void Dispose(bool disposing)
        {
            if (Instance == this)
                Instance = null;
            CoreUtils.Destroy(m_FusedMaterial);
            m_FusedMaterial = null;
            CoreUtils.Destroy(m_OriginalFsMaterial);
            m_OriginalFsMaterial = null;
            RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
            RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
            foreach (var history in s_histories.Values)
                history.Release();
            s_histories.Clear();
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            if (!ShouldProcess(renderingData.cameraData.camera))
                return;

            bool fusedFamily = variant >= Sgsr2Variant.FusedFull;
            if (fusedFamily && (m_FusedMaterial == null || !renderingData.cameraData.resolveFinalTarget))
                return;

            SetupKeywords();

            switch (variant)
            {
                case Sgsr2Variant.OriginalCompute2Pass:
                    if (m_Convert == null || m_Upscale == null) return;
                    m_ConvertComputePass.ConfigureInput(ScriptableRenderPassInput.Depth);
                    renderer.EnqueuePass(m_ConvertComputePass);
                    renderer.EnqueuePass(m_UpscaleComputePass);
                    break;

                case Sgsr2Variant.OriginalFragment2Pass:
                    if (m_OriginalFsMaterial == null) return;
                    m_FsConvertPass.ConfigureInput(ScriptableRenderPassInput.Depth);
                    renderer.EnqueuePass(m_FsConvertPass);
                    renderer.EnqueuePass(m_FsUpscalePass);
                    break;

                case Sgsr2Variant.OriginalCompute3Pass:
                    if (m_Convert3 == null || m_Activate3 == null || m_Upscale3 == null) return;
                    // Color requests the opaque texture for the reactivity mask.
                    m_Convert3Pass.ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Color);
                    renderer.EnqueuePass(m_Convert3Pass);
                    renderer.EnqueuePass(m_Activate3Pass);
                    renderer.EnqueuePass(m_Upscale3Pass);
                    break;

                case Sgsr2Variant.FusedUltra:
                    m_FusedPass.ConfigureInput(ScriptableRenderPassInput.Depth);
                    if (!passthroughFloor)
                        renderer.EnqueuePass(m_GuidesPass);
                    renderer.EnqueuePass(m_FusedPass);
                    break;

                case Sgsr2Variant.FusedPacked:
                    m_FusedPass.ConfigureInput(ScriptableRenderPassInput.Depth);
                    if (!passthroughFloor)
                        renderer.EnqueuePass(m_PackPass);
                    renderer.EnqueuePass(m_FusedPass);
                    break;

                default: // FusedFull, FusedLite
                    m_FusedPass.ConfigureInput(ScriptableRenderPassInput.Depth);
                    renderer.EnqueuePass(m_FusedPass);
                    break;
            }
        }

        void SetupKeywords()
        {
            const string kFp16 = "SGSR_FP16";
            const string kLite = "SGSR_LITE";
            const string kPacked = "SGSR_PACKED";
            const string kUltra = "SGSR_ULTRA";
            const string kPassthrough = "SGSR_PASSTHROUGH";

            if (m_Convert != null && m_Upscale != null)
            {
                if (halfPrecision)
                {
                    m_Convert.EnableKeyword(kFp16);
                    m_Upscale.EnableKeyword(kFp16);
                }
                else
                {
                    m_Convert.DisableKeyword(kFp16);
                    m_Upscale.DisableKeyword(kFp16);
                }
            }

            if (m_OriginalFsMaterial != null)
            {
                if (halfPrecision) m_OriginalFsMaterial.EnableKeyword(kFp16);
                else m_OriginalFsMaterial.DisableKeyword(kFp16);
            }

            if (m_FusedMaterial != null)
            {
                bool lite = variant == Sgsr2Variant.FusedLite ||
                            (variant == Sgsr2Variant.FusedUltra && liteGuides);
                bool packed = variant == Sgsr2Variant.FusedPacked && !passthroughFloor;
                bool ultra = variant == Sgsr2Variant.FusedUltra && !passthroughFloor;
                bool passthrough = passthroughFloor && variant >= Sgsr2Variant.FusedFull;

                if (halfPrecision) m_FusedMaterial.EnableKeyword(kFp16);
                else m_FusedMaterial.DisableKeyword(kFp16);
                if (lite) m_FusedMaterial.EnableKeyword(kLite);
                else m_FusedMaterial.DisableKeyword(kLite);
                if (packed) m_FusedMaterial.EnableKeyword(kPacked);
                else m_FusedMaterial.DisableKeyword(kPacked);
                if (ultra) m_FusedMaterial.EnableKeyword(kUltra);
                else m_FusedMaterial.DisableKeyword(kUltra);
                if (passthrough) m_FusedMaterial.EnableKeyword(kPassthrough);
                else m_FusedMaterial.DisableKeyword(kPassthrough);
            }
        }

        static bool ShouldProcess(Camera camera)
        {
            return camera.cameraType == CameraType.Game;
        }

        internal Sgsr2CameraHistory GetHistory(Camera camera)
        {
            if (!s_histories.TryGetValue(camera, out var history))
            {
                history = new Sgsr2CameraHistory();
                s_histories.Add(camera, history);
            }
            return history;
        }

        /// <summary>
        /// Builds (once per frame) the data shared by the active pipeline's
        /// passes: sizes, matrices, history handles, reset/stillness state.
        /// </summary>
        internal Sgsr2FrameData GetFrameData(RenderGraph renderGraph, ContextContainer frameData)
        {
            if (frameData.Contains<Sgsr2FrameData>())
                return frameData.Get<Sgsr2FrameData>();

            var d = frameData.Create<Sgsr2FrameData>();
            var cameraData = frameData.Get<UniversalCameraData>();
            Camera camera = cameraData.camera;
            var desc = cameraData.cameraTargetDescriptor;

            d.Feature = this;
            d.RenderW = desc.width;
            d.RenderH = desc.height;
            d.DisplayW = cameraData.pixelWidth;
            d.DisplayH = cameraData.pixelHeight;

            var history = GetHistory(camera);
            d.History = history;

            // Compute variants write history through a UAV; the fragment and
            // fused variants write it as a render attachment. UAV-capable
            // allocations disable Adreno's lossless bandwidth compression
            // (UBWC), so they are only requested where required.
            bool needRandomWrite = variant == Sgsr2Variant.OriginalCompute2Pass ||
                                   variant == Sgsr2Variant.OriginalCompute3Pass;
            if (history.HistoryA == null || history.Width != d.DisplayW || history.Height != d.DisplayH ||
                history.RandomWriteAlloc != needRandomWrite)
            {
                history.Release();
                history.HistoryA = RTHandles.Alloc(d.DisplayW, d.DisplayH,
                    colorFormat: GraphicsFormat.R16G16B16A16_SFloat,
                    enableRandomWrite: needRandomWrite, name: "_SGSR2HistoryA");
                history.HistoryB = RTHandles.Alloc(d.DisplayW, d.DisplayH,
                    colorFormat: GraphicsFormat.R16G16B16A16_SFloat,
                    enableRandomWrite: needRandomWrite, name: "_SGSR2HistoryB");
                history.Width = d.DisplayW;
                history.Height = d.DisplayH;
                history.HasHistory = false;
                history.RandomWriteAlloc = needRandomWrite;
            }

            if (variant == Sgsr2Variant.OriginalCompute3Pass &&
                (history.LumaA == null || history.LumaWidth != d.RenderW || history.LumaHeight != d.RenderH))
            {
                history.ReleaseLuma();
                history.LumaA = RTHandles.Alloc(d.RenderW, d.RenderH,
                    colorFormat: GraphicsFormat.R32_UInt,
                    enableRandomWrite: true, name: "_SGSR2LumaA");
                history.LumaB = RTHandles.Alloc(d.RenderW, d.RenderH,
                    colorFormat: GraphicsFormat.R32_UInt,
                    enableRandomWrite: true, name: "_SGSR2LumaB");
                history.LumaWidth = d.RenderW;
                history.LumaHeight = d.RenderH;
            }

            d.ResetHistory = !history.HasHistory;

            // clipToPrevClip = prevVP * inverse(curVP), both without jitter,
            // in GPU clip conventions (reversed-Z included).
            Matrix4x4 view = cameraData.GetViewMatrix();
            Matrix4x4 projNoJitter = GL.GetGPUProjectionMatrix(camera.nonJitteredProjectionMatrix, true);
            Matrix4x4 curVP = projNoJitter * view;
            d.ClipToPrevClip = history.HasHistory
                ? history.PrevViewProj * curVP.inverse
                : Matrix4x4.identity;

            d.IsStill = history.HasHistory && IsCameraStill(curVP, history.PrevViewProj);
            history.StillFrames = d.IsStill ? history.StillFrames + 1 : 0;
            history.PrevViewProj = curVP;

            float fovHor = Mathf.Tan(camera.fieldOfView * Mathf.Deg2Rad * 0.5f) * d.RenderW / (float)d.RenderH;
            float minLerp = history.StillFrames > 5 ? 0.3f : 0f;

            d.RenderSizeInfo = new Vector4(d.RenderW, d.RenderH, 1f / d.RenderW, 1f / d.RenderH);
            d.DisplaySizeInfo = new Vector4(d.DisplayW, d.DisplayH, 1f / d.DisplayW, 1f / d.DisplayH);
            d.Jitter = new Vector4(history.JitterPixels.x, history.JitterPixels.y, 0, 0);
            d.SgsrParams = new Vector4(1f /*preExposure*/, fovHor, minLerp, ultraSharpness);

            d.PrevHistory = renderGraph.ImportTexture(history.PrevHistory);
            d.NextHistory = renderGraph.ImportTexture(history.NextHistory);
            if (history.LumaA != null && variant == Sgsr2Variant.OriginalCompute3Pass)
            {
                d.PrevLuma = renderGraph.ImportTexture(history.PrevLuma);
                d.NextLuma = renderGraph.ImportTexture(history.NextLuma);
            }

            return d;
        }

        // Halton(2,3) low-discrepancy sequence, matching the reference
        // integration (index starts at 1, centered around 0).
        static float Halton(int index, int radix)
        {
            float result = 0f;
            float fraction = 1f / radix;
            while (index > 0)
            {
                result += (index % radix) * fraction;
                index /= radix;
                fraction /= radix;
            }
            return result;
        }

        const int kJitterPhases = 32;

        void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            if (!isActive || !ShouldProcess(camera))
                return;

            var asset = UniversalRenderPipeline.asset;
            if (asset == null)
                return;

            var history = GetHistory(camera);
            int index = history.FrameIndex % kJitterPhases;
            var jitter = new Vector2(
                (Halton(index + 1, 2) - 0.5f) * jitterScale.x,
                (Halton(index + 1, 3) - 0.5f) * jitterScale.y);
            history.JitterPixels = jitter;

            float renderW = Mathf.Max(1f, camera.pixelWidth * asset.renderScale);
            float renderH = Mathf.Max(1f, camera.pixelHeight * asset.renderScale);

            camera.ResetProjectionMatrix();
            Matrix4x4 proj = camera.projectionMatrix;
            camera.nonJitteredProjectionMatrix = proj;

            Matrix4x4 jitterTranslate = Matrix4x4.Translate(
                new Vector3(jitter.x * 2f / renderW, jitter.y * 2f / renderH, 0f));
            camera.projectionMatrix = jitterTranslate * proj;
            camera.useJitteredProjectionMatrixForTransparentRendering = true;
        }

        void OnEndCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            if (!ShouldProcess(camera))
                return;
            camera.ResetProjectionMatrix();
        }

        static bool IsCameraStill(in Matrix4x4 current, in Matrix4x4 previous, float threshold = 1e-5f)
        {
            float diff = 0f;
            for (int i = 0; i < 16; ++i)
                diff += Mathf.Abs(current[i] - previous[i]);
            return diff < threshold;
        }
    }
}
