using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// The optimized display pass: the upscale IS the final blit. MRT0 =
    /// backbuffer, MRT1 = new history, all convert math inline (or fetched
    /// from the guides/pack prepasses). Terminal pass: switches URP's active
    /// targets to the backbuffer so the stock final blit is skipped.
    /// </summary>
    internal sealed class Sgsr2FusedUpscalePass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly Material m_Material;

        public Sgsr2FusedUpscalePass(Sgsr2UpscaleFeature feature, Material material)
        {
            m_Feature = feature;
            m_Material = material;
        }

        class PassData
        {
            public Material Material;
            public TextureHandle Color;
            public TextureHandle Depth;
            public TextureHandle PrevHistory;
            public TextureHandle Packed;
            public TextureHandle GuideBox;
            public TextureHandle BackBuffer;
            public Vector4 RenderSizeInfo;
            public Vector4 DisplaySizeInfo;
            public Vector4 Jitter;
            public Vector4 SgsrParams;
            public Vector4 C0, C1, C2, C3;
            public float SameCamera;
            public float Reset;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();
            if (resourceData.isActiveTargetBackBuffer)
                return;
            var cameraData = frameData.Get<UniversalCameraData>();
            if (!cameraData.resolveFinalTarget)
                return;
            Sgsr2FrameData d0 = m_Feature.GetFrameData(renderGraph, frameData);

            // Stillness as a compile variant, not a uniform branch: branches
            // around fetches measurably cost registers and fetch scheduling
            // on mobile GPUs. The keyword flips only when the camera starts
            // or stops moving; both variants precompile.
            const string kStillKeyword = "SGSR_STILL";
            if (d0.IsStill) m_Material.EnableKeyword(kStillKeyword);
            else m_Material.DisableKeyword(kStillKeyword);

            using (IRasterRenderGraphBuilder builder =
                renderGraph.AddRasterRenderPass("SGSR2 Fused Upscale", out PassData data))
            {
                data.Material = m_Material;
                data.Color = resourceData.cameraColor;
                data.Depth = resourceData.cameraDepthTexture;
                data.PrevHistory = d0.PrevHistory;
                data.BackBuffer = resourceData.backBufferColor;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.DisplaySizeInfo = d0.DisplaySizeInfo;
                data.Jitter = d0.Jitter;
                data.SgsrParams = d0.SgsrParams;
                data.C0 = d0.ClipToPrevClip.GetColumn(0);
                data.C1 = d0.ClipToPrevClip.GetColumn(1);
                data.C2 = d0.ClipToPrevClip.GetColumn(2);
                data.C3 = d0.ClipToPrevClip.GetColumn(3);
                data.SameCamera = d0.IsStill ? 1f : 0f;
                data.Reset = d0.ResetHistory ? 1f : 0f;
                data.Packed = d0.Packed;
                data.GuideBox = d0.GuideBox;

                builder.UseTexture(data.Color, AccessFlags.Read);
                builder.UseTexture(data.Depth, AccessFlags.Read);
                builder.UseTexture(data.PrevHistory, AccessFlags.Read);
                if (d0.Packed.IsValid())
                    builder.UseTexture(d0.Packed, AccessFlags.Read);
                if (d0.GuideBox.IsValid())
                    builder.UseTexture(d0.GuideBox, AccessFlags.Read);
                builder.SetRenderAttachment(resourceData.backBufferColor, 0);
                builder.SetRenderAttachment(d0.NextHistory, 1);
                builder.AllowPassCulling(false);
                builder.AllowGlobalStateModification(true);

                builder.SetRenderFunc((PassData d, RasterGraphContext ctx) =>
                {
                    Vector4 scaleBias = RenderingUtils.GetFinalBlitScaleBias(ctx, d.Color, d.BackBuffer);
                    var cmd = ctx.cmd;
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrColor, d.Color);
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrDepth, d.Depth);
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrPrevHistory, d.PrevHistory);
                    if (d.Packed.IsValid())
                        cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrPacked, d.Packed);
                    if (d.GuideBox.IsValid())
                        cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrGuideBox, d.GuideBox);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.DisplaySizeInfo, d.DisplaySizeInfo);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.JitterOffset, d.Jitter);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip0, d.C0);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip1, d.C1);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip2, d.C2);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip3, d.C3);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.BlitScaleBias, scaleBias);
                    cmd.SetGlobalFloat(Sgsr2ShaderIDs.SameCamera, d.SameCamera);
                    cmd.SetGlobalFloat(Sgsr2ShaderIDs.Reset, d.Reset);
                    cmd.DrawProcedural(Matrix4x4.identity, d.Material, 0, MeshTopology.Triangles, 3, 1);
                });
            }

            resourceData.SwitchActiveTexturesToBackbuffer();
            d0.FinishFrame();
        }
    }
}
