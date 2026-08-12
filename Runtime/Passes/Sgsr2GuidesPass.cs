using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// Ultra prepass: folds the per-render-pixel neighborhood analysis
    /// (dilated depth, depth clip, motion, 3x3 rectbox, average luma - and
    /// on a still camera the center color in the motion slot) into one uint4
    /// texel that the display pass reads with a single Load.
    /// </summary>
    internal sealed class Sgsr2GuidesPass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly Material m_Material;

        public Sgsr2GuidesPass(Sgsr2UpscaleFeature feature, Material material)
        {
            m_Feature = feature;
            m_Material = material;
        }

        class PassData
        {
            public Material Material;
            public TextureHandle Color;
            public TextureHandle Depth;
            public Vector4 RenderSizeInfo;
            public Vector4 SgsrParams;
            public Vector4 C0, C1, C2, C3;
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

            d0.GuideBox = renderGraph.CreateTexture(new TextureDesc(d0.RenderW, d0.RenderH)
            {
                name = "_SGSR2GuideBox",
                format = GraphicsFormat.R32G32B32A32_UInt,
            });

            using (IRasterRenderGraphBuilder builder =
                renderGraph.AddRasterRenderPass("SGSR2 Guides", out PassData data))
            {
                data.Material = m_Material;
                data.Color = resourceData.cameraColor;
                data.Depth = resourceData.cameraDepthTexture;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.SgsrParams = d0.SgsrParams;
                data.C0 = d0.ClipToPrevClip.GetColumn(0);
                data.C1 = d0.ClipToPrevClip.GetColumn(1);
                data.C2 = d0.ClipToPrevClip.GetColumn(2);
                data.C3 = d0.ClipToPrevClip.GetColumn(3);

                builder.UseTexture(data.Color, AccessFlags.Read);
                builder.UseTexture(data.Depth, AccessFlags.Read);
                builder.SetRenderAttachment(d0.GuideBox, 0);
                builder.AllowPassCulling(false);
                builder.AllowGlobalStateModification(true);

                builder.SetRenderFunc((PassData d, RasterGraphContext ctx) =>
                {
                    var cmd = ctx.cmd;
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrColor, d.Color);
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrDepth, d.Depth);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip0, d.C0);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip1, d.C1);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip2, d.C2);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip3, d.C3);
                    cmd.DrawProcedural(Matrix4x4.identity, d.Material, 2, MeshTopology.Triangles, 3, 1);
                });
            }
        }
    }
}
