using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// FusedPacked prepass: tonemapped YCoCg packed to R32_UInt, so the
    /// display pass fetches its 5 taps with two gathers.
    /// </summary>
    internal sealed class Sgsr2PackPass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly Material m_Material;

        public Sgsr2PackPass(Sgsr2UpscaleFeature feature, Material material)
        {
            m_Feature = feature;
            m_Material = material;
        }

        class PassData
        {
            public Material Material;
            public TextureHandle Color;
            public Vector4 RenderSizeInfo;
            public Vector4 SgsrParams;
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

            d0.Packed = renderGraph.CreateTexture(new TextureDesc(d0.RenderW, d0.RenderH)
            {
                name = "_SGSR2Packed",
                format = GraphicsFormat.R32_UInt,
            });

            using (IRasterRenderGraphBuilder builder =
                renderGraph.AddRasterRenderPass("SGSR2 Pack", out PassData data))
            {
                data.Material = m_Material;
                data.Color = resourceData.cameraColor;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.SgsrParams = d0.SgsrParams;

                builder.UseTexture(data.Color, AccessFlags.Read);
                builder.SetRenderAttachment(d0.Packed, 0);
                builder.AllowPassCulling(false);
                builder.AllowGlobalStateModification(true);

                builder.SetRenderFunc((PassData d, RasterGraphContext ctx) =>
                {
                    var cmd = ctx.cmd;
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrColor, d.Color);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.DrawProcedural(Matrix4x4.identity, d.Material, 1, MeshTopology.Triangles, 3, 1);
                });
            }
        }
    }
}
