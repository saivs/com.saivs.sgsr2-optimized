using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// Reference 2-pass fragment variant, pass 1: motion + depth clip into a
    /// render-resolution rgba16f target (raw RGB pipeline, no color pack).
    /// </summary>
    internal sealed class Sgsr2FsConvertPass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly Material m_Material;

        public Sgsr2FsConvertPass(Sgsr2UpscaleFeature feature, Material material)
        {
            m_Feature = feature;
            m_Material = material;
        }

        class PassData
        {
            public Material Material;
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
            Sgsr2FrameData d0 = m_Feature.GetFrameData(renderGraph, frameData);

            d0.Mda = renderGraph.CreateTexture(new TextureDesc(d0.RenderW, d0.RenderH)
            {
                name = "_SGSR2FsMda",
                format = GraphicsFormat.R16G16B16A16_SFloat,
            });

            using (IRasterRenderGraphBuilder builder =
                renderGraph.AddRasterRenderPass("SGSR2 FS Convert", out PassData data))
            {
                data.Material = m_Material;
                data.Depth = resourceData.cameraDepthTexture;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.SgsrParams = d0.SgsrParams;
                data.C0 = d0.ClipToPrevClip.GetColumn(0);
                data.C1 = d0.ClipToPrevClip.GetColumn(1);
                data.C2 = d0.ClipToPrevClip.GetColumn(2);
                data.C3 = d0.ClipToPrevClip.GetColumn(3);

                builder.UseTexture(data.Depth, AccessFlags.Read);
                builder.SetRenderAttachment(d0.Mda, 0);
                builder.AllowPassCulling(false);
                builder.AllowGlobalStateModification(true);

                builder.SetRenderFunc((PassData d, RasterGraphContext ctx) =>
                {
                    var cmd = ctx.cmd;
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrDepth, d.Depth);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip0, d.C0);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip1, d.C1);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip2, d.C2);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.ClipToPrevClip3, d.C3);
                    cmd.DrawProcedural(Matrix4x4.identity, d.Material, 0, MeshTopology.Triangles, 3, 1);
                });
            }
        }
    }
}
