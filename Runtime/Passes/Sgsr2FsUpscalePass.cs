using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// Reference 2-pass fragment variant, pass 2: temporal upscale in raw
    /// RGB into the display-resolution history target, which doubles as the
    /// frame output (the reference's design: PrevOutput = last frame's
    /// Output). Terminal pass.
    /// </summary>
    internal sealed class Sgsr2FsUpscalePass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly Material m_Material;

        public Sgsr2FsUpscalePass(Sgsr2UpscaleFeature feature, Material material)
        {
            m_Feature = feature;
            m_Material = material;
        }

        class PassData
        {
            public Material Material;
            public TextureHandle Color;
            public TextureHandle Mda;
            public TextureHandle PrevHistory;
            public Vector4 RenderSizeInfo;
            public Vector4 DisplaySizeInfo;
            public Vector4 Jitter;
            public Vector4 SgsrParams;
            public float Reset;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();
            if (resourceData.isActiveTargetBackBuffer)
                return;
            Sgsr2FrameData d0 = m_Feature.GetFrameData(renderGraph, frameData);

            using (IRasterRenderGraphBuilder builder =
                renderGraph.AddRasterRenderPass("SGSR2 FS Upscale", out PassData data))
            {
                data.Material = m_Material;
                data.Color = resourceData.cameraColor;
                data.Mda = d0.Mda;
                data.PrevHistory = d0.PrevHistory;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.DisplaySizeInfo = d0.DisplaySizeInfo;
                data.Jitter = d0.Jitter;
                data.SgsrParams = d0.SgsrParams;
                data.Reset = d0.ResetHistory ? 1f : 0f;

                builder.UseTexture(data.Color, AccessFlags.Read);
                builder.UseTexture(data.Mda, AccessFlags.Read);
                builder.UseTexture(data.PrevHistory, AccessFlags.Read);
                builder.SetRenderAttachment(d0.NextHistory, 0);
                builder.AllowPassCulling(false);
                builder.AllowGlobalStateModification(true);

                builder.SetRenderFunc((PassData d, RasterGraphContext ctx) =>
                {
                    var cmd = ctx.cmd;
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrColor, d.Color);
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrMda, d.Mda);
                    cmd.SetGlobalTexture(Sgsr2ShaderIDs.SgsrPrevHistory, d.PrevHistory);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.DisplaySizeInfo, d.DisplaySizeInfo);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.JitterOffset, d.Jitter);
                    cmd.SetGlobalVector(Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.SetGlobalFloat(Sgsr2ShaderIDs.Reset, d.Reset);
                    cmd.DrawProcedural(Matrix4x4.identity, d.Material, 1, MeshTopology.Triangles, 3, 1);
                });
            }

            resourceData.cameraColor = d0.NextHistory;
            d0.FinishFrame();
        }
    }
}
