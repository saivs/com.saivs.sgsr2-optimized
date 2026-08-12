using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// Reference 3-pass compute variant, pass 2: depth clip against the
    /// reprojected previous depth plus the luma stability history.
    /// </summary>
    internal sealed class Sgsr2Activate3Pass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly ComputeShader m_Shader;

        public Sgsr2Activate3Pass(Sgsr2UpscaleFeature feature, ComputeShader shader)
        {
            m_Feature = feature;
            m_Shader = shader;
        }

        class PassData
        {
            public ComputeShader Shader;
            public int Kernel;
            public TextureHandle PrevLuma;
            public TextureHandle Mda;
            public TextureHandle YCoCg;
            public TextureHandle MdcaClip;
            public TextureHandle LumaOut;
            public Vector4 RenderSizeInfo;
            public Vector4 SgsrParams;
            public int Reset;
            public int GroupsX, GroupsY;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();
            if (resourceData.isActiveTargetBackBuffer)
                return;
            Sgsr2FrameData d0 = m_Feature.GetFrameData(renderGraph, frameData);

            d0.MdcaClip = renderGraph.CreateTexture(new TextureDesc(d0.RenderW, d0.RenderH)
            {
                name = "_SGSR2MdcaClip3",
                format = GraphicsFormat.R16G16B16A16_SFloat,
                enableRandomWrite = true,
            });

            using (IComputeRenderGraphBuilder builder =
                renderGraph.AddComputePass("SGSR2 Activate3", out PassData data))
            {
                data.Shader = m_Shader;
                data.Kernel = m_Shader.FindKernel("Activate3");
                data.PrevLuma = d0.PrevLuma;
                data.Mda = d0.Mda;
                data.YCoCg = d0.YCoCg;
                data.MdcaClip = d0.MdcaClip;
                data.LumaOut = d0.NextLuma;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.SgsrParams = d0.SgsrParams;
                data.Reset = d0.ResetHistory ? 1 : 0;
                data.GroupsX = (d0.RenderW + 7) / 8;
                data.GroupsY = (d0.RenderH + 7) / 8;

                builder.UseTexture(data.PrevLuma, AccessFlags.Read);
                builder.UseTexture(data.Mda, AccessFlags.Read);
                builder.UseTexture(data.YCoCg, AccessFlags.Read);
                builder.UseTexture(data.MdcaClip, AccessFlags.Write);
                builder.UseTexture(data.LumaOut, AccessFlags.Write);
                builder.AllowPassCulling(false);

                builder.SetRenderFunc((PassData d, ComputeGraphContext ctx) =>
                {
                    var cmd = ctx.cmd;
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.SetComputeIntParam(d.Shader, Sgsr2ShaderIDs.Reset, d.Reset);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.PrevLumaHistory, d.PrevLuma);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.Mda, d.Mda);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.YCoCg, d.YCoCg);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.MdcaClip, d.MdcaClip);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.LumaHistory, d.LumaOut);
                    cmd.DispatchCompute(d.Shader, d.Kernel, d.GroupsX, d.GroupsY, 1);
                });
            }
        }
    }
}
