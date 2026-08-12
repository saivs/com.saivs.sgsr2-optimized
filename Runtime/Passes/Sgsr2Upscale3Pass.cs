using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// Reference 3-pass compute variant, pass 3: gather-based 9-tap YCoCg
    /// temporal upscale with reactivity and luma stability. Terminal pass.
    /// </summary>
    internal sealed class Sgsr2Upscale3Pass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly ComputeShader m_Shader;

        public Sgsr2Upscale3Pass(Sgsr2UpscaleFeature feature, ComputeShader shader)
        {
            m_Feature = feature;
            m_Shader = shader;
        }

        class PassData
        {
            public ComputeShader Shader;
            public int Kernel;
            public TextureHandle PrevHistory;
            public TextureHandle MdcaClip;
            public TextureHandle YCoCg;
            public TextureHandle HistoryOut;
            public TextureHandle SceneOut;
            public Vector4 RenderSizeInfo;
            public Vector4 DisplaySizeInfo;
            public Vector4 Jitter;
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

            TextureHandle sceneOut = renderGraph.CreateTexture(new TextureDesc(d0.DisplayW, d0.DisplayH)
            {
                name = "_SGSR2Output3",
                format = GraphicsFormat.R16G16B16A16_SFloat,
                enableRandomWrite = true,
            });

            using (IComputeRenderGraphBuilder builder =
                renderGraph.AddComputePass("SGSR2 Upscale3", out PassData data))
            {
                data.Shader = m_Shader;
                data.Kernel = m_Shader.FindKernel("Upscale3");
                data.PrevHistory = d0.PrevHistory;
                data.MdcaClip = d0.MdcaClip;
                data.YCoCg = d0.YCoCg;
                data.HistoryOut = d0.NextHistory;
                data.SceneOut = sceneOut;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.DisplaySizeInfo = d0.DisplaySizeInfo;
                data.Jitter = d0.Jitter;
                data.SgsrParams = d0.SgsrParams;
                data.Reset = d0.ResetHistory ? 1 : 0;
                data.GroupsX = (d0.DisplayW + 7) / 8;
                data.GroupsY = (d0.DisplayH + 7) / 8;

                builder.UseTexture(data.PrevHistory, AccessFlags.Read);
                builder.UseTexture(data.MdcaClip, AccessFlags.Read);
                builder.UseTexture(data.YCoCg, AccessFlags.Read);
                builder.UseTexture(data.HistoryOut, AccessFlags.Write);
                builder.UseTexture(data.SceneOut, AccessFlags.Write);
                builder.AllowPassCulling(false);

                builder.SetRenderFunc((PassData d, ComputeGraphContext ctx) =>
                {
                    var cmd = ctx.cmd;
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.DisplaySizeInfo, d.DisplaySizeInfo);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.JitterOffset, d.Jitter);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.SetComputeIntParam(d.Shader, Sgsr2ShaderIDs.Reset, d.Reset);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.PrevHistoryOutput, d.PrevHistory);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.MdcaClip, d.MdcaClip);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.YCoCg, d.YCoCg);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.HistoryOutput, d.HistoryOut);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.SceneColorOutput, d.SceneOut);
                    cmd.DispatchCompute(d.Shader, d.Kernel, d.GroupsX, d.GroupsY, 1);
                });
            }

            resourceData.cameraColor = sceneOut;
            d0.FinishFrame();
        }
    }
}
