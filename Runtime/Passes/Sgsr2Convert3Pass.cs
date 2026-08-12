using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace Sgsr2Optimized
{
    /// <summary>
    /// Reference 3-pass compute variant, pass 1: dilated depth, motion,
    /// packed YCoCg and the transparency reactivity mask (final vs opaque
    /// color). Needs the URP opaque texture; without it the final color is
    /// bound to both inputs and the mask collapses to zero.
    /// </summary>
    internal sealed class Sgsr2Convert3Pass : ScriptableRenderPass
    {
        readonly Sgsr2UpscaleFeature m_Feature;
        readonly ComputeShader m_Shader;

        public Sgsr2Convert3Pass(Sgsr2UpscaleFeature feature, ComputeShader shader)
        {
            m_Feature = feature;
            m_Shader = shader;
        }

        class PassData
        {
            public ComputeShader Shader;
            public int Kernel;
            public TextureHandle Color;
            public TextureHandle OpaqueColor;
            public TextureHandle Depth;
            public TextureHandle Mda;
            public TextureHandle YCoCg;
            public Vector4 RenderSizeInfo;
            public Vector4 SgsrParams;
            public Vector4 C0, C1, C2, C3;
            public int GroupsX, GroupsY;
        }

        public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
        {
            var resourceData = frameData.Get<UniversalResourceData>();
            if (resourceData.isActiveTargetBackBuffer)
                return;
            Sgsr2FrameData d0 = m_Feature.GetFrameData(renderGraph, frameData);

            d0.Mda = renderGraph.CreateTexture(new TextureDesc(d0.RenderW, d0.RenderH)
            {
                name = "_SGSR2Mda3",
                format = GraphicsFormat.R16G16B16A16_SFloat,
                enableRandomWrite = true,
            });
            d0.YCoCg = renderGraph.CreateTexture(new TextureDesc(d0.RenderW, d0.RenderH)
            {
                name = "_SGSR2YCoCg3",
                format = GraphicsFormat.R32_UInt,
                enableRandomWrite = true,
            });

            bool hasOpaque = resourceData.cameraOpaqueTexture.IsValid();
            TextureHandle opaque = hasOpaque
                ? resourceData.cameraOpaqueTexture
                : resourceData.cameraColor;

            using (IComputeRenderGraphBuilder builder =
                renderGraph.AddComputePass("SGSR2 Convert3", out PassData data))
            {
                data.Shader = m_Shader;
                data.Kernel = m_Shader.FindKernel("Convert3");
                data.Color = resourceData.cameraColor;
                data.OpaqueColor = opaque;
                data.Depth = resourceData.cameraDepthTexture;
                data.Mda = d0.Mda;
                data.YCoCg = d0.YCoCg;
                data.RenderSizeInfo = d0.RenderSizeInfo;
                data.SgsrParams = d0.SgsrParams;
                data.C0 = d0.ClipToPrevClip.GetColumn(0);
                data.C1 = d0.ClipToPrevClip.GetColumn(1);
                data.C2 = d0.ClipToPrevClip.GetColumn(2);
                data.C3 = d0.ClipToPrevClip.GetColumn(3);
                data.GroupsX = (d0.RenderW + 7) / 8;
                data.GroupsY = (d0.RenderH + 7) / 8;

                builder.UseTexture(data.Color, AccessFlags.Read);
                if (hasOpaque)
                    builder.UseTexture(data.OpaqueColor, AccessFlags.Read);
                builder.UseTexture(data.Depth, AccessFlags.Read);
                builder.UseTexture(data.Mda, AccessFlags.Write);
                builder.UseTexture(data.YCoCg, AccessFlags.Write);
                builder.AllowPassCulling(false);

                builder.SetRenderFunc((PassData d, ComputeGraphContext ctx) =>
                {
                    var cmd = ctx.cmd;
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.RenderSizeInfo, d.RenderSizeInfo);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.SgsrParams, d.SgsrParams);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.ClipToPrevClip0, d.C0);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.ClipToPrevClip1, d.C1);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.ClipToPrevClip2, d.C2);
                    cmd.SetComputeVectorParam(d.Shader, Sgsr2ShaderIDs.ClipToPrevClip3, d.C3);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.InputColor, d.Color);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.InputOpaqueColor, d.OpaqueColor);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.InputDepth, d.Depth);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.Mda, d.Mda);
                    cmd.SetComputeTextureParam(d.Shader, d.Kernel, Sgsr2ShaderIDs.YCoCg, d.YCoCg);
                    cmd.DispatchCompute(d.Shader, d.Kernel, d.GroupsX, d.GroupsY, 1);
                });
            }
        }
    }
}
