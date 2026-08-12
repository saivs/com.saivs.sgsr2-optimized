using Sgsr2Optimized;
using UnityEngine;
using UnityEngine.Profiling;

namespace Sgsr2Optimized.Samples
{
    /// <summary>
    /// Runtime panel for on-device A/B comparison: variant selector, the
    /// feature toggles, FPS, and per-pass timings read from Unity's built-in
    /// profiling samplers (every Render Graph pass carries one named after
    /// the pass). GPU times require a platform where GPU recorders resolve;
    /// where they do not, the CPU-inline time of the pass is shown instead.
    /// Drop this on any GameObject in the scene.
    /// </summary>
    public class Sgsr2ComparisonHud : MonoBehaviour
    {
        static readonly string[] kVariantNames =
        {
            "Compute 2p", "Fragment 2p", "Compute 3p",
            "Fused Full", "Fused Lite", "Fused Packed", "Fused Ultra",
        };

        static readonly string[][] kVariantPasses =
        {
            new[] { "SGSR2 Convert", "SGSR2 Upscale" },
            new[] { "SGSR2 FS Convert", "SGSR2 FS Upscale" },
            new[] { "SGSR2 Convert3", "SGSR2 Activate3", "SGSR2 Upscale3" },
            new[] { "SGSR2 Fused Upscale" },
            new[] { "SGSR2 Fused Upscale" },
            new[] { "SGSR2 Pack", "SGSR2 Fused Upscale" },
            new[] { "SGSR2 Guides", "SGSR2 Fused Upscale" },
        };

        float _fps;
        float _fpsTimer;
        int _fpsFrames;
        static GUIStyle _style;
        static GUIStyle _smallStyle;
        static GUIStyle _toggleStyle;

        void Update()
        {
            _fpsFrames++;
            _fpsTimer += Time.unscaledDeltaTime;
            if (_fpsTimer >= 0.5f)
            {
                _fps = _fpsFrames / _fpsTimer;
                _fpsFrames = 0;
                _fpsTimer = 0f;
            }
        }

        static void DrawPassTime(string passName)
        {
            Recorder recorder = Recorder.Get(passName);
            if (recorder == null || !recorder.isValid)
                return;
            recorder.enabled = true;
            long gpuNs = recorder.gpuElapsedNanoseconds;
            long cpuNs = recorder.elapsedNanoseconds;
            if (gpuNs > 0)
                GUILayout.Label($"{passName}: <b>{gpuNs / 1e6:F3} ms</b> GPU", _smallStyle);
            else if (cpuNs > 0)
                GUILayout.Label($"{passName}: {cpuNs / 1e6:F3} ms CPU (GPU recorder unavailable)", _smallStyle);
        }

        void OnGUI()
        {
            _style ??= new GUIStyle(GUI.skin.label) { fontSize = 16, richText = true };
            _smallStyle ??= new GUIStyle(GUI.skin.label) { fontSize = 12, richText = true };
            _toggleStyle ??= new GUIStyle(GUI.skin.toggle) { fontSize = 14, richText = true };

            var sgsr = Sgsr2UpscaleFeature.Instance;

            GUILayout.BeginArea(new Rect(10, 10, 620, 360), GUI.skin.box);
            GUILayout.Label($"<b>SGSR2 Optimized</b>   {_fps:F0} FPS", _style);

            if (sgsr == null)
            {
                GUILayout.Label("Sgsr2UpscaleFeature is not active on the current renderer.", _smallStyle);
                GUILayout.EndArea();
                return;
            }

            int variantIndex = (int)sgsr.variant;
            int newIndex = GUILayout.SelectionGrid(variantIndex, kVariantNames, 4);
            if (newIndex != variantIndex)
                sgsr.variant = (Sgsr2Variant)newIndex;

            sgsr.halfPrecision = GUILayout.Toggle(sgsr.halfPrecision,
                $" Half Precision — <b>{(sgsr.halfPrecision ? "fp16" : "fp32")}</b>", _toggleStyle);

            if (sgsr.variant >= Sgsr2Variant.FusedFull)
            {
                if (sgsr.variant == Sgsr2Variant.FusedUltra)
                {
                    sgsr.liteGuides = GUILayout.Toggle(sgsr.liteGuides,
                        $" Lite Guides — <b>{(sgsr.liteGuides ? "2x2 dilation, no depthclip" : "full 3x3")}</b>", _toggleStyle);
                    GUILayout.BeginHorizontal();
                    GUILayout.Label($"Sharpness {sgsr.ultraSharpness:F2}", _smallStyle, GUILayout.Width(110));
                    sgsr.ultraSharpness = GUILayout.HorizontalSlider(sgsr.ultraSharpness, 0f, 1.5f, GUILayout.Width(220));
                    GUILayout.EndHorizontal();
                }
                sgsr.passthroughFloor = GUILayout.Toggle(sgsr.passthroughFloor,
                    $" Passthrough Floor — <b>{(sgsr.passthroughFloor ? "1 bilinear tap (no upscaler logic)" : "off")}</b>", _toggleStyle);
            }

            foreach (string passName in kVariantPasses[(int)sgsr.variant])
                DrawPassTime(passName);

            GUILayout.EndArea();
        }
    }
}
