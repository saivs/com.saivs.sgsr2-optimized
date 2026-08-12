# SGSR2 Optimized for Unity URP

Snapdragon Game Super Resolution 2 as a self-contained URP renderer feature, optimized until it became practical on low-tier mobile GPUs. No TAA dependency: the feature applies its own camera jitter and reconstructs motion from depth.

The faithful port of the reference costs 40 ms on an Adreno 610 at 2400x1080 (Render Scale 0.5). That number is the reason this package exists. The final kernel does the same job in 10 ms.

## Results

Adreno 610 (Snapdragon 685, Redmi Note 12), output 2400x1080, Render Scale 0.5, static camera. Each row uses the best precision for that kernel, all rows measured in one run on a cooled device:

| Variant | Time |
|---|---|
| Reference port, 2 compute passes | 40 ms |
| Fused fragment: the upscale is the final blit, fp16 | 31.9 ms |
| Lite: 5 taps, fp16 | 15.1 ms |
| Ultra: guides prepass + 2 fetches per pixel, fp32 | 10.1 ms |

Passthrough floor (one bilinear tap in the same pass structure, the theoretical minimum of any upscaler): 2 ms. The final kernel is one texture fetch away from it, and that fetch is the history read, which is the temporal algorithm itself.

Adreno 660 (Snapdragon 888) for scale: reference 7.9 ms, Lite fp16 3.6 ms, Ultra 3.2 ms at Render Scale 0.5; at native scale (pure anti-aliasing) Lite runs at 3.0 ms.

## What is inside

- **Reference compute port** of the 2-pass SGSR2, adapted to reversed-Z. Kept as the baseline for honest comparison.
- **Fused fragment path**: the upscale replaces URP's final blit entirely (MRT0 = backbuffer, MRT1 = history), convert math runs inline, zero intermediate buffers.
- **Lite kernel**: always 5 taps, 2x2 depth dilation, no depth-clip weighting. Best at native scale and on faster GPUs.
- **Ultra kernel**: a render-resolution guides pass folds the neighborhood analysis (dilated depth, depth clip, motion, 3x3 rectbox, average luma) into one texel; the display pass reads 2 textures per pixel on a still camera. Best at aggressive downscale on weak GPUs.
- **Still camera as a compile variant**: measured on Adreno, a branch around a texture fetch costs more than it saves (registers for both paths plus lost fetch scheduling). The feature flips the SGSR_STILL keyword only when the camera starts or stops moving; the still variant carries the center color in the guide's motion slot (motion is identically zero) and drops the color fetch entirely.
- **Passthrough floor toggle** for measuring the device's theoretical minimum.
- **Comparison HUD sample** with every toggle and per-pass GPU timings.

## Choosing a kernel

The optimum depends on hardware tier and render scale. The Ultra advantage is proportional to the squared display/render resolution ratio (its analysis runs where pixels are few); Lite is a single pass, so it wins when that ratio approaches 1:

| Situation | Pick |
|---|---|
| Low-tier GPU (Adreno 610 class), scale 0.5 and below | Ultra, fp32 |
| Mid-tier GPU, aggressive downscale | Ultra |
| Native scale, anti-aliasing only | Lite, fp16 |
| Mid-tier GPU, scale around 0.7 and above | Lite, fp16 |

fp16 rule of thumb from measurements on two GPUs: it helps large kernels (29% on the full kernel), and loses in tiny ones where boundary conversions outweigh the savings. The HUD makes checking on your target device a ten second job.

## Installation

Requires Unity 6 (URP 17, Render Graph). Add via Package Manager as a git URL or a local folder:

```
https://github.com/saivs/unity-sgsr-optimized.git
```

The Runtime code compiles into the URP runtime assembly through an `.asmref`. This is what makes the package work with stock URP: the fused path needs two internal URP entry points (`SwitchActiveTexturesToBackbuffer` to replace the final blit, `GetFinalBlitScaleBias` for backbuffer orientation). No URP source modifications, no fork.

## Setup

1. Add **Sgsr2 Upscale Feature** to your URP Renderer's feature list.
2. Pick a **Variant**: three faithful reference pipelines (2-pass compute, 2-pass fragment, 3-pass compute) for baseline comparison, or the optimized Fused family (Full, Lite, Packed, Ultra). The feature enqueues only the passes the selected pipeline needs; each pass lives in its own file under `Runtime/Passes`.
3. Set **Render Scale** on the URP asset (0.5 is the intended working point, 1.0 gives pure temporal anti-aliasing).
4. Keep TAA and MSAA off. The feature jitters the camera projection itself.
5. For the 3-pass variant's reactivity mask, enable **Opaque Texture** on the URP asset; without it the mask gracefully collapses to zero.
6. Optional: import the Comparison HUD sample and drop `Sgsr2ComparisonHud` on any GameObject.

## Timings

Every pipeline stage is a separate Render Graph pass with its own built-in profiling sampler, so the passes show up by name in the Unity Profiler, the Frame Debugger, RenderDoc and Snapdragon Profiler. The Comparison HUD reads them through `UnityEngine.Profiling.Recorder` and shows GPU times where the platform resolves GPU recorders, falling back to the CPU-inline time otherwise.

Tile-based GPU warning for external timers: timestamps recorded inside a render pass execute per tile and return garbage; a correct Vulkan timer must place its queries outside the render pass (native plugin events: `kUnityVulkanRenderPass_EnsureOutside`).

## What did not work

Negative results, so you do not repeat them on Adreno:

- Splitting the 128-bit guide fetch into two narrow fetches behind stillness branches: slower. Branches around fetches cost registers and scheduling.
- Checkerboard 2x2 interleave (half the blocks coast on history each frame): much slower. The wave covers 64-128 pixels, so a 2x2 pattern guarantees full divergence in every wave.
- UAV writes and formats that disable UBWC compression: measurable loss on history, no gain elsewhere.
- Moving work to compute: at best equal to the fragment path in fp32, and the fragment compiler extracts far more from fp16.
- Packed input via two gathers: loses on Adreno 610, wins about 15% on Adreno 660 at extreme downscale. Generation dependent, kept as a toggle.

## Limitations

- Motion is camera-only. Static geometry reprojects exactly; dynamic objects rely on the rectbox clamp and show short trails instead of tracked motion. Per-object motion vectors are a possible extension (the original consumes a velocity input).
- The fused path targets the backbuffer directly, so it must be the last thing in the frame; screen-space UI (overlay canvas) composes on top normally.
- Requires URP 17 (Unity 6) Render Graph. The internal API surface the asmref touches may drift in future URP versions.
- HDR output swapchains and XR are untested.

## License and attribution

BSD 3-Clause, see [LICENSE.md](LICENSE.md). Derived from [Snapdragon Game Super Resolution 2](https://github.com/SnapdragonGameStudios/snapdragon-gsr) by Qualcomm Innovation Center, Inc. Snapdragon is a trademark of Qualcomm; this project is not affiliated with or endorsed by Qualcomm. URP integration, the TAA-free setup, the Lite and Ultra kernels and the mobile optimization work by Denis Zhukov.
