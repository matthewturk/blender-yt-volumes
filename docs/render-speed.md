# Render speed and GPU memory: quick reference

Measured on dd0087 (1080p stereo, RTX 2060 6 GB, Vulkan backend).

**Done for you by the pipeline**
- 3D viewports in Material Preview/Rendered switch to Solid while rendering
  (a shaded viewport made stereo EEVEE renders 5-7x slower).
- Cycles: its GPU path-state pool is capped (`CYCLES_CONCURRENT_STATES_MAX`,
  set it yourself to override); about 1.7 GB less GPU memory, same speed.
- Cycles with yt_idv composites on the CPU (faster here than the GPU
  compositor, and it can't exhaust GPU memory).
- F12 keeps the frame in Render Result; only Render Animation writes files.

**Settings worth checking**
- Output > PNG Compression 0%: saves about 0.4 s per stereo frame, same pixels.
- EEVEE Shadows > Pool Size: as small as the scene allows (128-256 MB). The
  512 MB default pushed a 6 GB card into memory thrashing (pass 1 2.3 s -> 3.5 s).
  With no lights, turn Shadows off.
- Cycles: 32-64 samples with the OptiX denoiser (not OpenImageDenoise on the CPU).
- Smooth Surface Edges costs about 0.2-0.5 s per frame; turn it off when nothing
  overlaps the volume.

**When frames are slow**
- Watch `nvidia-smi -l 1`: within a few hundred MB of the card's limit, frames
  slow down sharply, then crash. Lower the shadow pool or resolution.
- Stereo doubles the time, not the memory.
- The first frame pays the dataset load (about 90 s for dd0087).
- `samples_per_cell` on the volume is the yt_idv speed/quality trade-off.
