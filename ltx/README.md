# ltx — LTX-2 Video Workflows

Skills for working with Lightricks **LTX-2** models.

## Skills

| Skill | What it does |
|-------|--------------|
| [`lipdub`](skills/lipdub/) | Set up and run **LTX-2.3 LipDub** video dubbing — re-dub the speech in an existing clip to new dialogue while preserving the speaker's face/scene/motion (IC-LoRA on the LTX-2.3-22B distilled model). Clones the [LTX-2](https://github.com/Lightricks/LTX-2) code, documents the Hugging Face gate/auth requirement, downloads the gated weights (base + spatial upscaler + LipDub IC-LoRA + Gemma text encoder) via `download_lipdub_models.sh`, and runs the verified `ltx_pipelines.lipdub` CLI via `run_lipdub.sh` (frame count/fps from the reference video; low-VRAM `fp8-cast`/offload option). |
