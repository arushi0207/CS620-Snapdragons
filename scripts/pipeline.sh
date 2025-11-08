python -m scripts.extract_from_video --video input.mp4 --out outputs/run3 --extractors facemap_3dmm --output-video outputs/run3/annotated.mp4

# Run LLaVA-OneVision evaluation (uses full video by default)
python -m scripts.llm_eval --video input.mp4 --out outputs/run3 --model lmms-lab/LLaVA-OneVision-1.5-4B-stage0 --max-new-tokens 512 --temperature 0.2
