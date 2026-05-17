# LM Studio VLM Judge

Preferred local model for current experiments:

`nsfwvision-v4-qwen3.5-9b`

The user's manual LM Studio tests found generic `Qwen2.5-VL-7B-Instruct`
unreliable for VaM pose/family classification. It over-guessed and produced
generic safety/moral commentary. `nsfwvision-v4-qwen3.5-9b` gave more useful
technical pose descriptions on VaM-like screenshots.

This pipeline uses LM Studio only through a local OpenAI-compatible endpoint,
for example `http://localhost:1234/v1`. No cloud API is called and no model is
downloaded by this project.

Dry run:

```powershell
$env:PYTHONPATH='src'
python -m vam_timeline_ai.cli run-lmstudio-vlm-judge-v0 `
  --requests data\runs\clean_v3\audits\ml_assisted_cowgirl_review_v1\visual_judge_requests.jsonl `
  --base-url http://localhost:1234/v1 `
  --model "nsfwvision-v4-qwen3.5-9b" `
  --out-jsonl data\runs\clean_v3\audits\ml_assisted_cowgirl_review_v1\visual_judge_results.jsonl `
  --out-raw-dir data\runs\clean_v3\audits\ml_assisted_cowgirl_review_v1\visual_judge_raw_outputs `
  --dry-run true
```

Live local run after starting LM Studio's local server:

```powershell
$env:PYTHONPATH='src'
python -m vam_timeline_ai.cli run-lmstudio-vlm-judge-v0 `
  --requests data\runs\clean_v3\audits\ml_assisted_cowgirl_review_v1\visual_judge_requests.jsonl `
  --base-url http://localhost:1234/v1 `
  --model "nsfwvision-v4-qwen3.5-9b" `
  --out-jsonl data\runs\clean_v3\audits\ml_assisted_cowgirl_review_v1\visual_judge_results.jsonl `
  --out-raw-dir data\runs\clean_v3\audits\ml_assisted_cowgirl_review_v1\visual_judge_raw_outputs `
  --dry-run false
```

For judging, prefer a real VaM capture contact sheet over a video file. A
single 8- or 16-frame contact sheet lets the model see pose and motion context
without making one slow request per frame. MP4/GIF inputs remain fallbacks
because OpenAI-compatible local servers vary in how well they support them.

Visual judge output is review-assist only. It is not manual truth, not training
truth, and not an auto-label source.
