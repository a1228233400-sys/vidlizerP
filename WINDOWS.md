# MovieMind on Windows

## Prerequisites

1. Install Python 3.10+ (Python 3.12 is a good baseline).
2. Install FFmpeg and make sure this works in a new terminal:

ffmpeg -version

3. Install a vision model provider. The simplest local path is Ollama with a vision-capable model already installed.

## Install MovieMind

From the cloned repository:

python -m pip install -e ".[longvideo]"

Windows PowerShell may require:

py -m pip install -e ".[longvideo]"

## Check the computer before processing

movieai doctor "C:\\path\\to\\movie.mp4"

The command reports CPU/RAM/disk state, required long-video dependencies, and can optionally preflight a provider:

movieai doctor "C:\\path\\to\\movie.mp4" --provider ollama --model qwen2.5vl:7b

MovieMind uses one worker by default and automatically selects safe/balanced/deep settings from available RAM. Low-memory conditions cause a pause/refusal rather than deliberately pushing the machine harder.

## Process a feature-length movie

movieai process "C:\\path\\to\\movie.mp4"

Or explicitly choose the safest profile:

movieai process "C:\\path\\to\\movie.mp4" --profile safe

Balanced:

movieai process "C:\\path\\to\\movie.mp4" --profile balanced

Deep:

movieai process "C:\\path\\to\\movie.mp4" --profile deep

The first run may take a long time. That is expected: the system analyzes the movie shot-by-shot, transcribes audio, builds scenes, characters, events, and global story memory. Completed shots are stored immediately.

## Check progress

movieai status "C:\\path\\to\\movie.mp4.movie.db"

Look for:

- shots equals shots_complete
- visual_coverage_pct near 100
- scenes greater than 0
- characters greater than 0
- events greater than 0
- global_memory true

## Generate a report

movieai report "C:\\path\\to\\movie.mp4.movie.db"

## Ask questions

movieai ask "C:\\path\\to\\movie.mp4.movie.db" "为什么男主最后背叛了女主？"
movieai ask "C:\\path\\to\\movie.mp4.movie.db" "第47分钟发生了什么？"
movieai ask "C:\\path\\to\\movie.mp4.movie.db" "这个红色盒子第一次出现在哪里？"

For an explicit high-density rewatch:

movieai ask "C:\\path\\to\\movie.mp4.movie.db" "这个女人之前是不是出现过？" --deep

The normal ask path can also trigger an automatic targeted rewatch when the first answer reports insufficient evidence.

## Transcription on Windows

MovieMind prefers the Vidlizer MLX Whisper path on macOS. On Windows/Linux it uses faster-whisper when installed. The default is CPU INT8 and one worker to keep resource use conservative.

Environment overrides:

set MOVIEMIND_WHISPER_MODEL=base
set MOVIEMIND_WHISPER_DEVICE=cpu
set MOVIEMIND_WHISPER_COMPUTE_TYPE=int8

PowerShell:

$env:MOVIEMIND_WHISPER_MODEL="base"
$env:MOVIEMIND_WHISPER_DEVICE="cpu"
$env:MOVIEMIND_WHISPER_COMPUTE_TYPE="int8"

If no Whisper backend is available, sidecar SRT/VTT or an embedded subtitle stream is used when available.

## Legal scope

MovieMind is intended for media the operator is legally entitled to access and analyze. It does not implement DRM bypass or unauthorized media acquisition.
