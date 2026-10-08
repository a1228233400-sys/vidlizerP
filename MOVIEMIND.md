# MovieMind long-video mode

MovieMind is the long-form analysis layer added on top of Vidlizer.

The design is:

movie -> shots -> multi-frame shot observations -> narrative scenes -> characters/events ->
timeline -> persistent movie memory -> evidence retrieval -> grounded QA -> targeted rewatch

## Why it is different from the original Vidlizer flow

The original command intentionally caps whole-video frame extraction. MovieMind does not use
a whole-movie frame budget. It detects shots first and gives each shot a small adaptive visual
budget. Frames for one shot are extracted in a single FFmpeg process, sent to the model, then
deleted. The full movie is never loaded as one image batch.

## Install

~~~bash
pip install -e ".[longvideo]"
~~~

## Check the machine

~~~bash
movieai doctor movie.mp4
~~~

The doctor checks disk and available RAM. MovieMind defaults to one worker, does not auto-start
Ollama, and does not auto-download an Ollama model.

On macOS, MovieMind can use the Vidlizer MLX Whisper path. On Windows/Linux, the long-video
extra includes faster-whisper; its default is CPU INT8 with four threads and one worker. You can
override MOVIEMIND_WHISPER_MODEL, MOVIEMIND_WHISPER_DEVICE, or MOVIEMIND_WHISPER_COMPUTE_TYPE.

## Run the complete pipeline

~~~bash
movieai process movie.mp4
~~~

Profiles:

~~~bash
movieai process movie.mp4 --profile safe
movieai process movie.mp4 --profile balanced
movieai process movie.mp4 --profile deep
~~~

The auto profile selects settings from available RAM.

## Local model

~~~bash
PROVIDER=ollama MOVIEMIND_MODEL=qwen2.5vl:7b movieai process movie.mp4
~~~

The model must already be installed in Ollama.

## Ask questions

~~~bash
movieai status movie.mp4.movie.db

Status reports total shots, completed shots, visual coverage percentage, scenes,
characters, events, transcript segments, and evidence records.
movieai ask movie.mp4.movie.db "Why did the protagonist betray her?"
movieai ask movie.mp4.movie.db "When did the red box first appear?"
movieai ask movie.mp4.movie.db "What happened around 00:47:00?"
~~~

For a targeted rewatch of relevant windows:

~~~bash
movieai ask movie.mp4.movie.db "Was this woman shown earlier?" --deep
~~~

## Resumability

Every completed shot is committed immediately. Ctrl+C, a failed provider call, or a low-memory
pause does not discard successful work. Re-running the same process command continues from the
shots that are still pending or failed.

## Current scope

The long-video mode is the foundation of a MovieMind engine. It should be validated against
real feature-length films before being described as production-grade. The original vidlizer
command remains untouched.
