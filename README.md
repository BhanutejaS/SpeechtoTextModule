**Speech-to-Text (STT) Module**

The Speech-to-Text module forms the front end of the real-time speech interaction pipeline.
It continuously captures live microphone input, segments it into manageable audio chunks using RMS-based adaptive Voice Activity Detection (VAD), and transcribes the speech using Whisper Tiny (OpenAI) for low-latency inference.

To ensure responsiveness, the module employs multithreaded audio streaming with sounddevice callbacks, queue-based buffering, and asynchronous Flask–SocketIO communication — allowing simultaneous audio capture, transcription, and data transfer.

 **Key Engineering Features**

Adaptive Silence Detection: Dynamically adjusts thresholds based on real-time RMS averages for accurate speech segmentation.

Echo & Noise Suppression: Uses a rolling RMS baseline and playback gating to prevent self-capture during TTS playback.

Chunk-Level Segmentation: Buffers audio streams into 1–2 s segments before passing them to Whisper for near real-time decoding.

Concurrency & Low Latency: Thread-safe queues (audio_q, tts_synth_q, play_q) handle parallel capture / processing to minimize blocking.

Context-Aware Filtering: Filters short or repetitive utterances and ignores echoed assistant responses via text-similarity metrics (difflib).

Latency Tracking: Records STT latency statistics for end-to-end benchmarking and optimization.

**Technical Summary**

Model: Whisper Tiny (OpenAI) – optimized for low-latency streaming

Latency: < 500 ms average STT response

Frameworks: Flask–SocketIO | SoundDevice | NumPy | Whisper

Goal: Real-time, context-aware speech-to-speech interaction
