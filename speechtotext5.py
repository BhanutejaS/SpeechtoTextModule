import whisper
import sounddevice as sd
import numpy as np
import os
import queue
import threading
import time
import soundfile as sf
import noisereduce as nr
import librosa
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv

# ----------------------------- CONFIG -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisper model (tiny English for low latency)
model = whisper.load_model("tiny.en")

# Audio parameters
samplerate = 16000
blocksize = 512
q = queue.Queue()

# Adaptive RMS pause detection parameters
RMS_HISTORY = 30          # Tracks recent RMS to adapt to background noise
SILENCE_MULTIPLIER = 1.5  # Speech threshold multiplier
MIN_CHUNK_SIZE = samplerate // 2  # Minimum 0.5 sec audio

rms_values = []

# -------------------------- FUNCTIONS ----------------------------

def audio_callback(indata, frames, time_info, status):
    """Callback to put microphone audio into the queue."""
    if status:
        print(status)
    q.put(indata.copy())

def is_speech(audio_chunk):
    """Check if audio chunk is speech using adaptive RMS threshold."""
    rms = np.sqrt(np.mean(audio_chunk**2))
    rms_values.append(rms)
    if len(rms_values) > RMS_HISTORY:
        rms_values.pop(0)
    avg_rms = np.mean(rms_values)
    threshold = avg_rms * SILENCE_MULTIPLIER
    return rms > threshold

def speculative_gpt_tts_stream(user_text: str):
    """Send transcribed text to GPT, get response, and play via noise-reduced TTS."""
    try:
        print(f"\n[STT -> GPT] {user_text}")

        # GPT response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_text},
            ]
        )
        gpt_reply = response.choices[0].message.content.strip()
        print(f"[GPT -> TTS] {gpt_reply}")

        # Generate TTS (Google TTS as example)
        tts = gTTS(gpt_reply, lang="en")
        tts.save("response_raw.wav")

        # Load audio for noise reduction
        audio, sr = librosa.load("response_raw.wav", sr=None)

        # Apply noise reduction
        reduced_audio = nr.reduce_noise(y=audio, sr=sr)

        # Normalize audio to avoid distortion
        reduced_audio = reduced_audio / np.max(np.abs(reduced_audio))

        # Save cleaned audio
        sf.write("response_clean.wav", reduced_audio, sr)

        # Play once
        os.system("afplay response_clean.wav")  # Mac
        # Linux: os.system("mpg123 response_clean.wav")
        # Windows: os.system("start response_clean.wav")

    except Exception as e:
        print(f"[Error in GPT/TTS]: {e}")

def live_transcribe_gpt_tts_stream():
    print("Start speaking (Ctrl+C to stop)...")
    buffer = np.zeros((0,), dtype=np.float32)
    last_speech_time = None
    PAUSE_THRESHOLD = 1.2  # seconds of silence = end of utterance
    MIN_WORDS = 3          # require at least 3 words before sending to GPT

    try:
        with sd.InputStream(
            samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback
        ):
            while True:
                chunk = q.get()
                audio_data = chunk[:, 0]
                buffer = np.concatenate((buffer, audio_data))

                if is_speech(audio_data):
                    last_speech_time = time.time()

                # Trigger GPT only after long pause
                if buffer.size >= MIN_CHUNK_SIZE and last_speech_time and (time.time() - last_speech_time > PAUSE_THRESHOLD):
                    audio_chunk = buffer.copy()
                    buffer = np.zeros((0,), dtype=np.float32)
                    last_speech_time = None

                    result = model.transcribe(audio_chunk, fp16=False)
                    text = result["text"].strip()

                    # Word filter: skip short or incomplete transcriptions
                    if text and (len(text.split()) >= MIN_WORDS or text.endswith("?")):
                        print(f"[STT Candidate] {text}")
                        threading.Thread(
                            target=speculative_gpt_tts_stream,
                            args=(text,),
                            daemon=True
                        ).start()
                    else:
                        print(f"[STT Skipped] {text}")

    except KeyboardInterrupt:
        print("\nStopped live transcription.")

# --------------------------- RUN ---------------------------
if __name__ == "__main__":
    live_transcribe_gpt_tts_stream()
