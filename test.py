from faster_whisper import WhisperModel

model = WhisperModel("medium.en", device="cpu", compute_type="int8")

segments, _ = model.transcribe("test_audio.wav")
transcription = " ".join(segment.text for segment in segments)
print("Transcription:", transcription)
