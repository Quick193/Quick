import numpy as np
import soundfile as sf
import os
import librosa

from main_fixed import CleanTravisScottAutotune

def generate_test_input(filename="test.wav", duration=3.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)  # A3 note
    sf.write(filename, y, sr)
    print(f"✅ Generated {filename} with {duration}s A3 tone")

def run_processing(input_path="test.wav", output_path="test_autotuned.wav"):
    if not os.path.exists(input_path):
        print("❌ Input file not found.")
        return

    try:
        y, sr = librosa.load(input_path, sr=22050, mono=True)
        print(f"🔍 Loaded input: {len(y)} samples at {sr} Hz")

        processor = CleanTravisScottAutotune(sr=sr)
        print("🎤 Running autotune + beat processing...")

        processed_y = processor.process_audio_clean(
            y, sr,
            strength=1.0,
            retune_speed=0.9,
            reverb_amount=0.4,
            beat_volume=0.5
        )

        if processed_y is None or len(processed_y) == 0:
            print("❌ Processed audio is empty or None.")
            return

        sf.write(output_path, processed_y, sr)
        print(f"✅ Output written to {output_path}")

    except Exception as e:
        print("❌ Exception occurred:", e)

if __name__ == "__main__":
    generate_test_input()
    run_processing()