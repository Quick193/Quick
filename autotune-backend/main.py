from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
from pydub import AudioSegment
import soundfile as sf
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import uuid
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# Check for required dependencies
try:
    from scipy import signal
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: SciPy not available, using fallback methods")
    SCIPY_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'ogg'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_kick(duration, sr):
    t = np.linspace(0, duration, int(sr * duration), False)
    freq = 50
    decay = 0.5
    envelope = np.exp(-t / decay)
    wave = np.sin(2 * np.pi * freq * t) * envelope
    return wave / np.max(np.abs(wave))

def generate_snare(duration, sr):
    t = np.linspace(0, duration, int(sr * duration), False)
    noise = np.random.normal(0, 1, len(t))
    envelope = np.exp(-t / 0.05)
    wave = noise * envelope
    return wave / np.max(np.abs(wave))

def generate_hihat(duration, sr):
    t = np.linspace(0, duration, int(sr * duration), False)
    noise = np.random.normal(0, 1, len(t))
    envelope = np.exp(-t / 0.01)
    wave = noise * envelope
    return wave / np.max(np.abs(wave))

def generate_bass_note(freq, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), False)
    envelope = np.exp(-t / 0.5)
    wave = np.sin(2 * np.pi * freq * t) * envelope
    return wave / np.max(np.abs(wave))

def generate_guitar_chord(freqs, duration, sr):
    """Generate a guitar chord with multiple frequencies and harmonics"""
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = np.zeros_like(t)
    
    # Add fundamental and harmonics for each note in the chord
    for freq in freqs:
        # Fundamental
        wave += 0.6 * np.sin(2 * np.pi * freq * t)
        # First harmonic
        wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        # Second harmonic
        wave += 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    
    # Guitar-like envelope with quick attack and longer decay
    envelope = np.exp(-t / 0.8)
    wave = wave * envelope
    
    # Add some guitar-like distortion
    wave = np.tanh(wave * 0.8)
    
    return wave / np.max(np.abs(wave))

def generate_electric_guitar(freq, duration, sr, style='lead'):
    """Generate electric guitar sounds with different styles"""
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = np.zeros_like(t)
    
    if style == 'lead':
        # Lead guitar with more harmonics and distortion
        wave += 0.5 * np.sin(2 * np.pi * freq * t)
        wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        wave += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
        wave += 0.1 * np.sin(2 * np.pi * freq * 4 * t)
        
        # Electric guitar envelope with sustain
        envelope = np.exp(-t / 1.2)
        wave = wave * envelope
        
        # Heavy distortion for electric sound
        wave = np.tanh(wave * 2.0)
        
    elif style == 'rhythm':
        # Rhythm guitar with palm muting effect
        wave += 0.4 * np.sin(2 * np.pi * freq * t)
        wave += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        
        # Quick attack, short decay for palm muting
        envelope = np.exp(-t / 0.3)
        wave = wave * envelope
        
        # Moderate distortion
        wave = np.tanh(wave * 1.5)
    
    return wave / np.max(np.abs(wave))

def generate_flute(freq, duration, sr):
    """Generate flute sounds with breathy, airy characteristics"""
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = np.zeros_like(t)
    
    # Flute has a pure, breathy tone with air noise
    # Fundamental frequency
    wave += 0.7 * np.sin(2 * np.pi * freq * t)
    # First harmonic (octave)
    wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    # Second harmonic
    wave += 0.15 * np.sin(2 * np.pi * freq * 3 * t)
    
    # Add breathy air noise
    air_noise = np.random.normal(0, 0.1, len(t))
    air_envelope = np.exp(-t / 0.5)
    wave += air_noise * air_envelope * 0.2
    
    # Flute envelope with slow attack and long sustain
    envelope = np.exp(-t / 1.5)
    wave = wave * envelope
    
    # Add slight vibrato for natural flute sound
    vibrato_freq = 6.0  # 6 Hz vibrato
    vibrato_depth = 0.02
    vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t)
    wave = wave * vibrato
    
    return wave / np.max(np.abs(wave))

class CleanTravisScottAutotune:
    def __init__(self, sr=22050):
        self.sr = sr
        self.hop_length = 512
        self.frame_length = 2048

    def detect_key_simple(self, y, sr):
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_profile = np.sum(magnitudes, axis=1)
        key_idx = np.argmax(pitch_profile) % 12
        is_major = True
        return key_idx, is_major

    def get_scale_notes(self, key_idx, is_major):
        chromatic = [261.63 * (2 ** (i / 12)) for i in range(12)]  # Middle C base
        intervals = [0, 2, 4, 5, 7, 9, 11] if is_major else [0, 2, 3, 5, 7, 8, 10]
        root = chromatic[key_idx]
        scale = [root * (2 ** (i / 12)) for i in intervals]
        return scale

    def smooth_pitch_gentle(self, f0, window_size=3):
        f0_smooth = np.copy(f0)
        for i in range(len(f0)):
            window = f0[max(0, i - window_size // 2):min(len(f0), i + window_size // 2 + 1)]
            valid = window[window > 0]
            f0_smooth[i] = np.mean(valid) if len(valid) > 0 else f0[i]
        return f0_smooth

    def find_nearest_note(self, freq, scale_freqs):
        if freq <= 0:
            return freq
        differences = np.abs(np.array(scale_freqs) - freq)
        nearest_idx = np.argmin(differences)
        nearest_freq = scale_freqs[nearest_idx]
        cents_diff = abs(1200 * np.log2(nearest_freq / freq))
        if cents_diff < 100:  # Increased to 100 cents for stronger correction
            return nearest_freq
        else:
            return freq

    def soft_clip(self, y, drive=1.0):
        # Gentler soft clipping to prevent distortion
        clipped = np.tanh(y * drive)
        max_val = np.max(np.abs(clipped))
        if max_val > 0:
            return clipped / max_val * 0.95  # Leave some headroom
        return clipped

    def clean_reverb(self, y, sr, room_size=0.5, wet_level=0.3, pre_delay_ms=0):
        impulse = np.zeros(int(sr * 0.5))
        impulse[0] = 1
        delay_samples = int(pre_delay_ms * sr / 1000)
        decay = np.exp(-np.linspace(0, 0.5, len(impulse)) / room_size)
        impulse += np.random.normal(0, 0.1, len(impulse)) * decay
        if delay_samples > 0 and delay_samples < len(decay):
            impulse[delay_samples:] += decay[:-delay_samples] * 0.5
        wet = signal.convolve(y, impulse, mode='same')
        return (1 - wet_level) * y + wet_level * wet

    def add_delay(self, y, sr, delay_ms=30, feedback=0.3, mix=0.3):
        delay_samples = int(delay_ms * sr / 1000)
        output = np.copy(y)
        delayed = np.zeros_like(y)
        if delay_samples < len(y):
            delayed[delay_samples:] = y[:-delay_samples] * feedback
            output += delayed
        return (1 - mix) * y + mix * output

    def enhance_vocals(self, y, sr):
        """Enhanced vocal processing with quality preservation"""
        # High-quality vocal enhancement
        enhanced = np.copy(y)
        
        # Gentle high-frequency boost for clarity
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        spectrum = np.fft.rfft(y)
        
        # Boost presence frequencies (2-4kHz) for clarity
        presence_boost = 1 + 0.15 * np.exp(-((freqs - 3000) ** 2) / (2 * 1000 ** 2))
        
        # Gentle low-mid boost for warmth (200-800Hz)
        warmth_boost = 1 + 0.1 * np.exp(-((freqs - 500) ** 2) / (2 * 300 ** 2))
        
        # Apply boosts
        enhanced_spectrum = spectrum * presence_boost * warmth_boost
        
        # Convert back to time domain
        enhanced = np.fft.irfft(enhanced_spectrum, n=len(y))
        
        # Gentle saturation for harmonics
        enhanced = np.tanh(enhanced * 1.1) / 1.1
        
        # Preserve dynamics with gentle limiting
        max_val = np.max(np.abs(enhanced))
        if max_val > 0.99:
            enhanced = enhanced * 0.99 / max_val
            
        return enhanced

    def process_audio_clean(self, y, sr, strength=1.0, retune_speed=0.9, reverb_amount=0.4, beat_volume=0.5):
        def gentle_vocal_eq(y, sr):
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            spectrum = np.fft.rfft(y)
            gain = 1 + 0.2 * np.exp(-((freqs - 1500) ** 2) / (2 * 700 ** 2))  # Boost 1.5kHz band
            eq_spectrum = spectrum * gain
            return np.fft.irfft(eq_spectrum, n=len(y))
    
        try:
            print("🎵 Starting enhanced autotune processing...")
            y = y / (np.max(np.abs(y)) + 1e-8)
            key_idx, is_major = self.detect_key_simple(y, sr)
            scale_freqs = self.get_scale_notes(key_idx, is_major)
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            scale_type = "Major" if is_major else "Minor"
            print(f"   Key: {note_names[key_idx]} {scale_type}")
            f0, _, _ = librosa.pyin(y, fmin=80, fmax=800, sr=sr, hop_length=self.hop_length, n_thresholds=100, resolution=0.1)
            f0 = np.nan_to_num(f0, nan=0.0)
            print(f"   Found {np.sum(f0 > 0)} voiced frames")
            f0_smooth = self.smooth_pitch_gentle(f0, window_size=3)
            target_pitches = np.zeros_like(f0_smooth)
            for i, freq in enumerate(f0_smooth):
                if freq > 0:
                    target_pitches[i] = self.find_nearest_note(freq, scale_freqs)
            corrected_y = np.copy(y)
            for i in range(0, len(f0_smooth) - 1):
                if f0_smooth[i] > 0 and target_pitches[i] > 0:
                    pitch_ratio = target_pitches[i] / f0_smooth[i]
                    correction_cents = 1200 * np.log2(pitch_ratio)
                    if abs(correction_cents) > 25:  # Lower threshold for more corrections
                        final_correction = correction_cents * strength * 0.8  # Increased strength for better correction
                        max_correction_per_frame = 80 / (retune_speed + 0.1)  # Increased max correction
                        final_correction = np.clip(final_correction, -max_correction_per_frame, max_correction_per_frame)
                        semitone_shift = final_correction / 100
                        semitone_shift = np.clip(semitone_shift, -1.5, 1.5)  # Slightly increased range
                        start_sample = int(i * self.hop_length)
                        end_sample = min(start_sample + self.frame_length, len(y))
                        if end_sample > start_sample and abs(semitone_shift) > 0.08:
                            segment = y[start_sample:end_sample]
                            if np.mean(np.abs(segment)) > 0.01:
                                try:
                                    shifted_segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=semitone_shift)
                                    if len(shifted_segment) == len(segment):
                                        fade_samples = 16
                                        if len(segment) > fade_samples * 2:
                                            fade_in = np.linspace(0, 1, fade_samples)
                                            shifted_segment[:fade_samples] *= fade_in
                                            segment[:fade_samples] *= (1 - fade_in)
                                            # shifted_segment[:fade_samples] += segment[:fade_samples]
                                            fade_out = np.linspace(1, 0, fade_samples)
                                            shifted_segment[-fade_samples:] *= fade_out
                                            segment[-fade_samples:] *= (1 - fade_out)
                                            # shifted_segment[-fade_samples:] += segment[-fade_samples:]
                                        replace_len = min(len(shifted_segment), len(corrected_y) - start_sample)
                                        corrected_y[start_sample:start_sample + replace_len] = shifted_segment[:replace_len]
                                except:
                                    continue
            # Enhanced voice processing with quality preservation
            # Apply gentle but effective voice enhancement
            corrected_y = self.enhance_vocals(corrected_y, sr)
            
            # Very subtle reverb - minimal to avoid echo
            if reverb_amount > 0:
                # Calculate very subtle reverb based on vocal intensity
                vocal_rms = np.sqrt(np.mean(corrected_y**2))
                dynamic_wet_level = reverb_amount * 0.1 * min(1.0, vocal_rms * 0.5)  # 10% louder reverb
                corrected_y = self.clean_reverb(corrected_y, sr, room_size=0.08, wet_level=dynamic_wet_level, pre_delay_ms=1)
            
            # Very subtle delay for depth
            corrected_y = self.add_delay(corrected_y, sr, delay_ms=10, feedback=0.05, mix=0.08)  # Much gentler delay
            
            # Quality-preserving compression
            max_val = np.max(np.abs(corrected_y))
            if max_val > 0.98:  # Only compress if absolutely necessary
                corrected_y = corrected_y * 0.98 / max_val

            # Generate backing beat
            print("🥁 Generating backing beat...")
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            if tempo < 40 or tempo > 200:
                print('⚠️ Tempo out of bounds, setting default 120 BPM')
                tempo = 120
            print(f"   Detected tempo: {tempo:.2f} BPM")
            root_freq = 65.41 * (2 ** (key_idx / 12))  # C2 for key_idx=0
            beat_length = len(corrected_y)
            beat_duration = beat_length / sr
            beat = np.zeros(beat_length)
            beat_interval = 60 / tempo  # seconds per beat
            for bar in range(int(beat_duration / (4 * beat_interval)) + 1):
                bar_start = bar * 4 * beat_interval
                for beat_pos in [0, 2]:  # Kick on beats 1 and 3
                    t = bar_start + beat_pos * beat_interval
                    if t >= beat_duration:
                        break
                    start_sample = int(t * sr)
                    kick = generate_kick(0.1, sr)
                    end_sample = min(start_sample + len(kick), beat_length)
                    beat[start_sample:end_sample] += kick[:end_sample - start_sample]
                for beat_pos in [1, 3]:  # Snare on beats 2 and 4
                    t = bar_start + beat_pos * beat_interval
                    if t >= beat_duration:
                        break
                    start_sample = int(t * sr)
                    snare = generate_snare(0.1, sr)
                    end_sample = min(start_sample + len(snare), beat_length)
                    beat[start_sample:end_sample] += snare[:end_sample - start_sample]
                for eighth in range(8):  # Hi-hat on every eighth note
                    t = bar_start + (eighth / 2.0) * beat_interval
                    if t >= beat_duration:
                        break
                    start_sample = int(t * sr)
                    hihat = generate_hihat(0.05, sr)
                    end_sample = min(start_sample + len(hihat), beat_length)
                    beat[start_sample:end_sample] += hihat[:end_sample - start_sample]
                for beat_pos in range(4):  # Bass on every beat
                    t = bar_start + beat_pos * beat_interval
                    if t >= beat_duration:
                        break
                    start_sample = int(t * sr)
                    bass_duration = beat_interval
                    bass_note = generate_bass_note(root_freq, bass_duration, sr)
                    end_sample = min(start_sample + len(bass_note), beat_length)
                    beat[start_sample:end_sample] += bass_note[:end_sample - start_sample]
            beat = self.clean_reverb(beat, sr, room_size=0.3, wet_level=0.1)
            
            # Add guitar chords to the backing track
            print("🎸 Adding guitar chords...")
            guitar_beat = np.zeros(beat_length)
            
            # Define chord progressions based on key
            if is_major:
                # Major key chord progression (I-IV-V)
                chord_progression = [
                    [0, 4, 7],    # I chord (root, major third, perfect fifth)
                    [5, 9, 0],    # IV chord (subdominant)
                    [7, 11, 2],   # V chord (dominant)
                    [0, 4, 7]     # Back to I
                ]
            else:
                # Minor key chord progression (i-iv-v)
                chord_progression = [
                    [0, 3, 7],    # i chord (root, minor third, perfect fifth)
                    [5, 8, 0],    # iv chord (subdominant)
                    [7, 10, 2],   # v chord (dominant)
                    [0, 3, 7]     # Back to i
                ]
            
            # Convert chord intervals to frequencies
            base_freq = 65.41 * (2 ** (key_idx / 12))  # C2 for key_idx=0
            chord_freqs = []
            for chord in chord_progression:
                chord_notes = [base_freq * (2 ** (interval / 12)) for interval in chord]
                chord_freqs.append(chord_notes)
            
            # Add guitar chords every 2 bars
            for bar in range(int(beat_duration / (4 * beat_interval)) + 1):
                bar_start = bar * 4 * beat_interval
                chord_idx = bar % len(chord_progression)
                
                # Play chord on beat 1 of each bar
                t = bar_start
                if t < beat_duration:
                    start_sample = int(t * sr)
                    chord_duration = beat_interval * 2  # Hold chord for 2 beats
                    guitar_chord = generate_guitar_chord(chord_freqs[chord_idx], chord_duration, sr)
                    end_sample = min(start_sample + len(guitar_chord), beat_length)
                    guitar_beat[start_sample:end_sample] += guitar_chord[:end_sample - start_sample] * 0.6  # Higher volume for guitar chords
            
            # Mix guitar with existing beat
            beat = beat + guitar_beat * 1.5  # Much higher guitar volume
            
            # Add electric guitar parts
            print("🎸 Adding electric guitar...")
            electric_beat = np.zeros(beat_length)
            
            # Lead guitar fills on off-beats
            for bar in range(int(beat_duration / (4 * beat_interval)) + 1):
                bar_start = bar * 4 * beat_interval
                
                # Lead guitar fill on beat 3 of each bar
                t = bar_start + 2 * beat_interval
                if t < beat_duration:
                    start_sample = int(t * sr)
                    lead_duration = beat_interval * 0.5
                    # Use scale notes for lead guitar
                    scale_note = scale_freqs[bar % len(scale_freqs)]
                    lead_guitar = generate_electric_guitar(scale_note, lead_duration, sr, style='lead')
                    end_sample = min(start_sample + len(lead_guitar), beat_length)
                    electric_beat[start_sample:end_sample] += lead_guitar[:end_sample - start_sample] * 0.4
            
            # Rhythm electric guitar on every beat
            for bar in range(int(beat_duration / (4 * beat_interval)) + 1):
                bar_start = bar * 4 * beat_interval
                for beat_pos in range(4):
                    t = bar_start + beat_pos * beat_interval
                    if t < beat_duration:
                        start_sample = int(t * sr)
                        rhythm_duration = beat_interval * 0.25
                        # Use root note for rhythm
                        rhythm_guitar = generate_electric_guitar(root_freq, rhythm_duration, sr, style='rhythm')
                        end_sample = min(start_sample + len(rhythm_guitar), beat_length)
                        electric_beat[start_sample:end_sample] += rhythm_guitar[:end_sample - start_sample] * 0.3
            
            # Mix electric guitar with existing beat
            beat = beat + electric_beat * 1.5  # Electric guitar at 80% volume
            
            # Add flute parts
            print("🎶 Adding flute...")
            flute_beat = np.zeros(beat_length)
            
            # Flute melody using scale notes
            for bar in range(int(beat_duration / (4 * beat_interval)) + 1):
                bar_start = bar * 4 * beat_interval
                
                # Flute plays on beats 2 and 4 of each bar
                for beat_pos in [1, 3]:  # Beats 2 and 4
                    t = bar_start + beat_pos * beat_interval
                    if t < beat_duration:
                        start_sample = int(t * sr)
                        flute_duration = beat_interval * 1.5  # Hold note longer
                        
                        # Use different scale notes for melody
                        scale_idx = (bar + beat_pos) % len(scale_freqs)
                        flute_note = scale_freqs[scale_idx] * 2  # Play an octave higher
                        
                        flute_sound = generate_flute(flute_note, flute_duration, sr)
                        end_sample = min(start_sample + len(flute_sound), beat_length)
                        flute_beat[start_sample:end_sample] += flute_sound[:end_sample - start_sample] * 0.3
            
            # Flute counter-melody on off-beats
            for bar in range(int(beat_duration / (4 * beat_interval)) + 1):
                bar_start = bar * 4 * beat_interval
                
                # Flute counter-melody on eighth notes
                for eighth in range(8):
                    if eighth % 2 == 1:  # Only on off-beats
                        t = bar_start + (eighth / 2.0) * beat_interval
                        if t < beat_duration:
                            start_sample = int(t * sr)
                            counter_duration = beat_interval * 0.25
                            
                            # Use different scale notes for counter-melody
                            scale_idx = (bar * 2 + eighth) % len(scale_freqs)
                            counter_note = scale_freqs[scale_idx] * 1.5  # Different octave
                            
                            counter_flute = generate_flute(counter_note, counter_duration, sr)
                            end_sample = min(start_sample + len(counter_flute), beat_length)
                            flute_beat[start_sample:end_sample] += counter_flute[:end_sample - start_sample] * 0.2
            
            # Mix flute with existing beat
            beat = beat + flute_beat * 0.55  # Flute at 60% volume
            
            max_beat = np.max(np.abs(beat))
            if max_beat > 0:
                beat = beat / max_beat * beat_volume if max_beat > 0 else beat  # Increased to beat_volume (default 0.5)
            vocal_boost = 1.25  # Balanced vocal boost - audible but not overpowering
            beat_scale = 0.6   # Slightly increased beat volume for better balance

            corrected_y *= vocal_boost
            beat *= beat_scale

            final_mix = corrected_y + beat

            # Very gentle final normalization
            max_final = np.max(np.abs(final_mix))
            if max_final > 0:
                final_mix = final_mix / max_final * 0.98  # Minimal normalization
            print("✅ Enhanced processing complete!")
            return final_mix
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return y

def process_audio_clean_version(input_path, output_path, output_format, strength=1.0, retune_speed=0.9, reverb_amount=0.4, beat_volume=0.5):
    try:
        print(f"🎵 Loading: {input_path}")
        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            print("❌ Invalid input file")
            return False
        y, sr = librosa.load(input_path, sr=22050, mono=True)
        print(f"✅ Loaded: {sr}Hz, {len(y)/sr:.1f}s")
        processor = CleanTravisScottAutotune(sr=sr)
        print(f"🎤 Processing with enhanced algorithm...")
        print(f"   Strength: {strength}")
        print(f"   Retune Speed: {retune_speed}")
        print(f"   Reverb: {reverb_amount}")
        print(f"   Beat Volume: {beat_volume}")
        processed_y = processor.process_audio_clean(y, sr, strength, retune_speed, reverb_amount, beat_volume)
        if processed_y is None or len(processed_y) == 0:
            print("❌ Processing returned empty result")
            return False
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_y = np.asarray(processed_y, dtype=np.float32)
        if output_format.lower() == 'mp3':
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, processed_y, sr, subtype='PCM_16')
            audio = AudioSegment.from_wav(temp_wav)
            audio.export(output_path, format="mp3", bitrate="192k")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
        elif output_format.lower() == 'wav':
            sf.write(output_path, processed_y, sr, subtype='PCM_16')
        elif output_format.lower() == 'flac':
            sf.write(output_path, processed_y, sr, format='FLAC')
        else:
            sf.write(output_path, processed_y, sr, subtype='PCM_16')
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("🎉 Success! Enhanced autotune applied")
            return True
        else:
            print("❌ Output file not created")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    strength = float(request.form.get('strength', 1.0))
    strength = max(0.1, min(1.0, strength))
    retune_speed = float(request.form.get('retune_speed', 0.9))
    retune_speed = max(0.1, min(1.0, retune_speed))
    reverb_amount = float(request.form.get('reverb_amount', 0.4))
    reverb_amount = max(0.0, min(0.6, reverb_amount))
    beat_volume = float(request.form.get('beat_volume', 0.5))
    beat_volume = max(0.0, min(1.0, beat_volume))
    output_format = request.form.get('output_format', 'wav').lower()
    if output_format not in ['wav', 'mp3', 'flac']:
        output_format = 'wav'
    if file and allowed_file(file.filename):
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        original_extension = filename.rsplit('.', 1)[1].lower()
        input_filename = f"{file_id}_input.{original_extension}"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        file.save(input_path)
        output_filename = f"{file_id}_clean_autotune.{output_format}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        success = process_audio_clean_version(input_path, output_path, output_format, strength, retune_speed, reverb_amount, beat_volume)
        try:
            os.remove(input_path)
        except:
            pass
        if success:
            return jsonify({
                'message': 'Enhanced autotune processing successful',
                'file_id': file_id,
                'original_name': filename,
                'processed_name': f"{filename.rsplit('.', 1)[0]}_clean_autotune.{output_format}",
                'parameters': {'strength': strength, 'retune_speed': retune_speed, 'reverb_amount': reverb_amount, 'beat_volume': beat_volume},
                'output_format': output_format,
                'style': 'Travis Scott-Style Autotune with Beat'
            }), 200
        else:
            return jsonify({'error': 'Processing failed'}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<file_id>')
def download_file(file_id):
    try:
        for ext in ['wav', 'mp3', 'flac']:
            output_filename = f"{file_id}_clean_autotune.{ext}"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return send_file(output_path, as_attachment=True, download_name=f"clean_autotune.{ext}", mimetype=f'audio/{ext}' if ext != 'wav' else 'audio/wav')
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'Enhanced AutoTune Server Running',
        'version': '4.0 - Travis Scott Style with Beat',
        'features': [
            '🎼 Major/minor key detection',
            '🎯 Fast pitch correction',
            '🌊 Vibrato-preserving smoothing',
            '🌌 Ambient reverb',
            '⏳ Subtle delay',
            '🔥 Harmonic saturation',
            '🥁 Backing beat with drums and bass',
            '✨ Professional mix-ready output'
        ],
        'recommended_settings': {'strength': 1.0, 'retune_speed': 0.9, 'reverb_amount': 0.4, 'beat_volume': 0.5}
    }), 200

if __name__ == '__main__':
    print("🎵 Enhanced AutoTune Server Starting...")
    print("✨ Style: Travis Scott - Stylized & Emotional with Beat")
    print("🚀 Server on port 5001...")
    app.run(debug=True, port=5001)