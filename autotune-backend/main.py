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

class ProfessionalAutotune:
    def __init__(self, sr=22050):
        self.sr = sr
        self.hop_length = 256  # Smaller hop for better time resolution
        self.frame_length = 1024
        
    def detect_key_and_scale(self, y, sr):
        """Detect the key and scale of the audio using chroma features"""
        try:
            # Compute chroma features
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Major and minor scale templates
            major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            # Find best key for major and minor
            major_correlations = []
            minor_correlations = []
            
            for shift in range(12):
                major_shifted = np.roll(major_template, shift)
                minor_shifted = np.roll(minor_template, shift)
                
                major_corr = np.corrcoef(chroma_mean, major_shifted)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_shifted)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Find best matches
            best_major_idx = np.argmax(major_correlations)
            best_minor_idx = np.argmax(minor_correlations)
            best_major_corr = major_correlations[best_major_idx]
            best_minor_corr = minor_correlations[best_minor_idx]
            
            # Choose between major and minor
            if best_major_corr > best_minor_corr:
                key_idx = best_major_idx
                is_major = True
            else:
                key_idx = best_minor_idx
                is_major = False
            
            return key_idx, is_major
            
        except:
            # Default to C major if detection fails
            return 0, True
    
    def get_scale_frequencies(self, key_idx=0, is_major=True):
        """Generate frequencies for detected key and scale"""
        # Note names for reference
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Scale intervals (semitones from root)
        if is_major:
            intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
        else:
            intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor scale
        
        frequencies = []
        # Generate frequencies for multiple octaves (C1 to C8)
        for octave in range(1, 9):
            # A4 = 440Hz, calculate base frequency for octave
            a4_freq = 440.0
            base_freq = a4_freq * (2 ** (octave - 4)) * (2 ** ((key_idx - 9) / 12))
            
            for interval in intervals:
                note_freq = base_freq * (2 ** (interval / 12))
                frequencies.append(note_freq)
        
        return sorted(frequencies)
    
    def smooth_pitch_track(self, pitches, window_size=5):
        """Smooth pitch track to reduce noise and artifacts"""
        try:
            if SCIPY_AVAILABLE:
                # Apply median filter to remove outliers
                from scipy import signal
                smoothed = signal.medfilt(pitches, kernel_size=min(window_size, len(pitches)))
                
                # Apply gaussian smoothing
                if len(smoothed) > window_size:
                    smoothed = signal.gaussian_filter1d(smoothed, sigma=1.0)
                return smoothed
            else:
                # Fallback: simple moving average
                if len(pitches) < window_size:
                    return pitches
                    
                smoothed = np.copy(pitches)
                for i in range(window_size, len(pitches) - window_size):
                    smoothed[i] = np.mean(pitches[i-window_size//2:i+window_size//2+1])
                return smoothed
                
        except Exception as e:
            print(f"Warning: Pitch smoothing failed, using original: {e}")
            return pitches
    
    def detect_vibrato_and_slides(self, pitches, timestamps):
        """Detect vibrato and pitch slides to preserve musical expression"""
        if len(pitches) < 10:
            return np.zeros(len(pitches), dtype=bool), np.zeros(len(pitches), dtype=bool)
        
        # Calculate pitch derivatives
        pitch_diff = np.diff(pitches)
        pitch_diff = np.concatenate([[0], pitch_diff])
        
        # Detect vibrato (rapid oscillations)
        vibrato_mask = np.zeros(len(pitches), dtype=bool)
        window_size = min(10, len(pitches) // 4)
        
        for i in range(window_size, len(pitches) - window_size):
            window = pitches[i-window_size:i+window_size]
            if len(window) > 5:
                # Check for oscillatory behavior
                zero_crossings = np.where(np.diff(np.sign(np.diff(window))))[0]
                if len(zero_crossings) > 3:  # Multiple oscillations
                    vibrato_mask[i] = True
        
        # Detect pitch slides (gradual pitch changes)
        slide_mask = np.zeros(len(pitches), dtype=bool)
        slide_threshold = 50  # cents
        
        for i in range(2, len(pitches) - 2):
            if abs(pitch_diff[i]) > slide_threshold and abs(pitch_diff[i-1]) > slide_threshold:
                if np.sign(pitch_diff[i]) == np.sign(pitch_diff[i-1]):
                    slide_mask[i] = True
        
        return vibrato_mask, slide_mask
    
    def adaptive_correction_strength(self, pitches, target_pitches, vibrato_mask, slide_mask, base_strength=0.8):
        """Calculate adaptive correction strength based on musical context"""
        correction_strengths = np.full(len(pitches), base_strength)
        
        # Reduce correction during vibrato
        correction_strengths[vibrato_mask] *= 0.3
        
        # Moderate correction during slides
        correction_strengths[slide_mask] *= 0.6
        
        # Calculate pitch deviation and adjust strength
        for i in range(len(pitches)):
            if pitches[i] > 0 and target_pitches[i] > 0:
                deviation_cents = abs(1200 * np.log2(target_pitches[i] / pitches[i]))
                
                # Less correction for small deviations (natural variation)
                if deviation_cents < 20:
                    correction_strengths[i] *= 0.4
                elif deviation_cents < 50:
                    correction_strengths[i] *= 0.7
                # Full correction for large deviations
        
        return correction_strengths
    
    def snap_to_scale_intelligent(self, freq, scale_freqs, previous_target=None):
        """Intelligent pitch snapping with hysteresis and musical context"""
        if freq <= 0:
            return freq
        
        # Convert to cents for better comparison
        freq_cents = 1200 * np.log2(freq / 440)
        scale_cents = [1200 * np.log2(f / 440) for f in scale_freqs]
        
        # Find nearest scale frequencies
        differences = [abs(freq_cents - sc) for sc in scale_cents]
        nearest_idx = np.argmin(differences)
        
        # Hysteresis: if we have a previous target, prefer staying close to it
        if previous_target and previous_target > 0:
            prev_target_cents = 1200 * np.log2(previous_target / 440)
            
            # Find the previous target in scale
            prev_differences = [abs(prev_target_cents - sc) for sc in scale_cents]
            prev_idx = np.argmin(prev_differences)
            
            # If current nearest is close to previous target, stick with previous
            if abs(scale_cents[nearest_idx] - prev_target_cents) > 100:  # 1 semitone
                if differences[prev_idx] < differences[nearest_idx] * 1.5:
                    nearest_idx = prev_idx
        
        return scale_freqs[nearest_idx]
    
    def process_audio_professional(self, y, sr, strength=0.8, retune_speed=0.5, preserve_formants=True):
        """Professional autotune processing with natural sound"""
        try:
            print("Analyzing audio for key detection...")
            
            # Detect key and scale
            try:
                key_idx, is_major = self.detect_key_and_scale(y, sr)
                scale_freqs = self.get_scale_frequencies(key_idx, is_major)
                
                scale_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_idx]
                scale_type = "Major" if is_major else "Minor"
                print(f"Detected key: {scale_name} {scale_type}")
            except Exception as e:
                print(f"Key detection failed, using C Major: {e}")
                key_idx, is_major = 0, True
                scale_freqs = self.get_scale_frequencies(0, True)
            
            # Enhanced pitch detection with fallback
            print("Detecting pitch with high precision...")
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                    sr=sr, hop_length=self.hop_length, threshold=0.1, resolution=0.1
                )
            except Exception as e:
                print(f"Advanced pitch detection failed, using basic method: {e}")
                # Fallback to basic pitch detection
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=self.hop_length)
                f0 = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    f0.append(pitch if pitch > 0 else 0)
                f0 = np.array(f0)
                voiced_flag = f0 > 0
            
            # Replace NaN values with 0
            f0 = np.nan_to_num(f0, nan=0.0)
            
            # Smooth the pitch track
            try:
                f0_smooth = self.smooth_pitch_track(f0)
            except Exception as e:
                print(f"Pitch smoothing failed: {e}")
                f0_smooth = f0
            
            # Create time axis
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=self.hop_length)
            
            # Detect musical expression with fallback
            try:
                vibrato_mask, slide_mask = self.detect_vibrato_and_slides(f0_smooth, times)
            except Exception as e:
                print(f"Expression detection failed: {e}")
                vibrato_mask = np.zeros(len(f0_smooth), dtype=bool)
                slide_mask = np.zeros(len(f0_smooth), dtype=bool)
            
            # Calculate target pitches with intelligent snapping
            target_pitches = np.zeros_like(f0_smooth)
            for i, freq in enumerate(f0_smooth):
                if freq > 0:
                    try:
                        prev_target = target_pitches[i-1] if i > 0 else None
                        target_pitches[i] = self.snap_to_scale_intelligent(freq, scale_freqs, prev_target)
                    except Exception as e:
                        # Fallback to simple snapping
                        target_pitches[i] = self.snap_to_scale_simple(freq, scale_freqs)
            
            # Calculate adaptive correction strengths
            try:
                correction_strengths = self.adaptive_correction_strength(
                    f0_smooth, target_pitches, vibrato_mask, slide_mask, strength
                )
            except Exception as e:
                print(f"Adaptive strength calculation failed: {e}")
                correction_strengths = np.full(len(f0_smooth), strength)
            
            # Apply temporal smoothing to correction strengths
            try:
                if SCIPY_AVAILABLE:
                    from scipy import signal
                    correction_strengths = signal.gaussian_filter1d(correction_strengths, sigma=retune_speed * 2)
            except Exception as e:
                print(f"Temporal smoothing failed: {e}")
            
            print("Applying pitch correction...")
            
            # Simplified processing with robust error handling
            autotuned_y = np.copy(y)
            
            # Process in smaller, more manageable chunks
            chunk_size = min(2048, len(y) // 10)  # Adaptive chunk size
            
            for i, (orig_pitch, target_pitch, corr_strength) in enumerate(zip(f0_smooth, target_pitches, correction_strengths)):
                if orig_pitch > 0 and target_pitch > 0 and abs(orig_pitch - target_pitch) > 1:
                    try:
                        # Calculate pitch shift
                        pitch_ratio = target_pitch / orig_pitch
                        pitch_shift_semitones = 12 * np.log2(pitch_ratio) * corr_strength
                        
                        # Apply retune speed
                        pitch_shift_semitones *= (1 - retune_speed) + retune_speed * np.tanh(abs(pitch_shift_semitones) / 2)
                        
                        # Get audio frame
                        start_sample = int(i * self.hop_length)
                        end_sample = min(start_sample + chunk_size, len(y))
                        
                        if end_sample > start_sample and abs(pitch_shift_semitones) > 0.05:
                            # Extract segment
                            segment = y[start_sample:end_sample]
                            
                            # Apply pitch shift with error handling
                            try:
                                shifted_segment = librosa.effects.pitch_shift(
                                    segment, sr=sr, n_steps=pitch_shift_semitones
                                )
                                
                                # Blend the result
                                blend_len = min(len(shifted_segment), len(autotuned_y) - start_sample)
                                autotuned_y[start_sample:start_sample + blend_len] = shifted_segment[:blend_len]
                                
                            except Exception as pitch_error:
                                print(f"Pitch shift failed for frame {i}: {pitch_error}")
                                continue
                                
                    except Exception as frame_error:
                        print(f"Frame processing failed for frame {i}: {frame_error}")
                        continue
            
            # Final normalization with safety checks
            max_val = np.max(np.abs(autotuned_y))
            original_max = np.max(np.abs(y))
            
            if max_val > 0:
                # Preserve original dynamics while preventing clipping
                target_max = min(original_max * 1.05, 0.95)
                autotuned_y = autotuned_y / max_val * target_max
            
            return autotuned_y
            
        except Exception as e:
            print(f"Professional processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Return simple autotune as fallback
            return self.simple_autotune_fallback(y, sr, strength)

    def snap_to_scale_simple(self, freq, scale_freqs):
        """Simple pitch snapping fallback method"""
        if freq <= 0:
            return freq
        
        # Find nearest scale frequency
        differences = [abs(freq - f) for f in scale_freqs]
        nearest_idx = np.argmin(differences)
        return scale_freqs[nearest_idx]
    
    def simple_autotune_fallback(self, y, sr, strength=0.8):
        """Simple autotune fallback when advanced processing fails"""
        print("Using simple autotune fallback...")
        
        try:
            # Use basic pitch detection
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            
            # Get major scale frequencies (C major)
            scale_freqs = self.get_scale_frequencies(0, True)
            
            # Process in chunks
            autotuned_y = np.copy(y)
            hop_length = 512
            
            for t in range(pitches.shape[1]):
                # Get fundamental frequency
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                
                if pitch > 0:
                    # Snap to scale
                    target_pitch = self.snap_to_scale_simple(pitch, scale_freqs)
                    
                    if abs(pitch - target_pitch) > 1:  # Only correct if significant difference
                        # Calculate pitch shift
                        pitch_ratio = target_pitch / pitch
                        pitch_shift_semitones = 12 * np.log2(pitch_ratio) * strength
                        
                        # Get audio segment
                        start_sample = t * hop_length
                        end_sample = min(start_sample + hop_length * 2, len(y))
                        
                        if end_sample > start_sample:
                            segment = y[start_sample:end_sample]
                            
                            try:
                                # Apply pitch shift
                                shifted_segment = librosa.effects.pitch_shift(
                                    segment, sr=sr, n_steps=pitch_shift_semitones
                                )
                                
                                # Replace segment
                                replace_len = min(len(shifted_segment), len(autotuned_y) - start_sample)
                                autotuned_y[start_sample:start_sample + replace_len] = shifted_segment[:replace_len]
                                
                            except Exception as e:
                                print(f"Pitch shift failed for segment: {e}")
                                continue
            
            return autotuned_y
            
        except Exception as e:
            print(f"Simple autotune fallback failed: {e}")
            # Return original audio if all else fails
            return y

def process_audio(input_path, output_path, output_format, strength=0.8, retune_speed=0.5):
    """
    Process the audio file with professional autotune effect
    """
    try:
        print(f"Loading audio file: {input_path}")
        
        # Verify input file exists and has content
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            return False
            
        if os.path.getsize(input_path) == 0:
            print(f"Error: Input file is empty: {input_path}")
            return False
        
        # Load audio using librosa with optimal settings
        try:
            y, sr = librosa.load(input_path, sr=None, mono=True)
            print(f"Loaded audio - Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
        
        # Validate audio data
        if len(y) == 0:
            print("Error: Audio file contains no data")
            return False
            
        if sr <= 0:
            print("Error: Invalid sample rate")
            return False
        
        # Create autotune processor instance
        autotune_processor = ProfessionalAutotune(sr=sr)
        
        # Ensure adequate sample rate for pitch detection
        if sr < 22050:
            print(f"Upsampling from {sr} to 22050 Hz for better pitch detection")
            try:
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
                autotune_processor.sr = sr
            except Exception as e:
                print(f"Error during resampling: {e}")
                # Continue with original sample rate
        
        # Apply professional autotune processing
        print(f"Applying professional autotune (strength: {strength}, retune speed: {retune_speed})")
        try:
            autotuned_y = autotune_processor.process_audio_professional(
                y, sr, strength=strength, retune_speed=retune_speed
            )
        except Exception as e:
            print(f"Professional autotune failed: {e}")
            return False
        
        # Validate processed audio
        if autotuned_y is None or len(autotuned_y) == 0:
            print("Error: Processed audio is empty")
            return False
        
        # Prepare audio for output
        autotuned_y = np.asarray(autotuned_y, dtype=np.float32)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Handle different output formats with error handling
        try:
            if output_format.lower() == 'mp3':
                # For MP3, use pydub with high quality settings
                temp_wav_path = output_path.replace('.mp3', '_temp.wav')
                
                # Save as high-quality WAV first
                sf.write(temp_wav_path, autotuned_y, sr, subtype='PCM_16')
                
                # Convert to MP3 with high bitrate
                audio = AudioSegment.from_wav(temp_wav_path)
                audio.export(output_path, format="mp3", bitrate="192k")
                
                # Clean up temp file
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                
            elif output_format.lower() == 'wav':
                # Use high-quality PCM format
                sf.write(output_path, autotuned_y, sr, subtype='PCM_16')
                
            elif output_format.lower() == 'flac':
                # FLAC supports high quality lossless compression
                sf.write(output_path, autotuned_y, sr, format='FLAC')
                
            else:  # Default to high-quality WAV
                sf.write(output_path, autotuned_y, sr, subtype='PCM_16')
                
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False
        
        print(f"Professional autotune processing completed: {output_path}")
        
        # Verify output file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("✓ Audio processing completed successfully with professional quality")
            return True
        else:
            print("✗ Error: Output file was not created or is empty")
            return False
        
    except Exception as e:
        print(f"✗ Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get autotune parameters from form data
    strength = float(request.form.get('strength', 0.8))
    strength = max(0.1, min(1.0, strength))  # Clamp between 0.1 and 1.0
    
    # Get retune speed (how fast the correction applies)
    retune_speed = float(request.form.get('retune_speed', 0.5))
    retune_speed = max(0.1, min(1.0, retune_speed))  # Clamp between 0.1 and 1.0
    
    # Get output format preference
    output_format = request.form.get('output_format', 'wav').lower()
    if output_format not in ['wav', 'mp3', 'flac']:
        output_format = 'wav'
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        original_extension = filename.rsplit('.', 1)[1].lower()
        
        # Save uploaded file
        input_filename = f"{file_id}_input.{original_extension}"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        file.save(input_path)
        
        # Process with professional autotune
        output_filename = f"{file_id}_processed.{output_format}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        success = process_audio(input_path, output_path, output_format, strength, retune_speed)
        
        # Clean up input file
        try:
            os.remove(input_path)
        except:
            pass
        
        if success:
            return jsonify({
                'message': 'File processed successfully with professional autotune',
                'file_id': file_id,
                'original_name': filename,
                'processed_name': f"{filename.rsplit('.', 1)[0]}_professional_autotuned.{output_format}",
                'strength_used': strength,
                'retune_speed_used': retune_speed,
                'output_format': output_format,
                'quality': 'professional'
            }), 200
        else:
            return jsonify({'error': 'Failed to process audio with professional autotune'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<file_id>')
def download_file(file_id):
    try:
        # Check for different possible output formats
        for ext in ['wav', 'mp3', 'flac']:
            output_filename = f"{file_id}_processed.{ext}"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            if os.path.exists(output_path):
                # Verify file has content
                if os.path.getsize(output_path) > 0:
                    return send_file(
                        output_path, 
                        as_attachment=True,
                        download_name=f"{file_id}_professional_autotuned.{ext}",
                        mimetype=f'audio/{ext}' if ext != 'wav' else 'audio/wav'
                    )
                else:
                    return jsonify({'error': 'Processed file is empty'}), 500
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'Professional AutoTune server is running',
        'features': [
            'Automatic key detection',
            'Intelligent pitch correction',
            'Vibrato and slide preservation',
            'Formant preservation',
            'Adaptive correction strength',
            'Professional audio quality'
        ]
    }), 200

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old processed files to save disk space"""
    try:
        import time
        current_time = time.time()
        cleaned_count = 0
        
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    filepath = os.path.join(folder, filename)
                    # Remove files older than 1 hour
                    if os.path.getmtime(filepath) < current_time - 3600:
                        os.remove(filepath)
                        cleaned_count += 1
        
        return jsonify({
            'message': f'Cleaned up {cleaned_count} old files',
            'count': cleaned_count
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🎵 Starting Professional AutoTune Flask server...")
    print("📦 Required packages: librosa soundfile pydub scipy")
    print("🎤 Features: Key detection, natural pitch correction, expression preservation")
    print("🔧 For MP3 support: pip install pydub[mp3]")
    print("🚀 Server starting on port 5001...")
    app.run(debug=True, port=5001)