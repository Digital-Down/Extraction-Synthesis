import math
import os
import json
import shutil
from Audio.AudioSampleValues import AudioSampleValues
import numpy as np
from scipy import signal
import soundfile as sf
from pydub import AudioSegment
import sys
from Audio.AudioStretch import *
import glob

class MusicToInstrument:
    sample_rate = 44100
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    min_samples = sample_rate  # 1 second
    max_samples = sample_rate * 10  # 10 seconds

    @classmethod
    def convert_to_wav_16bit_stereo_44100(cls, input_file, output_file):
        # Load the audio file
        audio = AudioSegment.from_file(input_file)

        # Convert to stereo if mono
        if audio.channels == 1:
            audio = audio.set_channels(2)

        # Set sample rate to 44100 Hz
        audio = audio.set_frame_rate(44100)

        # Set sample width to 2 bytes (16-bit)
        audio = audio.set_sample_width(2)

        # Export as WAV
        audio.export(output_file, format="wav")

    @classmethod
    def normalize_audio(cls, input_file, output_file):
        audio_data, sample_rate = sf.read(input_file)
        
        # Ensure audio is in float64 format for processing
        audio_data = audio_data.astype(np.float64)
        
        # Normalize audio to use full range
        max_value = np.max(np.abs(audio_data))
        if max_value > 0:
            normalized_audio = audio_data / max_value
        else:
            normalized_audio = audio_data
        
        # Save normalized audio
        sf.write(output_file, normalized_audio, sample_rate, subtype='PCM_16')

    @classmethod
    def get_next_part_number(cls, note_folder, note):
        existing_parts = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(note_folder) if f.startswith(f"{note}_") and f.endswith('.wav')]
        return max(existing_parts) + 1 if existing_parts else 1

    @classmethod
    def frequency_to_note(cls, frequency):
        A4 = 440
        C0 = A4 * pow(2, -4.75)
        
        if frequency == 0:
            return "Unknown"
        
        h = round(12 * math.log2(frequency / C0))
        octave = h // 12
        n = h % 12
        return f"{cls.note_names[n]}{octave}"

    @classmethod
    def note_to_frequency(cls, note):
        A4 = 440
        note_name = ''.join([c for c in note if not c.isdigit()])
        octave = int(''.join([c for c in note if c.isdigit()]))
        semitone = cls.note_names.index(note_name)
        return A4 * pow(2, (octave - 4) + (semitone - 9) / 12)

    @classmethod
    def average_absolute_value(cls, samples):
        return sum(abs(sample) for sample in samples) / len(samples)

    @classmethod
    def convert(cls, audio_data, oscillations, threshold=100, max_threshold=32767, channel=0):
        current_oscillation = []
        positive_count = 0
        negative_count = 0
        state = 'positive'
        total_oscillations = 0
        zero_crossings = 0

        print(f"Total audio samples: {len(audio_data)}")

        for i, sample in enumerate(audio_data):
            value = sample[channel]

            if state == 'positive':
                if value > 0:
                    positive_count += 1
                    current_oscillation.append(value)
                elif value < 0:
                    state = 'negative'
                    negative_count = 1
                    current_oscillation.append(value)
                    zero_crossings += 1
                else:  # value == 0
                    current_oscillation.append(value)
            else:  # state == 'negative'
                if value < 0:
                    negative_count += 1
                    current_oscillation.append(value)
                elif value > 0:
                    # Check if we've completed a full oscillation
                    if positive_count > 0 and negative_count > 0:
                        total_oscillations += 1
                        total_count = positive_count + negative_count
                        
                        # Calculate average absolute value
                        avg_abs_value = cls.average_absolute_value(current_oscillation)
                        
                        if threshold <= avg_abs_value <= max_threshold:
                            found_match = False
                            for note in cls.get_all_notes():
                                expected_count = round(cls.sample_rate / cls.note_to_frequency(note))
                                if abs(total_count - expected_count) <= 1:  # Allow for rounding error
                                    oscillations[note].extend(current_oscillation)
                                    print(f"Accepted oscillation for {note}, count: {total_count}")
                                    found_match = True
                                    break
                            if not found_match:
                                print(f"Rejected oscillation, total count: {total_count}")
                        elif avg_abs_value > max_threshold:
                            print(f"Rejected oscillation due to high average absolute value")
                        else:
                            print(f"Rejected oscillation due to low average absolute value")
                    
                    # Reset for the next oscillation
                    state = 'positive'
                    positive_count = 1
                    negative_count = 0
                    current_oscillation = [value]
                    zero_crossings += 1
                else:  # value == 0
                    current_oscillation.append(value)

            if i % 10000 == 0:
                print(f"Processed {i} samples")

        print(f"Total oscillations detected: {total_oscillations}")
        print(f"Total zero crossings: {zero_crossings}")

    @classmethod
    def get_all_notes(cls):
        notes = []
        for octave in range(1, 7):  # 1 to 6
            for note in cls.note_names:
                notes.append(f"{note}{octave}")
        return notes
    @classmethod
    def preprocess_files(cls, input_folder):
        # Delete existing mfast and mslow files
        for file in glob.glob(os.path.join(input_folder, "mfast*.wav")) + glob.glob(os.path.join(input_folder, "mslow*.wav")):
            os.remove(file)

        # Process each WAV file in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.wav'):
                input_path = os.path.join(input_folder, filename)
                base_name = os.path.splitext(filename)[0]

                # Create mslowTest.wav
                slow_output = os.path.join(input_folder, f"mslow{base_name}.wav")
                AudioSampleValues.list_to_wav(
                    AudioStretch.slow([AudioSampleValues.wav_to_list(input_path)], 1),
                    slow_output
                )

                # Create mfastTest.wav
                fast_output = os.path.join(input_folder, f"mfast{base_name}.wav")
                AudioSampleValues.list_to_wav(
                    AudioStretch.fast([AudioSampleValues.wav_to_list(input_path)], 1),
                    fast_output
                )

                print(f"Generated mfast{filename}")
                print(f"Generated mslow{filename}")

        print("Preprocessing completed")
    @classmethod
    def delete_mslow_mfast_files(cls, input_folder):
        for file in glob.glob(os.path.join(input_folder, "mfast*.wav")) + glob.glob(os.path.join(input_folder, "mslow*.wav")):
            os.remove(file)
        print("Checking for Methylphenethylamine files.")
    @classmethod
    def process_audio_folder_normalize(cls, input_folder, output_folder, incomplete_folder=None, threshold=100, max_threshold=32767, methylphenethylamine=False):
        output_messages = []
        if not os.path.isdir(input_folder):
            print(f"Error: {input_folder} is not a valid directory.")
            output_messages.append(f"Error: {input_folder} is not a valid directory.\n")
            return output_messages

        cls.delete_mslow_mfast_files(input_folder)

        temp_folder = os.path.join(input_folder, 'temp')

        # Check if temp folder exists and delete its contents if it does
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

        total_files_processed = 0

        def clear_output():
            # Clear the console output
            if sys.platform.startswith('win'):
                os.system('cls')  # For Windows
            else:
                os.system('clear')  # For Unix/Linux/macOS

        def save_oscillations(note, samples):
            if len(samples) > 0:
                # Duplicate the samples until we reach at least 10 seconds
                while len(samples) < cls.sample_rate * 10:
                    samples.extend(samples[:min(len(samples), cls.sample_rate * 10 - len(samples))])

                # Save as complete sample
                note_folder = os.path.join(output_folder, note)
                os.makedirs(note_folder, exist_ok=True)

                # Split samples into 10-second chunks
                chunks = [samples[i:i + cls.sample_rate * 10] for i in range(0, len(samples), cls.sample_rate * 10)]

                for i, chunk in enumerate(chunks):
                    part_number = cls.get_next_part_number(note_folder, note)
                    output_file = os.path.join(note_folder, f"{note}_{part_number}.wav")
                    AudioSampleValues.list_to_mono_wav(chunk, output_file)
                    print(f"Successfully created audio sample for {note} (part {part_number})")
                    output_messages.append(f"Successfully created audio sample for {note} (part {part_number})\n")

        def process_audio_file(file_path, file_type):
            audio_data = AudioSampleValues.wav_to_list(file_path)

            oscillations = {note: [] for note in cls.get_all_notes()}

            # Process left channel
            print(f"Processing {file_type} - left channel")
            output_messages.append(f"Processing {file_type} - left channel\n")
            cls.convert(audio_data, oscillations, threshold, max_threshold, channel=0)

            # Save oscillations after processing left channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)

            oscillations = {note: [] for note in cls.get_all_notes()}

            # Process right channel
            print(f"Processing {file_type} - right channel")
            output_messages.append(f"Processing {file_type} - right channel\n")
            cls.convert(audio_data, oscillations, threshold, max_threshold, channel=1)

            # Save oscillations after processing right channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)

        def process_file(item_path):
            nonlocal total_files_processed
            print(f"Processing file: {item_path}")
            output_messages.append(f"Processing file: {item_path}\n")

            # Prepare the path for the normalized file
            base_name = os.path.splitext(os.path.basename(item_path))[0]
            normalized_path = os.path.join(temp_folder, f"normalized_{base_name}.wav")

            # Check audio file properties
            info = sf.info(item_path)
            if info.subtype != 'PCM_16' or info.samplerate != 44100 or info.channels != 2:
                print(f"Adjusting to 16-bit stereo 44100 Hz WAV and normalizing")
                output_messages.append(f"Adjusting to 16-bit stereo 44100 Hz WAV and normalizing\n")
                cls.convert_to_wav_16bit_stereo_44100(item_path, normalized_path)
                cls.normalize_audio(normalized_path, normalized_path)  # Normalize in-place
            else:
                print(f"Normalizing")
                output_messages.append(f"Normalizing\n")
                cls.normalize_audio(item_path, normalized_path)

            # Process the normalized file
            print(f"Processing normalized file")
            output_messages.append(f"Processing normalized file\n")
            process_audio_file(normalized_path, "normalized")

            # Now that the file is normalized and in the correct format, we can apply methylphenethylamine if needed
            if methylphenethylamine:
                print(f"Applying methylphenethylamine processing")
                output_messages.append(f"Applying methylphenethylamine processing\n")
                slow_output = os.path.join(temp_folder, f"mslow_{base_name}.wav")
                fast_output = os.path.join(temp_folder, f"mfast_{base_name}.wav")

                AudioSampleValues.list_to_wav(
                    AudioStretch.slow([AudioSampleValues.wav_to_list(normalized_path)], 1),
                    slow_output
                )
                AudioSampleValues.list_to_wav(
                    AudioStretch.fast([AudioSampleValues.wav_to_list(normalized_path)], 1),
                    fast_output
                )

                # Process both the slow and fast versions
                for speed in ['slow', 'fast']:
                    speed_path = os.path.join(temp_folder, f"m{speed}_{base_name}.wav")
                    print(f"Processing {speed} version")
                    output_messages.append(f"Processing {speed} version\n")
                    process_audio_file(speed_path, speed)

            clear_output()  # Clear the output after processing each file
            total_files_processed += 1

        def process_folder(current_folder):
            for item in os.listdir(current_folder):
                item_path = os.path.join(current_folder, item)
                if os.path.isdir(item_path) and item != 'temp':
                    # Recursively process subfolders, excluding the temp folder
                    process_folder(item_path)
                elif os.path.isfile(item_path) and item_path.lower().endswith('.wav'):
                    process_file(item_path)

        process_folder(input_folder)
        print(f"Finished processing {total_files_processed} audio files.")
        output_messages.append(f"Finished processing {total_files_processed} audio files.\n")

        # Delete temp folder
        shutil.rmtree(temp_folder)

        return output_messages

    @classmethod
    def process_audio_folder(cls, input_folder, output_folder, incomplete_folder=None, threshold=100, max_threshold=32767, methylphenethylamine=False):
        output_messages = []
        if not os.path.isdir(input_folder):
            print(f"Error: {input_folder} is not a valid directory.")
            output_messages.append(f"Error: {input_folder} is not a valid directory.\n")
            return output_messages

        cls.delete_mslow_mfast_files(input_folder)

        temp_folder = os.path.join(input_folder, 'temp')

        # Check if temp folder exists and delete its contents if it does
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

        total_files_processed = 0

        def clear_output():
            # Clear the console output
            if sys.platform.startswith('win'):
                os.system('cls')  # For Windows
            else:
                os.system('clear')  # For Unix/Linux/macOS

        def save_oscillations(note, samples):
            if len(samples) > 0:
                # Duplicate the samples until we reach at least 10 seconds
                while len(samples) < cls.sample_rate * 10:
                    samples.extend(samples[:min(len(samples), cls.sample_rate * 10 - len(samples))])

                # Save as complete sample
                note_folder = os.path.join(output_folder, note)
                os.makedirs(note_folder, exist_ok=True)

                # Split samples into 10-second chunks
                chunks = [samples[i:i + cls.sample_rate * 10] for i in range(0, len(samples), cls.sample_rate * 10)]

                for i, chunk in enumerate(chunks):
                    part_number = cls.get_next_part_number(note_folder, note)
                    output_file = os.path.join(note_folder, f"{note}_{part_number}.wav")
                    AudioSampleValues.list_to_mono_wav(chunk, output_file)
                    print(f"Successfully created audio sample for {note} (part {part_number})")
                    output_messages.append(f"Successfully created audio sample for {note} (part {part_number})\n")

        def process_audio_file(file_path, file_type):
            audio_data = AudioSampleValues.wav_to_list(file_path)

            oscillations = {note: [] for note in cls.get_all_notes()}

            # Process left channel
            print(f"Processing {file_type} - left channel")
            output_messages.append(f"Processing {file_type} - left channel\n")
            cls.convert(audio_data, oscillations, threshold, max_threshold, channel=0)

            # Save oscillations after processing left channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)

            oscillations = {note: [] for note in cls.get_all_notes()}

            # Process right channel
            print(f"Processing {file_type} - right channel")
            output_messages.append(f"Processing {file_type} - right channel\n")
            cls.convert(audio_data, oscillations, threshold, max_threshold, channel=1)

            # Save oscillations after processing right channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)

        def process_file(item_path):
            nonlocal total_files_processed
            print(f"Processing file: {item_path}")
            output_messages.append(f"Processing file: {item_path}\n")

            # Check audio file properties
            info = sf.info(item_path)
            base_name = os.path.splitext(os.path.basename(item_path))[0]
            processed_path = os.path.join(temp_folder, f"processed_{base_name}.wav")

            if info.subtype != 'PCM_16' or info.samplerate != 44100 or info.channels != 2:
                print(f"Converting to 16-bit stereo 44100 Hz WAV")
                output_messages.append(f"Converting to 16-bit stereo 44100 Hz WAV\n")
                cls.convert_to_wav_16bit_stereo_44100(item_path, processed_path)
            else:
                print(f"File is already in correct format")
                output_messages.append(f"File is already in correct format\n")
                shutil.copy(item_path, processed_path)

            # Process the original file
            print(f"Processing original file")
            output_messages.append(f"Processing original file\n")
            process_audio_file(processed_path, "original")

            # Now that the file is in the correct format, we can apply methylphenethylamine if needed
            if methylphenethylamine:
                print(f"Applying methylphenethylamine processing")
                output_messages.append(f"Applying methylphenethylamine processing\n")
                slow_output = os.path.join(temp_folder, f"mslow_{base_name}.wav")
                fast_output = os.path.join(temp_folder, f"mfast_{base_name}.wav")

                AudioSampleValues.list_to_wav(
                    AudioStretch.slow([AudioSampleValues.wav_to_list(processed_path)], 1),
                    slow_output
                )
                AudioSampleValues.list_to_wav(
                    AudioStretch.fast([AudioSampleValues.wav_to_list(processed_path)], 1),
                    fast_output
                )

                # Process both the slow and fast versions
                for speed in ['slow', 'fast']:
                    speed_path = os.path.join(temp_folder, f"m{speed}_{base_name}.wav")
                    print(f"Processing {speed} version")
                    output_messages.append(f"Processing {speed} version\n")
                    process_audio_file(speed_path, speed)

            clear_output()  # Clear the output after processing each file
            total_files_processed += 1

        def process_folder(current_folder):
            for item in os.listdir(current_folder):
                item_path = os.path.join(current_folder, item)
                if os.path.isdir(item_path) and item != 'temp':
                    # Recursively process subfolders, excluding the temp folder
                    process_folder(item_path)
                elif os.path.isfile(item_path) and item_path.lower().endswith('.wav'):
                    process_file(item_path)

        process_folder(input_folder)
        print(f"Finished processing {total_files_processed} audio files.")
        output_messages.append(f"Finished processing {total_files_processed} audio files.\n")

        # Delete temp folder
        shutil.rmtree(temp_folder)
        return output_messages