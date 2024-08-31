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
    def convert(cls, audio_data, oscillations, threshold=100, channel=0):
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
                        
                        if avg_abs_value >= threshold:
                            found_match = False
                            for note in cls.get_all_notes():
                                expected_count = round(cls.sample_rate / cls.note_to_frequency(note))
                                if abs(total_count - expected_count) <= 1:  # Allow for rounding error
                                    oscillations[note].extend(current_oscillation)
                                    print(f"Accepted oscillation for {note}, count: {total_count}")
                                    found_match = True
                                    break
                            if not found_match:
                                print(f"Rejected  oscillation, total count: {total_count}")
                        else:
                            print(f"Rejected  oscillation due to low average absolute value")
                    
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
    def process_audio_folder_normalize(cls, input_folder, output_folder, incomplete_folder=None, threshold=100):
        output_messages = []
        if not os.path.isdir(input_folder):
            print(f"Error: {input_folder} is not a valid directory.")
            output_messages.append(f"Error: {input_folder} is not a valid directory.\n")
            return output_messages

        temp_folder = os.path.join(input_folder, 'temp')
        
        # Check if temp folder exists and delete its contents if it does
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

        oscillations = {note: [] for note in cls.get_all_notes()}
        
        # Import incomplete data from previous runs
        if incomplete_folder and os.path.isdir(incomplete_folder):
            for item in os.listdir(incomplete_folder):
                if item.endswith('_incomplete.txt'):
                    note = item.split('_incomplete.txt')[0]
                    if note in oscillations:  # Only import notes for valid octaves
                        with open(os.path.join(incomplete_folder, item), 'r') as f:
                            incomplete_data = json.load(f)
                        oscillations[note].extend(incomplete_data)
                        print(f"Imported {len(incomplete_data)} incomplete samples for {note}")
                        output_messages.append(f"Imported {len(incomplete_data)} incomplete samples for {note}\n")

        total_files_processed = 0

        def clear_output():
            # Clear the console output
            if sys.platform.startswith('win'):
                os.system('cls')  # For Windows
            else:
                os.system('clear')  # For Unix/Linux/macOS

        def save_oscillations(note, samples):
            if len(samples) < cls.sample_rate:
                # Save as incomplete
                incomplete_file = os.path.join(incomplete_folder, f"{note}_incomplete.txt")
                with open(incomplete_file, 'w') as f:
                    json.dump(samples, f)
                print(f"Saved incomplete data for {note}")
                output_messages.append(f"Saved incomplete data for {note}\n")
            else:
                # Save as complete sample
                note_folder = os.path.join(output_folder, note)
                os.makedirs(note_folder, exist_ok=True)
                
                # Split samples into 10-second chunks
                chunks = [samples[i:i + cls.sample_rate * 10] for i in range(0, len(samples), cls.sample_rate * 10)]
                
                for i, chunk in enumerate(chunks):
                    if len(chunk) >= cls.sample_rate:
                        part_number = cls.get_next_part_number(note_folder, note)
                        output_file = os.path.join(note_folder, f"{note}_{part_number}.wav")
                        AudioSampleValues.list_to_mono_wav(chunk, output_file)
                        print(f"Successfully created audio sample for {note} (part {part_number})")
                        output_messages.append(f"Successfully created audio sample for {note} (part {part_number})\n")
                
                # Remove the incomplete file if it exists
                incomplete_file = os.path.join(incomplete_folder, f"{note}_incomplete.txt")
                if os.path.exists(incomplete_file):
                    os.remove(incomplete_file)
                    print(f"Removed incomplete file for {note}")
                    output_messages.append(f"Removed incomplete file for {note}\n")

        def process_file(item_path):
            nonlocal total_files_processed
            print(f"Processing file")
            output_messages.append(f"Processing file\n")

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

            # Process normalized audio
            audio_data = AudioSampleValues.wav_to_list(normalized_path)
            
            # Process left channel
            print("Processing left channel")
            output_messages.append("Processing left channel\n")
            cls.convert(audio_data, oscillations, threshold)
            
            # Save oscillations after processing left channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)
                oscillations[note] = []  # Clear the processed samples
            
            clear_output()  # Clear the output after processing left channel
            print(f"Finished processing left channel\n")
            output_messages.append(f"Finished processing left channel\n")
            
            # Process right channel
            print("Processing right channel")
            output_messages.append("Processing right channel\n")
            cls.convert(audio_data, oscillations, threshold)
            
            # Save oscillations after processing right channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)
                oscillations[note] = []  # Clear the processed samples
            
            clear_output()  # Clear the output after processing right channel
            print(f"Finished processing right channel")
            output_messages.append(f"Finished processing right channel\n")
            
            total_files_processed += 1

        def process_folder(current_folder):
            for item in os.listdir(current_folder):
                item_path = os.path.join(current_folder, item)
                if os.path.isdir(item_path) and item != 'temp':
                    # Recursively process subfolders, excluding the temp folder
                    process_folder(item_path)
                elif os.path.isfile(item_path):
                    process_file(item_path)

        process_folder(input_folder)
        print(f"Finished processing {total_files_processed} audio files.")
        output_messages.append(f"Finished processing {total_files_processed} audio files.\n")
            

        # Delete temp folder
        shutil.rmtree(temp_folder)
        return output_messages


    @classmethod
    def process_audio_folder(cls, input_folder, output_folder, incomplete_folder=None, threshold=100):
        output_messages = []
        if not os.path.isdir(input_folder):
            print(f"Error: {input_folder} is not a valid directory.")
            output_messages.append(f"Error: {input_folder} is not a valid directory.\n")
            return output_messages

        temp_folder = os.path.join(input_folder, 'temp')
        
        # Check if temp folder exists and delete its contents if it does
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        os.makedirs(temp_folder)

        oscillations = {note: [] for note in cls.get_all_notes()}
        
        # Import incomplete data from previous runs
        if incomplete_folder and os.path.isdir(incomplete_folder):
            for item in os.listdir(incomplete_folder):
                if item.endswith('_incomplete.txt'):
                    note = item.split('_incomplete.txt')[0]
                    if note in oscillations:  # Only import notes for valid octaves
                        with open(os.path.join(incomplete_folder, item), 'r') as f:
                            incomplete_data = json.load(f)
                        oscillations[note].extend(incomplete_data)
                        print(f"Imported {len(incomplete_data)} incomplete samples for {note}")
                        output_messages.append(f"Imported {len(incomplete_data)} incomplete samples for {note}\n")

        total_files_processed = 0

        def clear_output():
            # Clear the console output
            if sys.platform.startswith('win'):
                os.system('cls')  # For Windows
            else:
                os.system('clear')  # For Unix/Linux/macOS

        def save_oscillations(note, samples):
            if len(samples) < cls.sample_rate:
                # Save as incomplete
                incomplete_file = os.path.join(incomplete_folder, f"{note}_incomplete.txt")
                with open(incomplete_file, 'w') as f:
                    json.dump(samples, f)
                print(f"Saved incomplete data for {note} to {incomplete_file}")
                output_messages.append(f"Saved incomplete data for {note} to {incomplete_file}\n")
            else:
                # Save as complete sample
                note_folder = os.path.join(output_folder, note)
                os.makedirs(note_folder, exist_ok=True)
                
                # Split samples into 10-second chunks
                chunks = [samples[i:i + cls.sample_rate * 10] for i in range(0, len(samples), cls.sample_rate * 10)]
                
                for i, chunk in enumerate(chunks):
                    if len(chunk) >= cls.sample_rate:
                        part_number = cls.get_next_part_number(note_folder, note)
                        output_file = os.path.join(note_folder, f"{note}_{part_number}.wav")
                        AudioSampleValues.list_to_mono_wav(chunk, output_file)
                        print(f"Successfully created audio sample for {note} (part {part_number})")
                        output_messages.append(f"Successfully created audio sample for {note} (part {part_number})\n")
                
                # Remove the incomplete file if it exists
                incomplete_file = os.path.join(incomplete_folder, f"{note}_incomplete.txt")
                if os.path.exists(incomplete_file):
                    os.remove(incomplete_file)
                    print(f"Removed incomplete file for {note}")
                    output_messages.append(f"Removed incomplete file for {note}\n")

        def process_file(item_path):
            nonlocal total_files_processed
            print(f"Processing file")
            output_messages.append(f"Processing file\n")

            # Prepare the path for the normalized file
            base_name = os.path.splitext(os.path.basename(item_path))[0]
            normalized_path = os.path.join(temp_folder, f"processed_{base_name}.wav")

            # Check audio file properties
            info = sf.info(item_path)
            if info.subtype != 'PCM_16' or info.samplerate != 44100 or info.channels != 2:
                print(f"Adjusting to 16-bit stereo 44100 Hz WAV")
                output_messages.append(f"Adjusting to 16-bit stereo 44100 Hz WAV\n")
                cls.convert_to_wav_16bit_stereo_44100(item_path, normalized_path)
            else:
                print(f"Using as is")
                output_messages.append(f"Using as is\n")
                normalized_path = item_path

            # Process normalized audio
            audio_data = AudioSampleValues.wav_to_list(normalized_path)
            
            # Process left channel
            print("Processing left channel")
            output_messages.append("Processing left channel\n")
            cls.convert(audio_data, oscillations, threshold)
            
            # Save oscillations after processing left channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)
                oscillations[note] = []  # Clear the processed samples
            
            clear_output()  # Clear the output after processing left channel
            print(f"Finished processing left channel")
            output_messages.append(f"Finished processing left channel\n")
            
            # Process right channel
            print("Processing right channel")
            output_messages.append("Processing right channel\n")
            cls.convert(audio_data, oscillations, threshold)
            
            # Save oscillations after processing right channel
            for note, samples in oscillations.items():
                save_oscillations(note, samples)
                oscillations[note] = []  # Clear the processed samples
            
            clear_output()  # Clear the output after processing right channel
            print(f"Finished processing right channel")
            output_messages.append(f"Finished processing right channel\n")
            
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