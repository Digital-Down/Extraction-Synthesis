"""
__________________________________________________________________________________
#####################
##AudioSampleValues##
#####################
__________________________________________________________________________________

###########
##Methods##
###########

1.wav_to_list		converts a wav to a stereo list of raw audio values
2.list_to_wav		converts a list of raw audio values to wav
__________________________________________________________________________________
####################
##Method Arguments##
####################
User Defined Arguments
1. wav_to_list			WAV File Name.wav
2. list_to_wav			[List]


# Convert the WAV to a list and then back to a WAV file
AudioSampleValues.list_to_wav([AudioSampleValues.wav_to_list('Words.wav')])
__________________________________________________________________________________
"""
import numpy as np
import wave
######
#Main#
######
class AudioSampleValues:
	def __init__(self): 
		pass

	@classmethod
	def wav_to_list(cls, wav_file):
	    with wave.open(wav_file, 'r') as wav:
	        frames = wav.readframes(wav.getnframes())
	        samples = np.frombuffer(frames, dtype=np.int16)
	        samples = samples.reshape(-1, wav.getnchannels())
	    return samples.tolist()

	@classmethod
	def list_to_wav(cls, samples_list, output_file):
	    samples_array = np.array(samples_list, dtype=np.int16)
	    
	    # Create a new WAV file
	    with wave.open(output_file, 'w') as wav_file:
	        wav_file.setnchannels(2)  # Assuming stereo
	        wav_file.setsampwidth(2)  # Assuming 16-bit audio
	        wav_file.setframerate(44100)  # Assuming 44.1kHz sample rate
	        wav_file.writeframes(samples_array.tobytes())
	@classmethod
	def list_to_mono_wav(cls, samples_list, output_file):
	    samples_array = np.array(samples_list, dtype=np.int16)
	    
	    # Create a new WAV file
	    with wave.open(output_file, 'w') as wav_file:
	        wav_file.setnchannels(1)  # Assuming stereo
	        wav_file.setsampwidth(2)  # Assuming 16-bit audio
	        wav_file.setframerate(44100)  # Assuming 44.1kHz sample rate
	        wav_file.writeframes(samples_array.tobytes())

	@classmethod
	def wavs_to_list(cls, folder_path):
		all_samples = []
		for file_name in os.listdir(folder_path):
			if file_name.endswith('.wav'):
				file_path = os.path.join(folder_path, file_name)
				samples = cls.wav_to_list(file_path)
				all_samples.extend(samples)
		return all_samples
# Convert the WAV to a list and then back to a WAV file
#AudioSampleValues.list_to_wav([AudioSampleValues.wav_to_list('Words.wav')])
#AudioSampleValues.list_to_wav([AudioValuesOperations.add([AudioSampleValues.wav_to_list('Words.wav')], 'all', 1000)])

