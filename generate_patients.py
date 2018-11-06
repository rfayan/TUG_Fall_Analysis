import pickle
import numpy as np
from scipy import signal
from gaitAnalysis import read_data_npy, data_segmentTS_TUG, count_TUGs, correct_mask
from gaitAnalysis import index_faller, index_faller3M, index_faller6M, index_faller9M, index_faller12M, index_excluded, index_60, index_70, index_80


def extract_tugs():
	data = read_data_npy()
	masks, _ = data_segmentTS_TUG(data)

	patients_tugs = []
	for patient, mask in enumerate(masks, start=1):
		print('Patient %d TUGs:' % patient)
		count5 = count_TUGs(mask)
		tugs = correct_mask(count5, mask)[0]
		patient_tug = []

		for i in range(tugs[0]):
			begin = tugs[2][i]
			end = begin + tugs[1][i]
			patient_tug.append([begin, end])
			print('TUG %d: [ %d : %d ]' % (i + 1, begin, end))

		patients_tugs.append(patient_tug)

	return patients_tugs


def normalize(a):
	x = np.log(a)
	return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


class Patient:
	"""Class created for storing each patient's features, TUG dataset, and other info

	Attributes:
		data: array of floating point values containing accelerometer values during the patient's TUG test
		fft: array of integers containing the Fourier transform of 'data'
		data_windows: list of float arrays, each containing a window 'data'
		fft_windows: list of integer arrays, each containing a window within 'fft'
		features: array of floating point values, each associated to a specific feature
				[patient_id, pse, psp1, psp2, psp3, pspf1, pspf2, pspf3, wpsp, cpt]
		tugs: list of 9 floating point lists, one for each TUG test, each one with 2 values: beginning and end
	"""

	def __init__(self, data, window_size, window_slide, features, tugs):
		self.data = data
		self.fft = pow(abs(np.fft.rfft(data)), 2)[1:100]
		self.data_windows = []
		self.fft_windows = []
		self.features = features
		self.tugs = tugs

		_, _, Sxx = signal.spectrogram(data, fs=100, nperseg=127)
		self.spectrogram_whole = normalize(Sxx)

		if (1 - window_size) % window_slide:
			print('Tamanho e deslocamento da janela inadequados!')
			exit(0)

		datalen = len(data)
		data_size = int(window_size * datalen)
		data_slide = int(window_slide * datalen)

		# print(datalen)
		# print(data_size)
		# print(data_slide)

		for i in range(0, datalen - data_size + 1, data_slide):
			# print(str(i) + ':' + str(i + data_size))
			window = data[i:i + data_size]
			self.data_windows.append(window)
			fft_window = pow(abs(np.fft.rfft(window)), 2)[1:100]
			self.fft_windows.append(fft_window)

	# print(len(self.data_windows))
	# print(len(self.fft_windows))


def load_features():
	feat_matrix = np.load('featuresAcc.npy').astype(np.float)

	for i in index_faller:
		feat_matrix[i][10] = 1
	for i in index_excluded:
		feat_matrix[i][10] = -1

	for i in index_faller3M:
		feat_matrix[i][11] = 1
	for i in index_faller6M:
		feat_matrix[i][12] = 1
	for i in index_faller9M:
		feat_matrix[i][13] = 1
	for i in index_faller12M:
		feat_matrix[i][14] = 1

	for i in index_60:
		feat_matrix[i][15] = 60
	for i in index_70:
		feat_matrix[i][15] = 70
	for i in index_80:
		feat_matrix[i][15] = 80

	return feat_matrix


if __name__ == "__main__":
	dataset = np.load('data_fusion.npy')
	feat_matrix = load_features()
	tugs_list = extract_tugs()
	print('Numero de pacientes: ' + str(len(dataset)))
	patients = []

	window_size = 0.2
	window_slide = 0.1

	for data, features, tugs in zip(dataset, feat_matrix, tugs_list):
		patients.append(Patient(data, window_size, window_slide, features, tugs))

	try:
		with open('patients.obj', 'wb') as patients_file:
			pickle.dump(patients, patients_file)

	except Exception as e:
		print('ERROR: ' + str(e))
