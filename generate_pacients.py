import pickle
import numpy as np


class Pacient:
    """Class created for storing each patient's features, TUG dataset, and other info

    Attributes:
        data: array of floating point values containing accelerometer values during the patient's TUG test
        fft: array of integers containing the Fourier transform of 'data'
        data_windows: list of float arrays, each containing a window 'data'
        fft_windows: list of integer arrays, each containing a window within 'fft'
    """

    def __init__(self, data, window_size, window_slide, features, tugs):
        self.data = data
        self.fft = pow(abs(np.fft.rfft(data)), 2)[1:100]
        self.data_windows = []
        self.fft_windows = []
        self.features = features
        self.tugs = tugs

        if (1 - window_size) % window_slide:
            print('Tamanho e deslocamento da janela inadequados!')
            exit(0)

        datalen = len(data)
        data_size = int(window_size * datalen)
        data_slide = int(window_slide * datalen)

        # print(datalen)
        # print(data_size)
        # print(data_slide)

        for i in range(0, datalen-data_size+1, data_slide):
            # print(str(i) + ':' + str(i + data_size))
            window = data[i:i + data_size]
            self.data_windows.append(window)
            fft_window = pow(abs(np.fft.rfft(window)), 2)[1:100]
            self.fft_windows.append(fft_window)

        # print(len(self.data_windows))
        # print(len(self.fft_windows))


if __name__ == "__main__":
    dataset = np.load('data_fusion.npy')
    feat_matrix = np.load('featuresAcc.npy')  # [n_patient][pse, psp1, psp2, psp3, pspf1, pspf2, pspf3, wpsp, cpt]
    tugs_list = np.load('segmented_tugs.npy')  # [n_patient][n_TUG][begin, end]
    print('Numero de pacientes: ' + str(len(dataset)))
    patients = []

    window_size = 0.2
    window_slide = 0.1

    for data, features, tugs in zip(dataset, feat_matrix, tugs_list):
        patients.append(Pacient(data, window_size, window_slide, features, tugs))

    try:
        with open('patients.obj', 'wb') as patients_file:
            pickle.dump(patients, patients_file)

    except Exception as e:
        print('ERROR: ' + str(e))
