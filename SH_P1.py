import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def compute_fft(y, sr):
    fft_result = np.fft.fft(y)
    fft_magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(fft_magnitude), 1/sr)
    return fft_magnitude, frequencies

def plot_spectrum(frequencies, magnitude, title):
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
    plt.title(title)
    plt.xlabel('Frekans (Hz)')
    plt.ylabel('Genlik')
    plt.show()

def plot_difference(frequencies, magnitude_diff, title):
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies[:len(frequencies)//2], magnitude_diff[:len(magnitude_diff)//2])
    plt.title(title)
    plt.xlabel('Frekans (Hz)')
    plt.ylabel('Genlik Farkı')
    plt.show()

def extract_features(fft_magnitude):
    N = 1000
    return np.mean(fft_magnitude[:N])

# Eğitim verilerini yükleme
y_yes, sr = load_audio('D:/SayisalHaberlesme_Proje/Ses_Dosyalari/ses_egitim_dosyalari/yes1.wav')
y_no, _ = load_audio('D:/SayisalHaberlesme_Proje/Ses_Dosyalari/ses_egitim_dosyalari/no1.wav')

# Ses uzunluklarını eşitleme (zero-padding)
max_length = max(len(y_yes), len(y_no))
y_yes = np.pad(y_yes, (0, max_length - len(y_yes)), 'constant')
y_no = np.pad(y_no, (0, max_length - len(y_no)), 'constant')

# FFT hesaplama
fft_yes, frequencies = compute_fft(y_yes, sr)
fft_no, _ = compute_fft(y_no, sr)

# Spektrumları görselleştirme
plot_spectrum(frequencies, fft_yes, 'Yes - Frekans Spektrumu')
plot_spectrum(frequencies, fft_no, 'No - Frekans Spektrumu')

# Özellik çıkarımı
features_yes = extract_features(fft_yes)
features_no = extract_features(fft_no)

# Tanıma algoritması (basit eşik değeri yöntemi)
def recognize(y, sr, threshold):
    y = np.pad(y, (0, max_length - len(y)), 'constant')  # Tanıma için zero-padding
    fft_magnitude, _ = compute_fft(y, sr)
    feature = extract_features(fft_magnitude)
    if feature > threshold:
        return "Yes"
    else:
        return "No"

# Eşik değeri belirleme
threshold = (features_yes + features_no) / 2

# Özellik farkını hesaplama ve yazdırma
feature_difference = features_yes - features_no
print(f"Features Yes: {features_yes}")
print(f"Features No: {features_no}")
print(f"Feature Difference: {feature_difference}")

# FFT magnitüd farklarını hesaplama
fft_difference = np.abs(fft_yes - fft_no)

# FFT farkını görselleştirme
plot_difference(frequencies, fft_difference, 'Yes ve No FFT Magnitüd Farkı')

# Yeni ses dosyasını tanıma örneği
y_test, _ = load_audio('D:/SayisalHaberlesme_Proje/Ses_Dosyalari/test.wav')
result = recognize(y_test, sr, threshold)
print(f"Recognition Result: {result}")
