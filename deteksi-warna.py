import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hue(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn

    if d == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / d)) % 360
    elif mx == g:
        h = (60 * ((b - r) / d)) + 120
    elif mx == b:
        h = (60 * ((r - g) / d)) + 240
    return int(h / 12)  

def get_hue_histogram(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    hist = np.zeros(30)

    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            idx = rgb_to_hue(r, g, b)
            hist[idx] += 1

    hist = hist / np.max(hist) 
    return hist

def get_general_histogram(hists):
    return np.min(np.array(hists), axis=0)

def get_specific_features(hist, general_hist):
    return np.abs(hist - general_hist)

def classify(test_hist, features):
    dists = [np.mean(np.abs(test_hist - f)) for f in features]
    return np.argmin(dists)

# Ambil fitur dari gambar training
hijau_hist = get_hue_histogram('training_image/hijau.bmp')
campur_hist = get_hue_histogram('training_image/campur.bmp')
merah_hist = get_hue_histogram('training_image/merah.bmp')

general_hist = get_general_histogram([hijau_hist, campur_hist, merah_hist])
features = [
    get_specific_features(hijau_hist, general_hist),
    get_specific_features(campur_hist, general_hist),
    get_specific_features(merah_hist, general_hist)
]

def plot_histograms(hists, labels, title, filename):
    plt.figure(figsize=(10, 6))
    for hist, label in zip(hists, labels):
        plt.plot(range(30), hist, label=label)
    plt.xlabel('Hue Bin')
    plt.ylabel('Normalized Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'hasil/{filename}.png')  
    plt.show()

# Plot histogram hue normalisasi
plot_histograms(
    [hijau_hist, campur_hist, merah_hist],
    ['Tomat Hijau', 'Tomat Campur', 'Tomat Merah'],
    'Histogram Hue Normalisasi Tiap Jenis Tomat',
    'hist_normalisasi'
)

# Plot histogram general (interseksi)
plot_histograms(
    [general_hist],
    ['Histogram General Tomat'],
    'Histogram Interseksi (General Histogram)',
    'hist_general'
)

# Plot histogram fitur spesifik
plot_histograms(
    features,
    ['Fitur Spesifik Hijau', 'Fitur Spesifik Campur', 'Fitur Spesifik Merah'],
    'Fitur Spesifik Berdasarkan Histogram Hue',
    'hist_spesifik'
)

uji_img_path = 'testing_image/uji_merah.bmp'  # path gambar uji

# 1. Load gambar uji
frame = cv2.imread(uji_img_path)

# 2. Hitung histogram hue gambar uji
test_hist = np.zeros(30)
height, width, _ = frame.shape
for i in range(height):
    for j in range(width):
        b, g, r = frame[i, j]
        idx = rgb_to_hue(r, g, b)
        test_hist[idx] += 1
test_hist = test_hist / np.max(test_hist)

# 3. Hitung fitur spesifik gambar uji
test_feature = get_specific_features(test_hist, general_hist)

# 4. Klasifikasi
klasifikasi = classify(test_feature, features)
label = ["Tomat Hijau", "Tomat Campur", "Tomat Merah"][klasifikasi]
print(f"Gambar tersebut diklasifikasikan sebagai: {label}")

cv2.putText(frame, label, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
cv2.imwrite('hasil/klasifikasi_uji_merah.png', frame)
cv2.imshow("Hasil Klasifikasi", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

