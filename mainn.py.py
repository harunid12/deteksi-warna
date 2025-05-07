import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsv(r, g, b):
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r_, g_, b_)
    cmin = min(r_, g_, b_)
    delta = cmax - cmin

    # Hue
    if delta == 0:
        h = 0
    elif cmax == r_:
        h = (60 * ((g_ - b_) / delta)) % 360
    elif cmax == g_:
        h = (60 * ((b_ - r_) / delta)) + 120
    elif cmax == b_:
        h = (60 * ((r_ - g_) / delta)) + 240

    # Saturation
    s = 0 if cmax == 0 else (delta / cmax)

    # Value
    v = cmax

    return int(h / 2), int(s * 255), int(v * 255)

def get_hsv_histogram(image_path, bins=(30, 32, 32)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gambar '{image_path}' tidak ditemukan.")
    
    height, width, _ = image.shape
    h_hist = np.zeros(bins[0])
    s_hist = np.zeros(bins[1])
    v_hist = np.zeros(bins[2])

    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            h, s, v = rgb_to_hsv(r, g, b)

            h_idx = int(h / (180 / bins[0]))
            s_idx = int(s / (256 / bins[1]))
            v_idx = int(v / (256 / bins[2]))

            h_hist[min(h_idx, bins[0]-1)] += 1
            s_hist[min(s_idx, bins[1]-1)] += 1
            v_hist[min(v_idx, bins[2]-1)] += 1

    # Normalisasi
    h_hist /= np.max(h_hist)
    s_hist /= np.max(s_hist)
    v_hist /= np.max(v_hist)

    return np.concatenate([h_hist, s_hist, v_hist])

def get_general_histogram(hists):
    return np.min(np.array(hists), axis=0)

def get_specific_features(hist, general_hist):
    return np.abs(hist - general_hist)

def classify(test_feature, features):
    dists = [np.mean(np.abs(test_feature - f)) for f in features]
    return np.argmin(dists)

def plot_histogram(hist, label, filename):
    plt.figure(figsize=(12, 4))
    plt.plot(hist, label=label)
    plt.title(label)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'hasil/{filename}.png')
    plt.close()

# === TRAINING ===
hijau_hist = get_hsv_histogram('training_image/hijau.bmp')
campur_hist = get_hsv_histogram('training_image/campur.bmp')
merah_hist = get_hsv_histogram('training_image/merah.bmp')

general_hist = get_general_histogram([hijau_hist, campur_hist, merah_hist])
features = [
    get_specific_features(hijau_hist, general_hist),
    get_specific_features(campur_hist, general_hist),
    get_specific_features(merah_hist, general_hist)
]

# Simpan grafik
plot_histogram(hijau_hist, 'Histogram spesifik Tomat Hijau', 'hist_spesifik_tomat_hijau')
plot_histogram(campur_hist, 'Histogram spesifik Tomat Campur', 'hist_spesifik_tomat_campur')
plot_histogram(merah_hist, 'Histogram spesifik Tomat Merah', 'hist_spesifik_tomat_merah')
plot_histogram(general_hist, 'Histogram General', 'hist_general')


# === UJI ===
uji_img_path = 'testing_image/uji_campur.bmp' #sesuaikan gambar uji
frame = cv2.imread(uji_img_path)
if frame is None:
    raise FileNotFoundError(f"Gambar uji '{uji_img_path}' tidak ditemukan.")

test_hist = get_hsv_histogram(uji_img_path)
test_feature = get_specific_features(test_hist, general_hist)

klasifikasi = classify(test_feature, features)
label = ["Tomat Hijau", "Tomat Campur", "Tomat Merah"][klasifikasi]
print(f"Gambar tersebut diklasifikasikan sebagai: {label}")

# Simpan hasil klasifikasi
cv2.putText(frame, label, (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
cv2.imwrite('hasil/klasifikasi_uji_campur.png', frame)
cv2.imshow("Hasil Klasifikasi", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
