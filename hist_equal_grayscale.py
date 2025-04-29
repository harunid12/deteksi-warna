import cv2
import numpy as np

# Baca gambar
image = cv2.imread('image/harun.jpg')  # ganti sesuai lokasi gambarmu

# Cek apakah gambar berhasil dibaca
if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# Konversi ke grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Equalisasi histogram
equalized = cv2.equalizeHist(gray)

# Tampilkan hasil
cv2.imshow('Original Grayscale Image', gray)
cv2.imshow('Histogram Equalized Grayscale Image', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
