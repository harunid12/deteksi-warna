import cv2
import numpy as np

# Baca gambar
image = cv2.imread('image/harun.jpg')  # ganti dengan nama file gambarmu

# Cek apakah gambar berhasil dibaca
if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# Konversi BGR ke YCrCb
img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Pisahkan channel-nya
y, cr, cb = cv2.split(img_y_cr_cb)

# Equalisasi histogram hanya di channel Y
y_eq = cv2.equalizeHist(y)

# Gabungkan kembali
img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))

# Kembalikan ke BGR
img_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)

# Tampilkan hasil
cv2.imshow('Original Image', image)
cv2.imshow('Histogram Equalized Image', img_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
