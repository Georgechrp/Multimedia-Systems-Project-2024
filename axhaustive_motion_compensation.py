import cv2
import numpy as np
import os

def split_into_macroblocks(frame_path, block_size=16):
    # Φόρτωση καρέ
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)  # Ανάγνωση σε grayscale
    height, width = frame.shape

    # Επικύρωση διαστάσεων
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError("Οι διαστάσεις του καρέ δεν είναι πολλαπλάσια του block_size.")

    # Δημιουργία macroblocks
    macroblocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = frame[y:y + block_size, x:x + block_size]
            macroblocks.append(block)

    return macroblocks

# Παράδειγμα χρήσης
frame_path = "auxiliary2024/300 original frames/original_frame_0.png"  # Παράδειγμα καρέ
macroblocks = split_into_macroblocks(frame_path)

print(f"Το καρέ χωρίστηκε σε {len(macroblocks)} macroblocks.")
