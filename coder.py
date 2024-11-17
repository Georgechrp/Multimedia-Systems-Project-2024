import cv2
import numpy as np
import zlib
import os

def main_menu():
    while True:
        print("\nΕπιλέξτε μια λειτουργία:")
        print("1. Πλήθος frames")
        print("2. Εκτέλεση κωδικοποίησης(Δημιουργία φακέλου encoded_frames)")
        print("3. Εκτέλεση αποκωδικοποίησης(Δημιουργία φακέλου decoded_frames)")
        print("4. Υπολογισμός βαθμού συμπίεσης(Θα πρέπει να προηγηθεί το βήμα 2)")
        print("5. Έξοδος")

        choice = input("Εισάγετε τον αριθμό της επιλογής σας: ")

        if choice == "1":
            print(f"Σύνολο καρέ στο βίντεο: {frame_count}")
        elif choice == "2":
            encode_video()
        elif choice == "3":
            decode_video()
        elif choice == "4":
            calculate_compression_ratio()
        elif choice == "5":
            print("Έξοδος από το πρόγραμμα.")
            break
        else:
            print("Μη έγκυρη επιλογή. Δοκιμάστε ξανά.")

# Φόρτωση βίντεο
video_path = "video.avi"  # Το βίντεο πρέπει να υπάρχει στον ίδιο φάκελο με το πρόγραμμα
video = cv2.VideoCapture(video_path)  # Φόρτωση του βίντεο
frames = []
frame_count = 0

while video.isOpened():
    flag, frame = video.read()
    if not flag:
        break
    frames.append(frame)
    frame_count += 1

video.release()

GOP = 12  # Κάθε 12ο καρέ είναι I-Frame
encoded_frames = []
encoded_folder = "encoded_frames"
decoded_folder = "decoded_frames"

def encode_video():
    if not os.path.exists(encoded_folder):
        os.makedirs(encoded_folder)

    print("\nΕκτέλεση κωδικοποίησης...")
    for i in range(frame_count):
        if i % GOP == 0:
            encoded_frame = zlib.compress(frames[i].tobytes())
        else:
            previous_frame = frames[i - 1]
            error_frame = (frames[i].astype(int) - previous_frame.astype(int)).astype(np.int16)
            encoded_frame = zlib.compress(error_frame.tobytes())

        encoded_frames.append(encoded_frame)
        # Αποθήκευση του συμπιεσμένου frame
        with open(f"{encoded_folder}/encoded_frame_{i}.bin", "wb") as f:
            f.write(encoded_frame)
    print(f"Η κωδικοποίηση ολοκληρώθηκε. Τα συμπιεσμένα καρέ αποθηκεύτηκαν στο φάκελο: {encoded_folder}")

def decode_video():
    if not os.path.exists(encoded_folder):
        print("Ο φάκελος κωδικοποιημένων καρέ δεν υπάρχει. Πρέπει να εκτελέσετε πρώτα την κωδικοποίηση.")
        return

    if not os.path.exists(decoded_folder):
        os.makedirs(decoded_folder)

    decoded_frames = []
    print("\nΕκτέλεση αποκωδικοποίησης...")
    for i in range(frame_count):
        with open(f"{encoded_folder}/encoded_frame_{i}.bin", "rb") as f:
            encoded_frame = f.read()

        if i % GOP == 0:
            i_frame_data = zlib.decompress(encoded_frame)
            decoded_frame = np.frombuffer(i_frame_data, dtype=np.uint8).reshape(frames[0].shape)
        else:
            error_data = zlib.decompress(encoded_frame)
            error_frame = np.frombuffer(error_data, dtype=np.int16).reshape(frames[0].shape[:2] + (3,))
            previous_frame = decoded_frames[-1]
            decoded_frame = np.clip(previous_frame.astype(int) + error_frame, 0, 255).astype(np.uint8)

        decoded_frames.append(decoded_frame)
        output_path = f"{decoded_folder}/decoded_frame_{i}.png"
        cv2.imwrite(output_path, decoded_frame)
        #print(f"Η αποκωδικοποιημένη εικόνα αποθηκεύτηκε στο: {output_path}")

    print(f"Η αποκωδικοποίηση ολοκληρώθηκε. Τα αποκωδικοποιημένα καρέ αποθηκεύτηκαν στο φάκελο: {decoded_folder}")

def calculate_compression_ratio():
    if not os.path.exists(encoded_folder):
        print("Ο φάκελος κωδικοποιημένων καρέ δεν υπάρχει. Πρέπει να εκτελέσετε πρώτα την κωδικοποίηση.")
        return

    original_size = sum(frame.nbytes for frame in frames)
    compressed_size = sum(os.path.getsize(f"{encoded_folder}/encoded_frame_{i}.bin") for i in range(frame_count))

    compression_ratio = original_size / compressed_size
    print(f"\nΒαθμός Συμπίεσης:")
    print(f"Μέγεθος αρχικών δεδομένων: {original_size / (1024 ** 2):.2f} MB")
    print(f"Μέγεθος συμπιεσμένων δεδομένων: {compressed_size / (1024 ** 2):.2f} MB")
    print(f"Συνολικός βαθμός συμπίεσης: {compression_ratio:.2f}")

if __name__ == "__main__":
    main_menu()
