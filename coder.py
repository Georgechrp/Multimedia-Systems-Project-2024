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
        print("5. Εξαντλητική αντιστάθμιση κίνησης (Δημιουργία διανυσμάτων κίνησης και πλαισίων σφάλματος)")
        print("6. Έξοδος")

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
            motion_vectors, error_frames = full_search_motion_compensation()
            print("Εξαντλητική αντιστάθμιση κίνησης ολοκληρώθηκε.")
            print(f"Σύνολο διανυσμάτων κίνησης που υπολογίστηκαν: {len(motion_vectors)}")
        elif choice == "6":
            print("Έξοδος από το πρόγραμμα.")
            break
        else:
            print("Μη έγκυρη επιλογή. Δοκιμάστε ξανά.")

# Φόρτωση βίντεο
video_path = "auxiliary2024/video.avi"  # Το βίντεο πρέπει να υπάρχει στο path auxiliary2024/video.avi
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
encoded_folder_motion = "encoded_frames_with_motion"


def encode_video():
    if not os.path.exists(encoded_folder):
        os.makedirs(encoded_folder)

    print("\nΕκτέλεση κωδικοποίησης...")
    for i in range(frame_count):
        if i % GOP == 0:  #Αν το καρέ είναι I - frame
            encoded_frame = zlib.compress(frames[i].tobytes())#Μετατρέπει το numpy array του frame σε ακολουθία bytes και έπειτα συμπιέζει
        else:      #Αν το καρέ είναι P - frame --> υπολογίζεται η εικονα σφάλματος και συμπιέζεται
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

    original_size = sum(frame.nbytes for frame in frames) # συνολικό μέγεθος των αρχικών δεδομένων

    compressed_size = sum(os.path.getsize(f"{encoded_folder}/encoded_frame_{i}.bin") for i in range(frame_count)) #συνολικό μέγεθος των συμπιεσμένων δεδομένων

    # Υπολογίζει τον βαθμό συμπίεσης ως λόγο του αρχικού μεγέθους προς το συμπιεσμένο μέγεθος
    compression_ratio = original_size / compressed_size
    print(f"\nΒαθμός Συμπίεσης:")
    print(f"Μέγεθος αρχικών δεδομένων: {original_size / (1024 ** 2):.2f} MB")
    print(f"Μέγεθος συμπιεσμένων δεδομένων: {compressed_size / (1024 ** 2):.2f} MB")
    print(f"Συνολικός βαθμός συμπίεσης: {compression_ratio:.2f}")

def calculate_mad(block1, block2):#Υπολογισμός της Μέσης Απόλυτης Διαφοράς (MAD) μεταξύ δύο macroblocks.
    return np.sum(np.abs(block1 - block2))

def full_search_motion_compensation():
    if not os.path.exists(encoded_folder_motion):
        os.makedirs(encoded_folder_motion)

    print("\nΕκτέλεση εξαντλητικής αντιστάθμισης κίνησης...")
    motion_vectors = []
    error_frames = []

    for i in range(frame_count):
        if i % GOP == 0:  # Αν το καρέ είναι I-frame
            encoded_frame = zlib.compress(frames[i].tobytes())  # Συμπίεση του I-frame
            encoded_frames.append(encoded_frame)
            motion_vectors.append(None)  # Δεν υπάρχει αντιστάθμιση κίνησης για I-frame
            error_frames.append(np.zeros_like(frames[i]))  # Μηδενικό σφάλμα για I-frame
        else:  # Αν το καρέ είναι P-frame
            current_frame = frames[i]
            previous_frame = frames[i - 1]

            height, width, _ = current_frame.shape
            error_frame = np.zeros_like(current_frame, dtype=np.int16)
            frame_motion_vectors = []

            # Εξέταση όλων των macroblocks
            for y in range(0, height, 16):
                for x in range(0, width, 16):
                    # Το macroblock του τρέχοντος frame
                    current_block = current_frame[y:y + 16, x:x + 16]

                    y_min = max(0, y - 8)
                    y_max = min(height - 16, y + 8)
                    x_min = max(0, x - 8)
                    x_max = min(width - 16, x + 8)

                    best_match = None
                    min_mad = float('inf')

                    # Search σε όλη την περιοχή αναζήτησης
                    for search_y in range(y_min, y_max + 1):
                        for search_x in range(x_min, x_max + 1):
                            # Το υποψήφιο macroblock από το προηγούμενο καρέ
                            candidate_block = previous_frame[search_y:search_y + 16, search_x:search_x + 16]

                            # Εξασφαλίζουμε ότι το υποψήφιο block έχει τις ίδιες διαστάσεις με το τρέχον block
                            if candidate_block.shape != current_block.shape:
                                continue  # Παράλειψη αν οι διαστάσεις ΔΕΝ ταιριάζουν

                            # Υπολογισμός MAD
                            mad = calculate_mad(current_block, candidate_block)

                            if mad < min_mad:
                                min_mad = mad
                                best_match = (search_y, search_x)

                    # Υπολογίζουμε το διανυσμα κίνησης
                    if best_match is None:
                        # Αν δεν βρέθηκε ταίριασμα, χρησιμοποίησε μηδενικό διάνυσμα κίνησης
                        motion_vector = (0, 0)
                        matched_block = current_block
                    else:
                        motion_vector = (best_match[0] - y, best_match[1] - x)
                        matched_block = previous_frame[best_match[0]:best_match[0] + 16, best_match[1]:best_match[1] + 16]

                    frame_motion_vectors.append(motion_vector)

                    # Δημιουργούμε την εικόνας σφάλματος
                    error_block = (current_block.astype(int) - matched_block.astype(int)).astype(np.int16)
                    error_frame[y:y + 16, x:x + 16] = error_block

            # Κωδικοποίηση του πλαισίου σφάλματος
            encoded_frame = zlib.compress(error_frame.tobytes())
            encoded_frames.append(encoded_frame)
            motion_vectors.append(frame_motion_vectors)
            error_frames.append(error_frame)

            # Αποθήκευση του συμπιεσμένου πλαισίου σφάλματος
            with open(f"{encoded_folder}/encoded_frame_{i}.bin", "wb") as f:
                f.write(encoded_frame)

    print("Η εξαντλητική αντιστάθμιση κίνησης ολοκληρώθηκε!")

    return motion_vectors, error_frames



if __name__ == "__main__":
    main_menu()
