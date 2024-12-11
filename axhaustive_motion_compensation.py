import cv2
import os
import numpy as np
import json
# Μέγεθος macroblock
MACROBLOCK_SIZE = 16

# Ακτίνα αναζήτησης
SEARCH_RADIUS = 8

def main_menu():
    while True:
        print("\nΕπιλέξτε μια λειτουργία:")
        print("1. Εξαγωγή frames σε ασπρόμαυρες εικόνες (new folder: frames)")
        print("2. Διαίρεση πλαισίων σε macroblocks και αναζήτηση διανυσμάτων κίνησης")
        print("3. Έξοδος")

        choice = input("Εισάγετε τον αριθμό της επιλογής σας: ")

        if choice == "1":
            export_frames()
        elif choice == "2":
            process_macroblocks()
        elif choice == "3":
            print("Έξοδος από το πρόγραμμα.")
            break
        else:
            print("Μη έγκυρη επιλογή. Δοκιμάστε ξανά.")

def export_frames():
    video_path = "video.avi"
    output_folder = "frames"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Σφάλμα: Αδυναμία φόρτωσης του βίντεο.")
        return

    GOP = 12  # Κάθε 12ο καρέ είναι τύπου I, τα υπόλοιπα P
    frames = []
    frame_types = []
    frame_count = 0

    print("Ανάγνωση καρέ από το βίντεο και μετατροπή σε grayscale...")

    while True:
        success, frame = video.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

        if frame_count % GOP == 0:
            frame_types.append("I")
        else:
            frame_types.append("P")

        output_path = f"{output_folder}/frame_{frame_count:03d}.png"
        cv2.imwrite(output_path, gray_frame)
        frame_count += 1

    video.release()

    print(f"Συνολικά καρέ: {frame_count}")
    print(f"Αποθήκευση καρέ ολοκληρώθηκε στον φάκελο: {output_folder}")

def process_macroblocks():
    input_folder = "frames"

    if not os.path.exists(input_folder):
        print("Σφάλμα: Ο φάκελος frames δεν υπάρχει. Εκτελέστε πρώτα την εξαγωγή καρέ (Βήμα 1).")
        return

    # Φόρτωση όλων των καρέ
    frames = []
    filenames = sorted(os.listdir(input_folder))

    print("Φόρτωση καρέ από τον φάκελο frames...")
    for filename in filenames:
        frame_path = os.path.join(input_folder, filename)
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        frames.append(frame)

    if len(frames) < 2:
        print("Σφάλμα: Απαιτούνται τουλάχιστον δύο καρέ για επεξεργασία.")
        return

    # Εκτέλεση της διαδικασίας macroblocks και διανυσμάτων κίνησης
    motion_vectors = process_frames(frames, MACROBLOCK_SIZE, SEARCH_RADIUS)

    # Αποθήκευση των διανυσμάτων κίνησης σε αρχείο
    output_file = "motion_vectors.txt"
    with open(output_file, "w") as f:
        for i, frame_vectors in enumerate(motion_vectors):
            f.write(f"Frame {i + 1} (P-Frame):\n")
            for x, y, motion_vector in frame_vectors:
                f.write(f"  Block ({x},{y}): Motion Vector {motion_vector}\n")
            f.write("\n")

    print(f"Η διαδικασία ολοκληρώθηκε. Τα διανύσματα κίνησης αποθηκεύτηκαν στο: {output_file}")

def divide_into_macroblocks(frame, block_size):
    """
    Χωρίζει ένα καρέ σε macroblocks.
    :param frame: Καρέ ως 2D πίνακας (grayscale).
    :param block_size: Το μέγεθος των macroblocks.
    :return: Λίστα από macroblocks.
    """
    height, width = frame.shape
    macroblocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Ελέγχουμε αν το macroblock είναι εντός των ορίων
            if y + block_size <= height and x + block_size <= width:
                block = frame[y:y + block_size, x:x + block_size]
                macroblocks.append((x, y, block))  # Αποθήκευση συντεταγμένων και block
    return macroblocks


def find_best_match(current_block, reference_frame, x, y, block_size, search_radius):
    """
    Εντοπίζει το macroblock με την καλύτερη αντιστοιχία σε ένα πλαίσιο αναφοράς.
    :param current_block: Το τρέχον macroblock.
    :param reference_frame: Το προηγούμενο πλαίσιο αναφοράς.
    :param x, y: Συντεταγμένες του τρέχοντος macroblock.
    :param block_size: Μέγεθος του macroblock.
    :param search_radius: Ακτίνα αναζήτησης.
    :return: Συντεταγμένες (dx, dy) του διανύσματος κίνησης.
    """
    best_match = None
    best_score = float('inf')  # Ξεκινάμε με την χειρότερη δυνατή βαθμολογία
    height, width = reference_frame.shape

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            ref_x = x + dx
            ref_y = y + dy

            # Ελέγχουμε αν το υποψήφιο macroblock είναι εντός ορίων
            if (0 <= ref_x < width - block_size + 1) and (0 <= ref_y < height - block_size + 1):
                reference_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

                # Υπολογισμός MAD (Mean Absolute Difference)
                score = np.mean(np.abs(current_block - reference_block))

                # Ενημέρωση αν βρέθηκε καλύτερη αντιστοιχία
                if score < best_score:
                    best_score = score
                    best_match = (dx, dy)

    return best_match


def process_frames(frames, macroblock_size, search_radius):
    """
    Διαδικασία αναζήτησης macroblocks για όλα τα P-frames.
    :param frames: Λίστα με καρέ (grayscale).
    :param macroblock_size: Μέγεθος των macroblocks.
    :param search_radius: Ακτίνα αναζήτησης.
    """
    motion_vectors = []  # Αποθήκευση διανυσμάτων κίνησης

    for i in range(1, len(frames)):  # Ξεκινάμε από το 2ο καρέ (P-Frame)
        current_frame = frames[i]
        previous_frame = frames[i - 1]

        print(f"Επεξεργασία καρέ {i} (P-Frame)...")

        frame_motion_vectors = []

        # Διαίρεση του τρέχοντος καρέ σε macroblocks
        macroblocks = divide_into_macroblocks(current_frame, macroblock_size)

        for (x, y, block) in macroblocks:
            # Αναζήτηση του καλύτερου macroblock στο προηγούμενο πλαίσιο
            motion_vector = find_best_match(block, previous_frame, x, y, macroblock_size, search_radius)
            frame_motion_vectors.append((x, y, motion_vector))

            # Εκτύπωση του διανύσματος κίνησης
            #print(f"Macroblock at ({x}, {y}): Motion vector {motion_vector}")

        motion_vectors.append(frame_motion_vectors)

        # Εκτύπωση των διανυσμάτων κίνησης για το καρέ
        print(f"Motion vectors for frame {i}: {frame_motion_vectors}")
    with open("motion_vectors.json", "w") as f:
        json.dump(motion_vectors, f)
    return motion_vectors

if __name__ == "__main__":
    main_menu()

