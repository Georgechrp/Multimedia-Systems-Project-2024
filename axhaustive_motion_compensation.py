def calculate_mad(block1, block2):
    """
    Υπολογισμός της Μέσης Απόλυτης Διαφοράς (MAD) μεταξύ δύο macroblocks.
    """
    return np.sum(np.abs(block1 - block2))

def full_search_motion_compensation():
    if not os.path.exists(encoded_folder):
        os.makedirs(encoded_folder)

    print("\nΕκτέλεση εξαντλητικής αντιστάθμισης κίνησης...")
    motion_vectors = []  # Λίστα για τα διανύσματα κίνησης
    error_frames = []    # Λίστα για τα πλαίσια σφάλματος

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
                    # Το macroblock του τρέχοντος καρέ
                    current_block = current_frame[y:y + 16, x:x + 16]

                    # Ορισμός περιοχής αναζήτησης στο προηγούμενο καρέ
                    y_min = max(0, y - 8)
                    y_max = min(height - 16, y + 8)
                    x_min = max(0, x - 8)
                    x_max = min(width - 16, x + 8)

                    best_match = None
                    min_mad = float('inf')

                    # Εξάντληση σε όλη την περιοχή αναζήτησης
                    for search_y in range(y_min, y_max + 1):
                        for search_x in range(x_min, x_max + 1):
                            # Το υποψήφιο macroblock από το προηγούμενο καρέ
                            candidate_block = previous_frame[search_y:search_y + 16, search_x:search_x + 16]

                            # Υπολογισμός MAD
                            mad = calculate_mad(current_block, candidate_block)
                            if mad < min_mad:
                                min_mad = mad
                                best_match = (search_y, search_x)

                    # Υπολογισμός διανύσματος κίνησης
                    motion_vector = (best_match[0] - y, best_match[1] - x)
                    frame_motion_vectors.append(motion_vector)

                    # Δημιουργία εικόνας σφάλματος
                    matched_block = previous_frame[best_match[0]:best_match[0] + 16, best_match[1]:best_match[1] + 16]
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

    print("Η εξαντλητική αντιστάθμιση κίνησης ολοκληρώθηκε.")

    # Επιστροφή των διανυσμάτων κίνησης και των πλαισίων σφάλματος
    return motion_vectors, error_frames
