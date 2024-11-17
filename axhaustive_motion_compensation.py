import numpy as np


def calculate_motion_vectors(prev_frame, curr_frame, block_size=16, search_radius=8):
    height, width = prev_frame.shape
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)
    compensated_frame = np.zeros_like(curr_frame)

    # Διαίρεση σε macroblocks
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            best_match = None
            min_error = float('inf')

            # Macroblock στο τρέχον πλαίσιο
            curr_block = curr_frame[i:i + block_size, j:j + block_size]

            # Εξάντληση αναζήτησης
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    ref_x, ref_y = i + dx, j + dy

                    # Έλεγχος ορίων
                    if (0 <= ref_x < height - block_size and
                            0 <= ref_y < width - block_size):

                        # Macroblock στο προηγούμενο πλαίσιο
                        ref_block = prev_frame[ref_x:ref_x + block_size, ref_y:ref_y + block_size]

                        # Υπολογισμός MAD
                        error = np.sum(np.abs(curr_block - ref_block))

                        # Ενημέρωση για το καλύτερο ταίριασμα
                        if error < min_error:
                            min_error = error
                            best_match = (dx, dy)

            # Καταγραφή διανύσματος κίνησης
            motion_vectors[i // block_size, j // block_size] = best_match

            # Εφαρμογή αντιστάθμισης
            dx, dy = best_match
            ref_x, ref_y = i + dx, j + dy
            compensated_frame[i:i + block_size, j:j + block_size] = prev_frame[ref_x:ref_x + block_size,
                                                                    ref_y:ref_y + block_size]

    return motion_vectors, compensated_frame
