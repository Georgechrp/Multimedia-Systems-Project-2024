import cv2

video_path = "video.avi"   # το βίντεο υπάρχει στο path C:\Users\georg\GITHUB PROJECTS\MultimediaSystemsProject
video = cv2.VideoCapture(video_path)

frames = []
frame_count = 0

while video.isOpened():   # Διαβάζουμε και αποθηκεύουμε κάθε καρέ
    flag, frame = video.read()
    if not flag: break   #Ουσιαστικά, η flag θα γίνει false όταν δεν υπάρχει επόμενο frame
    frames.append(frame)  #Ο πίνακας frames θα μας χρειαστεί αργότερα
    frame_count += 1

video.release() #διασφαλίζουμε ότι το αρχείο δεν παραμένει ανοιχτο

print(f"Σύνολο καρέ: {frame_count}")

'''- - - - - Κωδικοποίηση - - - - -'''
import numpy as np
GOP = 12  # Κάθε 12ο καρέ είναι I-frame
encoded_frames = [] #εδώ θα μπουν τα frames κωδικοποιημένα

for i in range(frame_count):  #Για κάθε καρέ
    if i % GOP == 0:   #Έλεγχος αν το Frame είναι τύπου I
        encoded_frames.append(frames[i])    #για να αποθηκευτεί πλήρως, χωρίς καμία αλλαγή
    else:              #Αλλιώς αν το Frame είναι τύπου P --> υπολογίζεται η εικόνα σφάλματος
        previous_frame = frames[i - 1]
        error_frame = np.clip(frames[i].astype(int) - previous_frame.astype(int), 0, 255).astype(np.uint8)
        encoded_frames.append(error_frame)
        # Αποθήκευση της εικόνας σφάλματος
        if i==1:
            output_path = f"error_frame_{i}.png"
            cv2.imwrite(output_path, error_frame)
            print(f"Η εικόνα σφάλματος αποθηκεύτηκε στο: {output_path}")


'''- - - - - Αποκωδικοποίηση - - - - -'''
decoded_frames = []

for i in range(frame_count):  # Για κάθε καρέ
    if i % GOP == 0:  # Αν το καρέ είναι τύπου I
        # I-frame αποθηκεύεται πλήρως, χωρίς καμία αλλαγή
        decoded_frames.append(encoded_frames[i])
    else:  # Αν το καρέ είναι τύπου P
        # P-frame ανακατασκευάζεται από το προηγούμενο καρέ και την εικόνα σφάλματος
        previous_frame = decoded_frames[-1]
        reconstructed_frame = np.clip(previous_frame.astype(int) + encoded_frames[i].astype(int), 0, 255).astype(np.uint8) # προσθέτουμε την εικόνα σφάλματος
        decoded_frames.append(reconstructed_frame)

# Παράδειγμα αποθήκευσης ανακατασκευασμένου καρέ
output_reconstructed_frame = "reconstructed_frame_1.png"
cv2.imwrite(output_reconstructed_frame, decoded_frames[1])
print(f"Το ανακατασκευασμένο καρέ αποθηκεύτηκε στο: {output_reconstructed_frame}")
