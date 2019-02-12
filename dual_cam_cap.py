import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
outL = cv2.VideoWriter('outputL.avi', fourcc, 30.0, (1024, 768))  # original: 2048*1536
outR = cv2.VideoWriter('outputR.avi', fourcc, 30.0, (1024, 768))


while(cap.isOpened()):
    ret1, frame1 = cap.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        frame1 = cv2.flip(frame1,0)

        # write the flipped frame
        outL.write(frame1)
        outR.write(frame2)

        windowShow = np.vstack((frame1, frame2))

        cv2.imshow('frame',frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cap2.release()
outL.release()
outR.release()
cv2.destroyAllWindows()