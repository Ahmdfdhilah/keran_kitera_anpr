import cv2
import imutils
from imutils.video import VideoStream
from detection import process_frame
from config.hikvision import ip_cam_url

# comment ini jika menggunakan hikvision
# ip_cam_url = 0

vs = VideoStream(src=ip_cam_url).start()
print("[INFO] starting video stream...")


while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=1080)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        process_frame(frame)
    
    elif key == ord('q'):
        break

    process_frame(frame)

    cv2.imshow("Frame", frame)

cv2.destroyAllWindows()
vs.stop()
