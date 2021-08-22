import cv2
import uuid
import imutils
import os
from imutils.video import VideoStream


cap = VideoStream(src=0).start()

exit_num = 0
while True:
    frame = cap.read()
    exit_num += 1
    #imgname = "images/mask/{}.jpg".format(str(uuid.uuid1()))
    cv2.imwrite(r"img_cap\{}.jpg".format(str(uuid.uuid1())), frame)
    cv2.imshow("Window Camera",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or exit_num == 10:
        folder_name = r"img_cap"
        for filename in os.listdir(folder_name):
            path_file = os.path.join(folder_name, filename)
            os.remove(path_file)
        break

cv2.destroyAllWindows()
cap.stop()
