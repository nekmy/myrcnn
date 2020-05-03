import os
import glob

import cv2


class MovWriter():
    def __init__(self, name, shape):
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.video = cv2.VideoWriter(name+".mp4", fourcc, 30.0, shape)
    
    def write(self, img):
        self.video.write(img)

    def release(self):
        self.video.release()

def mov_split(output_dir, video_path, base_name, ext="jpg"):
    cap = cv2.VideoCapture(video_path)
    base_path = output_dir + "/" + base_name
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    length = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(length), ext), frame)
            n += 1
        else:
            return

def join_cap(video_name, shape, cap_dir):
    mov_writer = MovWriter(video_name, shape)
    cap_paths = glob.glob(cap_dir+"/*.jpg")
    for cap_path in cap_paths:
        img = cv2.imread(cap_path)
        mov_writer.write(img)
    mov_writer.release()

def main():
    # mov_split("cap", "bo.mp4", "cap")
    join_cap("omou", (160, 210), "cap")

if __name__ == "__main__":
    main()
    