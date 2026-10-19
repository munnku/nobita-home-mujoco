"""データの正解の枠を絵に重ねて確かめる: python learning/show_labels.py <dataset> <split> <n>"""
import glob, os, sys
import cv2
d, split, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
files = sorted(glob.glob(os.path.join(d, "images", split, "*.jpg")))[:n]
tiles = []
for f in files:
    im = cv2.imread(f)
    lab = f.replace("images", "labels").replace(".jpg", ".txt")
    for line in open(lab):
        c, x, y, w, h = map(float, line.split())
        H, W = im.shape[:2]
        p1 = (int((x - w / 2) * W), int((y - h / 2) * H)); p2 = (int((x + w / 2) * W), int((y + h / 2) * H))
        cv2.rectangle(im, p1, p2, (0, 0, 255) if c == 1 else (0, 200, 0), 2)
    tiles.append(cv2.resize(im, (320, 240)))
while len(tiles) % 4:
    tiles.append(tiles[0] * 0)
rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite(os.path.join(d, f"labels_{split}.png"), cv2.vconcat(rows))
