import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():
    blackImg = np.zeros(shape=(512, 512, 3), dtype=np.int16)

    draw_circle(blackImg, center=(400, 100), radius=50, color=(0, 255, 0), thickness=8)
    draw_circle(blackImg, center=(400, 200), radius=50, color=(0, 255, 255), thickness=-1)
    draw_rectangle(blackImg, pt1=(200, 200), pt2=(300, 300), color=(0, 0, 255), thickness=5)
    draw_polygon(blackImg, vertices=[[10, 450], [110, 350], [180, 450]], color=(100, 100, 255), thickness=3)
    draw_filled_rectangle(blackImg, pt1=(200, 25), pt2=(300, 150), color=(0, 150, 105))
    draw_filled_polygon(blackImg, vertices=[[10, 450], [110, 350], [180, 450]], color=(100, 100, 255))
    draw_line(blackImg, pt1=(512, 0), pt2=(0, 512), color=(100, 255, 100), thickness=3)
    draw_text(blackImg, text='MAR', org=(210, 500), font_scale=3, color=(100, 200, 250), thickness=3)

    show_image(blackImg)

def draw_circle(img, center, radius, color, thickness):
    cv2.circle(img=img, center=center, radius=radius, color=color, thickness=thickness)

def draw_rectangle(img, pt1, pt2, color, thickness):
    cv2.rectangle(img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

def draw_polygon(img, vertices, color, thickness):
    pts = np.array(vertices, np.int32)
    pts = pts.reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def draw_filled_rectangle(img, pt1, pt2, color):
    cv2.rectangle(img, pt1=pt1, pt2=pt2, color=color, thickness=-1)

def draw_filled_polygon(img, vertices, color):
    pts = np.array(vertices, np.int32)
    pts = pts.reshape(-1, 1, 2)
    cv2.fillPoly(img, [pts], color=color)

def draw_line(img, pt1, pt2, color, thickness):
    cv2.line(img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

def draw_text(img, text, org, font_scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text=text, org=org, fontFace=font, fontScale=font_scale, color=color, thickness=thickness, lineType=cv2.LINE_AA)

def show_image(img):
    plt.figure(figsize=(14, 9))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
