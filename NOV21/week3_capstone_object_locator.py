import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# color ranges dictionary
colors = {
    "blue":   (np.array([100,120,40]), np.array([130,255,255])),
    "green":  (np.array([40,70,40]),   np.array([80,255,255])),
    "orange": (np.array([10,150,80]),  np.array([25,255,255])),
    "white":  (np.array([0,0,200]),    np.array([180,40,255])),
    "black":  (np.array([0,0,0]),      np.array([180,255,50]))
}

# red needs 2 ranges
red_ranges = [
    (np.array([0,120,70]),  np.array([10,255,255])),
    (np.array([170,120,70]), np.array([180,255,255]))
]

min_area = 300
max_area = 2000

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # process normal colors
    for name, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(frame, f"{name} object", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # handle red separately
    for lower, upper in red_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                cv2.putText(frame, "red object", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
