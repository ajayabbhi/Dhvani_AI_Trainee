import cv2
import numpy as np

#shape detection
def detect_shape(approx):
    sides = len(approx)
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        return "Square" if 0.95 < aspectRatio < 1.05 else "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides > 6 and sides < 10:
        return "Polygon"
    else:
        return "Circle"


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0.4)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # -----------------------------
    # 1) DETECT SHAPES AND STORE THEM
    # -----------------------------
    detected = []   # list of: {shape, approx, area}
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        shape = detect_shape(approx)

        detected.append({
            "shape": shape,
            "approx": approx,
            "area": area
        })

    # -----------------------------------------
    # 2) SORT SAME SHAPES BY AREA (DESCENDING)
    # -----------------------------------------
    shape_groups = {}

    for d in detected:
        s = d["shape"]
        if s not in shape_groups:
            shape_groups[s] = []
        shape_groups[s].append(d)

    # Sort each group by area
    for s in shape_groups:
        shape_groups[s] = sorted(shape_groups[s], key=lambda x: x["area"], reverse=True)

    # -----------------------------------------
    # 3) DRAW WITH LABELS (Circle 1, Circle 2...)
    # -----------------------------------------
    for shape in shape_groups:
        for idx, obj in enumerate(shape_groups[shape], start=1):

            approx = obj["approx"]

            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            x = approx.ravel()[0]
            y = approx.ravel()[1] - 10

            label = f"{shape} {idx}"

            cv2.putText(frame, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

    cv2.imshow("Shapes Detection", frame)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
