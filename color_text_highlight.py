# color_text_highlight.py
# Requirements: opencv-python, numpy
# pip install opencv-python numpy

import cv2
import numpy as np

def nothing(x):
    pass

def create_trackbar_window(win_name="Trackbars"):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H_min", win_name, 0, 179, nothing)
    cv2.createTrackbar("H_max", win_name, 179, 179, nothing)
    cv2.createTrackbar("S_min", win_name, 0, 255, nothing)
    cv2.createTrackbar("S_max", win_name, 255, 255, nothing)
    cv2.createTrackbar("V_min", win_name, 0, 255, nothing)
    cv2.createTrackbar("V_max", win_name, 255, 255, nothing)

def get_trackbar_values(win_name="Trackbars"):
    h_min = cv2.getTrackbarPos("H_min", win_name)
    h_max = cv2.getTrackbarPos("H_max", win_name)
    s_min = cv2.getTrackbarPos("S_min", win_name)
    s_max = cv2.getTrackbarPos("S_max", win_name)
    v_min = cv2.getTrackbarPos("V_min", win_name)
    v_max = cv2.getTrackbarPos("V_max", win_name)
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

def highlight_text_by_color(img, hsv_ranges, debug=False):
    """
    img: BGR image (numpy array)
    hsv_ranges: list of (lower_hsv, upper_hsv) tuples where each is (h, s, v)
    returns: result image with highlighted regions and a mask
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in hsv_ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Clean mask: morphological open then close to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optionally do adaptive thresholding or Canny on masked grayscale for better text contours
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create overlay to highlight text areas
    overlay = img.copy()
    h, w = img.shape[:2]

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(50, 0.0005 * h * w):  # filter very small regions; adapt as needed
            continue
        x,y,ww,hh = cv2.boundingRect(cnt)
        aspect = ww/float(hh+1e-6)
        # Filter by aspect ratio and size heuristics to avoid non-text blobs (adjust thresholds for your image)
        if hh < 8 or ww < 8: 
            continue
        if area < 200: 
            continue
        # optional: require aspect ratio typical of text lines (e.g., between 0.2 and 10)
        if aspect < 0.15 and aspect > 15:
            pass
        boxes.append((x,y,ww,hh))
        # draw filled translucent rectangle
        cv2.rectangle(overlay, (x,y), (x+ww, y+hh), (0,255,255), -1)  # yellow highlight

    alpha = 0.35
    highlighted = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # draw bounding boxes on top for clarity
    for (x,y,ww,hh) in boxes:
        cv2.rectangle(highlighted, (x,y), (x+ww, y+hh), (0,140,255), 2)

    if debug:
        stacked = np.hstack([
            cv2.resize(img, (w, h)),
            cv2.cvtColor(cv2.resize(cleaned, (w, h)), cv2.COLOR_GRAY2BGR),
            cv2.resize(highlighted, (w, h))
        ])
        return highlighted, cleaned, stacked

    return highlighted, cleaned

if __name__ == "__main__":
    # Replace 'billboard.jpg' with your image filename
    img_path = "billboard.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Could not load image. Place 'billboard.jpg' in the script folder or change path.")
        exit(1)

    # Interactive trackbars to find a good HSV range (run once to pick ranges)
    create_trackbar_window()
    # A default initialization that often helps: set S and V min high to catch saturated text
    cv2.setTrackbarPos("S_min", "Trackbars", 50)
    cv2.setTrackbarPos("V_min", "Trackbars", 50)

    print("Adjust trackbars to isolate the text color, then press SPACE to accept ranges and run segmentation.")
    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower, upper = get_trackbar_values()
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_vis = np.hstack([cv2.resize(img, (400,300)), cv2.resize(mask_vis, (400,300))])
        cv2.imshow("Preview (image | mask)", combined_vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            print("Exited by user.")
            cv2.destroyAllWindows()
            exit(0)
        if key == 32:  # SPACE: accept
            break

    cv2.destroyWindow("Trackbars")
    lower_hsv, upper_hsv = get_trackbar_values()
    hsv_ranges = [(lower_hsv, upper_hsv)]
    highlighted, mask = highlight_text_by_color(img, hsv_ranges, debug=False)

    # Save results
    cv2.imwrite("billboard_highlighted.png", highlighted)
    cv2.imwrite("billboard_mask.png", mask)
    print("Saved billboard_highlighted.png and billboard_mask.png")

    # Show final
    cv2.imshow("Highlighted", highlighted)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
