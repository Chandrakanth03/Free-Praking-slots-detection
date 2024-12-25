import streamlit as st
import cv2
import pickle
import cvzone
import numpy as np
from PIL import Image

# Constants
width, height = 107, 48
positions_file = 'CarParkPos'

# Functions for Slot Picker
def load_positions():
    try:
        with open(positions_file, 'rb') as f:
            return pickle.load(f)
    except:
        return []

def save_positions(pos_list):
    with open(positions_file, 'wb') as f:
        pickle.dump(pos_list, f)

def mouse_click(event, x, y, flags, params):
    global pos_list
    if event == cv2.EVENT_LBUTTONDOWN:
        pos_list.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(pos_list):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                pos_list.pop(i)
    save_positions(pos_list)

# Function for Parking Space Detection
def check_parking_space(img, img_pro, pos_list):
    space_counter = 0
    for pos in pos_list:
        x, y = pos
        img_crop = img_pro[y:y + height, x:x + width]
        count = cv2.countNonZero(img_crop)
        color = (0, 255, 0) if count < 900 else (0, 0, 255)
        thickness = 5 if count < 900 else 2
        if count < 900:
            space_counter += 1
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)
    return space_counter, img

# Streamlit App
st.title("Free Parking Slots Detection")
st.markdown("""
This application detects free parking slots using OpenCV.  
Use the **'Slot Picker'** to select parking slots manually, and **'Detection'** for real-time parking slot status.
""")

# Tabs for Slot Picker and Detection
tabs = st.tabs(["Slot Picker", "Detection"])

# Slot Picker Tab
with tabs[0]:
    st.header("Slot Picker")
    uploaded_image = st.file_uploader("Upload a Parking Lot Image", type=["jpg", "png"])
    if uploaded_image:
        pos_list = load_positions()
        img = np.array(Image.open(uploaded_image))

        # Display image and instructions
        st.image(img, caption="Parking Lot Image", use_container_width=True)
        st.info("Left-click to add a slot. Right-click to remove a slot. Close the window to save positions.")

        if st.button("Open Slot Picker"):
            cv2.imshow("Select Parking Slots", img)
            cv2.setMouseCallback("Select Parking Slots", mouse_click)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        st.success("Slots have been saved!")

# Detection Tab
with tabs[1]:
    st.header("Parking Slot Detection")
    uploaded_video = st.file_uploader("Upload a Parking Lot Video", type=["mp4", "avi"])

    if uploaded_video:
        pos_list = load_positions()
        tfile = f"temp_{uploaded_video.name}"
        with open(tfile, 'wb') as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile)

        stframe = st.empty()  # Streamlit placeholder to display the video frame
        count_placeholder = st.empty()  # Streamlit placeholder to display free slots count

        if st.button("Start Detection"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
                img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 25, 16)
                img_median = cv2.medianBlur(img_thresh, 5)
                kernel = np.ones((3, 3), np.uint8)
                img_dilate = cv2.dilate(img_median, kernel, iterations=1)

                space_counter, processed_frame = check_parking_space(frame, img_dilate, pos_list)

                # Display the processed frame
                stframe.image(processed_frame, channels="BGR", use_container_width=True)

                # Update the count of free parking slots dynamically
                count_placeholder.text(f"Free Slots: {space_counter}/{len(pos_list)}")

            cap.release()
            st.success("Detection Complete!")
