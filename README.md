# Free Slot Detection Using OpenCV and Streamlit

## Overview
This project leverages OpenCV for detecting free parking slots and integrates it with a web interface using Streamlit. The goal is to provide a real-time solution for parking space management, where users can upload parking lot images or videos and view the number of available spaces directly on a web-based platform.

[Parking Slot Detection - first web page]![screencapture-localhost-8501-2024-12-26-01_19_44](https://github.com/user-attachments/assets/0877c69e-19c0-423e-abc6-13eece6fba00)

[Parking Slot Detection - second web page]![screencapture-localhost-8501-2024-12-26-01_18_58](https://github.com/user-attachments/assets/44bc96ca-3069-4173-9f8a-752875c4f51e)

## Features
- Real-time detection of free parking slots in parking lot images and videos.
- Interactive web interface for uploading images and videos.
- Visual representation of free and occupied parking slots.
- Displays the count of free parking slots.
- Easy integration with user-uploaded content for dynamic slot detection.

## Technologies Used
- Python
- OpenCV
- NumPy
- Streamlit

## Installation
To set up the project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chandrakanth03/Free-Praking-slots-detection-.git
   cd REPO-NAME
2.**Install the required dependencies:**

    pip install opencv-python numpy streamlit
3.**Run the project:**

   streamlit run app.py
This will launch the application on your local server.

## Usage:

1.Open the web application in your browser (default URL: http://localhost:8501).

2.Use the web interface to upload parking lot images or videos.

3.The application will automatically detect the free parking slots and display the number of available spaces.

4.The visual representation will highlight the free and occupied slots in the uploaded media.

## Why This Project Matters

Parking space management is a critical aspect of urban development and infrastructure. Finding available parking in busy urban areas can be a stressful and time-consuming task for drivers. This project aims to solve this problem by providing a solution that not only detects free parking slots but also offers a real-time web-based interface where users can track parking availability. By automating this process, the project simplifies the parking experience, saves time, and potentially contributes to reducing traffic congestion in parking areas.

The system is powered by OpenCV, a powerful computer vision library, which performs image processing techniques such as thresholding and contour detection to determine whether parking slots are occupied or free. By integrating this technology with Streamlit, we have created an intuitive and user-friendly platform that allows anyone to use it easily without requiring deep technical knowledge.

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, feel free to create an issue or submit a pull request.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

. OpenCV documentation for image processing techniques.
. Streamlit documentation for building interactive web applications.
. Inspiration from various computer vision projects.


### Explanation of Changes:

- The new section `### Why This Project Matters` adds a more detailed explanation of the project’s importance and use.
- You can continue adding more such large text blocks as you see fit, using regular Markdown headers, paragraphs, and code formatting for better organization.


