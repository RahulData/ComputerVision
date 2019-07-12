import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)


# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# We're using // here because in Python // allows for int classical division, 
# because we can't pass a float to the cv2.rectangle function

# Coordinates for Rectangle
x = width//2
y = height//2

# Width and height
w = width//4
h = height//4

#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_alt.xml');
def detect_faces(image1):
    gray_img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    return haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);  

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB color 
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     rgb_frame = frame[:, :, ::-1]
#     normal_frame = frame.copy()
    
    # Draw a rectangle on stream
    faces = detect_faces(frame)
    for (x, y, w, h) in faces:     
        cv2.rectangle(frame, (x, y), (x+w, y+h), color=(0,0,255),thickness= 4)

    # Display the resulting frame
    cv2.imshow('frame', frame)

   # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
# plt.imshow(rgb_frame)
# plt.imshow(normal_frame)
# plt.show()