# Driver-Drowsiness-Detection-Based-On-Machine-Learning
Driver drowsiness detection using machine learning is a system that utilizes advanced algorithms to recognize when a driver is becoming drowsy or fatigued while driving. The system is designed to monitor the driver's behavior, such as eye movements, head pose, facial expressions, and steering movements, and then use this information to identify signs of drowsiness.

The system is typically implemented using a camera installed on the dashboard that captures images of the driver's face and eyes, which are then analyzed by machine learning algorithms. The algorithms are trained on large datasets of images and driver behavior to learn to recognize the patterns associated with drowsiness.

Once the system detects signs of drowsiness, it can trigger an alarm to alert the driver, activate a vibration system to wake the driver, or even take control of the vehicle and guide it to a safe stop. By alerting drivers to their drowsiness and taking proactive measures to prevent accidents, driver drowsiness detection using machine learning has the potential to save countless lives and prevent serious injuries on the road.

Step 1 – Take image as input from a camera.

Step 2 – Detect the face in the image and create a Region of Interest (ROI).

Step 3 – Detect the eyes from ROI and feed it to the classifier.

Step 4 – Classifier will categorize whether eyes are open or closed.

Step 5 – If eyes are closed then it trigger the alarms for alerting the Driver.

#Project Requirements OpenCV – pip install opencv-python (face and eye detection). TensorFlow – pip install tensorflow (keras uses TensorFlow as backend). Keras – pip install keras (to build our classification model).

Now, Let's see how algorithm work step by step.

Step 1 – Take Image as Input from a Camera

With a webcam, we will take images as input. So to access the webcam, we made an infinite loop that will capture each frame. We use the method provided by OpenCV, cv2.VideoCapture(0) to access the camera and set the capture object (cap). cap.read() will read each frame and we store the image in a frame variable.

cap=cv2.VideoCapture(0) if cap.isOpened(): cap=cv2.VideoCapture(0)

Step 2 – Detect Face in the Image and Create a Region of Interest (ROI)

To detect the face in the image, we need to first convert the image into grayscale as the OpenCV algorithm for object detection takes gray images in the input. We don’t need color information to detect the objects. We will be using haar cascade classifier to detect faces. This line is used to set our classifier face = cv2.CascadeClassifier(‘ path to our haar cascade xml file’). Then we perform the detection using faces = face.detectMultiScale(gray). It returns an array of detections with x,y coordinates, and height, the width of the boundary box of the object. Now we can iterate over the faces and draw boundary boxes for each face.

for(x,y,w,h) in faces: cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

Step 3 – Detect the eyes from ROI and feed it to the classifier

The same procedure to detect faces is used to detect eyes. First, we set the cascade classifier for eyes in leye and reye respectively then detect the eyes using left_eye = leye.detectMultiScale(gray). Now we need to extract only the eyes data from the full image. This can be achieved by extracting the boundary box of the eye and then we can pull out the eye image from the frame with this code.

for x,y,w,h in eyes: roi_gray=gray[y:y+h,x:x+w] roi_color=frame[y:y+h,x:x+w] cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

Step 4 – Classifier will Categorize whether Eyes are Open or Closed

We are using CNN classifier for predicting the eye status. To feed our image into the model, we need to perform certain operations because the model needs the correct dimensions to start with. First, we convert the color image into grayscale using r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY). Then, we resize the image to 2424 pixels as our model was trained on 2424 pixel images cv2.resize(r_eye, (24,24)). We normalize our data for better convergence r_eye = r_eye/255 (All values will be between 0-1). Expand the dimensions to feed into our classifier. We loaded our model using model = load_model(‘my_model.h5’)

new_model = tf.keras.models.load_model('my_model.h5')

Step 5 – If eyes are closed then it trigger the alarms for alerting the Driver.

import cv2 path="haarcascade_frontalface_default.xml" faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0) if cap.isOpened(): cap=cv2.VideoCapture(0) if not cap.isOpened(): raise IOError("cannot open webcam")

while True: ret,frame=cap.read() eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml') gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) eyes=eye_cascade.detectMultiScale(gray,1.1,4) for x,y,w,h in eyes: roi_gray=gray[y:y+h,x:x+w] roi_color=frame[y:y+h,x:x+w] cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) eyess=eye_cascade.detectMultiScale(roi_gray) if len(eyess)==0: print("eyes are not detected") else: for(ex,ey,ew,eh) in eyess: eyes_roi=roi_color[ey:ey+eh,ex:ex+ew] final_image=cv2.resize(eyes_roi,(224,224)) final_image=np.expand_dims(final_image,axis=0) final_image=final_image/255.0
