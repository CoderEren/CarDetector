import cv2

img_file = "cars.png"
classifier = 'car_detector.xml'

img = cv2.imread(img_file)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier(classifier)
cars = car_tracker.detectMultiScale(grayscaled_img)

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Car Detector', img)
cv2.waitKey()

print("Code completed!")

