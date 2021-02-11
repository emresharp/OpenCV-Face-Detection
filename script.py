import cv2
cam=cv2.VideoCapture(0)
face_casc=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_casc=cv2.CascadeClassifier("haarcascade_eye.xml")


while True:
	_,screen=cam.read()

	grey=cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY)
	faces=face_casc.detectMultiScale(grey,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(screen,(x,y),(x+w,y+h),(0,255,0),3)
		roi_grey=grey[y:y+h,x:x+w]
		roi_colors=screen[y:y+h,x:x+w]
		eyes=eye_casc.detectMultiScale(roi_grey)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_colors,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)

	cv2.imshow("Screen",screen)
	if cv2.waitKey(1)==ord("q"):
		break;

cam.release()
cv2.destroyAllWindows()
