from datetime import datetime
import cv2

cap = cv2.VideoCapture(0) # 0 is the default camera by accessing the camera

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # get the width of the frame
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # get the height of the frame

print(f'Video Resolution: {width}x{height}') # print the resolution of the video (width, height)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set the width of the frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # set the height of the frame

fps = cap.get(cv2.CAP_PROP_FPS) # get the frames per second
print(f'FPS: {fps}') # print the frames per second

fourcc = cv2.VideoWriter_fourcc(*'XVID') # codec used to write the video
# fourcc is a 4-byte code used to specify the video codec
# XVID is a codec used to compress the video
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height)) # write the video to a file

while True:
    ret, frame = cap.read() # ret is a boolean value that returns true if the frame/camera is available 
    # frame is the image that is captured by the camera

    if not ret:
        print ("Error: failed to capture frame")
        break
    
    frame = cv2.flip(frame, 1) # flip the frame horizontally
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # apply the sobel filter in the x direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # apply the sobel filter in the y direction
    sobel_x = cv2.convertScaleAbs(sobel_x) # convert the image to 8-bit unsigned integers
    sobel_y = cv2.convertScaleAbs(sobel_y) # convert the image to 8-bit unsigned integers
    combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0) # combine the sobel_x and sobel_y images

    cv2.putText(gray, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(gray, f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(gray, f'Resolution: {width}x{height}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.imshow("Video", gray)
    out.write(gray) # write the frame to the video file

    if cv2.waitKey(1) & 0xFF == ord('q'): # if the user presses 'q' the video will stop
        break


cap.release() # release the camera
out.release() # release the video writer
cv2.destroyAllWindows() # close all the windows AND doesnt occupy the RAM.

