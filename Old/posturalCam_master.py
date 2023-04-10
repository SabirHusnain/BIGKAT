# -*- coding: utf-8 -*-
"""
"""

import time, io, os, glob, pickle, socket, threading, subprocess
# import paramiko
import picamera
import cv2
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

import pdb


def nothing(x):
    """A pass function for some openCV files"""
    pass


class tsOutput(object):
    """Object to write to. Saves camera time stamps to disk"""

    def __init__(self, camera, video_filename, ts_filename):

        self.camera = camera
        self.video_output = io.open(video_filename, 'wb')
        self.ts_output = io.open(ts_filename, 'w')
        self.start_time = None

    def write(self, buf):

        self.video_output.write(buf)  # Write the buffer to the video_output stream

        if self.camera.frame.complete and self.camera.frame.timestamp:

            if self.start_time is None:
                self.start_time = self.camera.frame.timestamp
            self.ts_output.write("{}\n".format((self.camera.frame.timestamp - self.start_time) / 1000))

    def flush(self):

        self.video_output.flush()
        self.ts_output.flush()

    def close(self):
        self.video_output.close()
        self.ts_output.close()


class ts2Output(object):
    """Same as tsOutput but saves the timestamps to an array in memory. The array (tsarray) must be created before calling this function"""

    def __init__(self, camera, video_filename, ts_filename):
        self.camera = camera
        self.video_output = io.open(video_filename, 'wb')
        self.start_time = None
        self.i = 0

        # File subdirectories
        self.calibration_subdir = "calibration_images"
        self.video_file_subdir = "video_files"

    def write(self, buf):
        self.video_output.write(buf)  # Write the buffer to the video_output stream

        if self.camera.frame.complete and self.camera.frame.timestamp:
            tsarray[self.i] = self.camera.frame.timestamp

            self.i += 1

    def flush(self):
        self.video_output.flush()

    def close(self):
        self.video_output.close()


class posturalCam:
    """Postural Camera instance"""

    def __init__(self, cam_no=0, kind='None', resolution=(1280, 720), framerate=60, vflip=True):
        """Initialise the postural cam"""

        self.cam_no = cam_no
        self.resolution = resolution
        self.framerate = framerate
        self.vflip = vflip

        self.start_time_delta = 0.5  # Delay between each stage of camera recording sequence when networked

        if networked:
            self.host_addr = '192.168.0.3'  # Server address
        else:
            self.host_addr = 'localhost'  # Server address

        # File subdirectories

        self.calibration_subdir = "calibration_images"
        self.video_file_subdir = "video_files"

    def start_UDP_server(self):
        """Postural cam server. It will wait for a message telling it to initialise the camera. Then wait for another telling it to start recording"""

        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Bind the socket to the port

        host, port = '192.168.0.3', 9999
        server_address = (host, port)  # The server's address
        print('UDP Server: starting server up on {} port {}'.format(server_address[0], server_address[1]))

        sock.bind(server_address)

        print('UDP Server: waiting to receive message')
        data, address = sock.recvfrom(1024)
        print('UDP Server: Message Recieved from {}'.format(address))
        server_init_time = time.time()  # Time server recieved time message

        data_decoded = data.decode()
        data_decoded = data_decoded.split(
            ',')  # Split the message into components. The first is the time on the client. The second is the time to record for.

        # THe decoded message is a string with two sections (comma seperated). The first indicates the time on the client. The second indicates the required recording time

        while True:

            if time.time() >= server_init_time + self.start_time_delta:
                self.init_camera()
                time.sleep(1)
                self.record_video(float(data_decoded[1]))
                self.destroy_camera()
                break

        if data:
            message = 'r'  # Signal that everything worked.
            sent = sock.sendto(message.encode(), address)
            print('UDP Server: sent {} bytes back to {}'.format(sent, address))

    def UDP_record(self, t):

        init_time = self.start_time + self.start_time_delta

        while True:

            if time.time() >= init_time:
                self.init_camera()
                time.sleep(1)
                self.record_video(t, preview=False)
                self.destroy_camera()
                print('UDP Client: Recording finished')
                break

    def start_UDP_client(self, t):

        recording_Thread = threading.Thread(target=self.UDP_record, args=(t,))

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host, port = '192.168.0.3', 9999
        server_address = (host, port)  # The server's address

        print('UDP Client: Sending message to UDP Server {} on port {}'.format(host, port))
        self.start_time = time.time()

        sent = sock.sendto('{},{}'.format(self.start_time, t).encode(), server_address)  # Send the current time and t)

        print("UDP Client: Start Recording")
        recording_Thread.start()  # Start recording Thread. May have constructer overhead

        print("UDP Client: Waiting on server response")

        data, server = sock.recvfrom(1024)  # Wait for a response

        print('Message from UDP Server: {}'.format(data.decode()))  # Decode data and print it

        sock.close()

        recording_Thread.join()

    def start_TCP_server(self):
        """Transfer the recorded file over to the client"""

        # Create a TCP socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a socket object

        host = self.host_addr  # Get local machine name
        port = 8888  # Choose a port

        print('TCP Server: starting on {} port {}'.format(host, port))

        # Bind to the port
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow the address to be used again
        serversocket.bind((host, port))

        # Queue up 1 request
        serversocket.listen(1)

        # Establist connection
        clientsocket, addr = serversocket.accept()
        print("TCP Server: Connection established with {}".format(addr))

        # Send over video file
        f_vid = open(self.v_fname, 'rb')  # Open file to send

        while True:
            print("TCP Server: Sending file")
            l = f_vid.read(1024)
            if not l:
                break
            clientsocket.send(l)

        clientsocket.close()

        print("TCP Server: Closing Socket")
        serversocket.close()

    def start_TCP_client(self):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        host = self.host_addr
        port = 8888

        i = 0
        while True:
            try:
                s.connect((host, port))
                break
            except ConnectionRefusedError as exp:
                print("TCP Client: Could not connect to server. Trying again in 0.5 seconds")
                i += 0
                if i > 100:
                    print("TCP Client: Failed to conned 100 times")
                    raise ConnectionRefusedError

                time.sleep(0.5)

        f_video_recv = open("temp_out_server.h264", 'wb')

        while True:
            data = s.recv(1024)

            if not data:
                break
            print("TCP Client: Recieveing data")
            f_video_recv.write(data)

        s.close()

    def init_camera(self):
        """Initialise the PiCamera"""
        self.camera = picamera.PiCamera(resolution=self.resolution, framerate=self.framerate)
        self.camera.vflip = True

    def record_video(self, t, preview=True, v_fname="temp_out.h264", v_format="h264"):
        """record video for t seconds and save to file. Save a second file with the time stamps
        Optionally pass the name and format to save the file to.        
        """

        self.v_format = v_format
        self.v_fname = v_fname

        if preview:
            self.camera.start_preview()

        try:
            self.camera.start_recording(tsOutput(self.camera, self.v_fname, "points.csv"), format=self.v_format,
                                        level="4.2")
            self.camera.wait_recording(t)
            self.camera.stop_recording()
        except:
            if preview:
                self.camera.stop_preview()
            raise IOError("Camera could not record")

        if preview:
            self.camera.stop_preview()

    def destroy_camera(self):
        """Close the PiCamera. Always call at the end of recording"""
        self.camera.close()


class posturalProc:
    """Play back video from the PiCamera with OpenCV and process the video files"""

    def __init__(self, cam_no=0, v_format='h264', v_fname="temp_out.h264"):
        """Initialise the processing object. Pass the format and name of the video file to be analysed"""

        self.cam_no = cam_no
        self.v_format = v_format
        self.v_fname = v_fname
        self.v_loaded = False

        ##If possible load the latest calibration parameters
        try:
            with open(os.path.join(os.getcwd(), "calibration", "camera_calib_params.pkl"), 'rb') as f:
                self.calib_params = pickle.load(f)  # Load the calib parameters. [ret, mtx, dist, rvecs, tvecs]
                self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.objpoints, self.imgpoints = self.calib_params
        except:
            print("Initialisation: No camera calibration parameters were found")

    def load_video(self):
        """Load the video. The video filename was specified on initialisation"""

        if self.v_loaded == False:

            self.cap = cv2.VideoCapture(self.v_fname)

            self.v_loaded = True
            print("Video Loading: Video Loaded from file: {}".format(self.v_fname))

        else:
            print("Video Loading: Video already loaded")

    def play_video(self):
        """Play back the video"""

        self.load_video()
        cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

        while True:

            ret, frame = self.cap.read()

            if ret == False:
                # If we are at the end of the video reload it and start again.
                self.v_loaded = False
                self.load_video()
                continue  # Skip to next iteration of while loop

            cv2.imshow("video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        self.v_loaded = False

    def check_v_framerate(self, f_name="points.csv"):
        """Plot the camera framerate. Optionally pass the framerate file name"""

        frame_data = pd.read_csv(f_name)
        plt.plot(1000 / frame_data.diff())
        plt.show()

    def get_calibration_frames(self):
        """Allows us to choose images for calibrating the RPi Camera. The more the merrier (get lots of images from different angles"""

        ##Check for previous files
        master_dir = os.getcwd()
        os.chdir(os.path.join(master_dir, "calibration"))  # Go to the calibration images directory
        file_list = glob.glob("*")

        if file_list != []:

            cont = input("Calibration images exist. Press 'c' to overwrite. Any other key to cancel ")

            if cont.lower() == 'c':

                for f in file_list:
                    os.remove(f)

            else:
                print("Escaping calibration")
                return

        os.chdir(master_dir)
        print("Camera Calibration: Loading Video")
        self.load_video()  # Make sure the calibration video is loaded

        i = 0

        ret, frame = self.cap.read()

        while True:

            if ret == False:
                print("Camera Calibration: No more Frames")
                self.cap.release()
                cv2.destroyAllWindows()
                self.v_loaded = False
                break

            cv2.imshow("Calibration", frame)

            key_press = cv2.waitKey(1) & 0xFF

            if key_press == ord("n"):

                ret, frame = self.cap.read()
                continue

            elif key_press == ord("s"):

                np.save(os.path.join(os.getcwd(), 'calibration', 'calib_img_{}'.format(i)), frame)
                i += 1
                ret, frame = self.cap.read()

                continue

            elif key_press == ord("q"):
                self.cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                self.v_loaded = False
                break

    def camera_calibration(self):
        """Use the camera calibration images to try and calibrate the camera"""

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0)... (8,5,0)

        squares = (9, 6)
        objp = np.zeros((np.prod(squares), 3), np.float32)
        objp[:, :2] = np.mgrid[0:squares[0], 0:squares[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points for all the images
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane

        master_dir = os.getcwd()
        os.chdir(os.path.join(master_dir, "calibration"))  # Go to the calibration images directory

        images = glob.glob("*npy")

        len_images = len(images)
        i = 0
        for fname in images:
            print("Camera Calibration: Processing Image: {} of {}".format(i, len_images))
            img = np.load(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, squares, None)

            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                imgpoints.append(corners)

                cv2.drawChessboardCorners(img, squares, corners, ret)
                cv2.imshow("img", img)
                cv2.waitKey(250)

            i += 1
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        ##Run the calibration

        print("Camera Calibration: Calibrating Camera. Please wait... This may take several moments")
        self.calib_params = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.calib_params = list(self.calib_params)
        self.calib_params.append(objpoints)
        self.calib_params.append(imgpoints)
        print("Camera Calibration: Calibration complete")

        with open("camera_calib_params.pkl", 'wb') as f:
            pickle.dump(self.calib_params, f)

        os.chdir(master_dir)  # Set directory back to master directory.

    def check_camera_calibration(self):
        """Check the reprojection error"""

        tot_error = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error

        print("Check Camera Calibration: Total Error: {}".format(tot_error / len(self.objpoints)))

    def undistort(self, img):
        """undistort an image"""
        h, w = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        return dst

    def undistort_points(self, p):

        dst = cv2.undistortPoints(p, self.mtx, self.dist, R=None, P=self.mtx)

        return dst

    def cube_render(self, img=None):
        """Render a cube over an image
        
        DOESNT WORK AT THE MOMENT FOR SOME REASON! WONT SHOW IMAGE"""

        def draw(img, corners, imgpts):

            corner = tuple(corners[0].ravel())

            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)

            return img

        def draw_cube(img, corners, imgpts):

            imgpts = np.int32(imgpts).reshape(-1, 2)

            img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

            for i, j in zip(range(4), range(4, 8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

            img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

            return img

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0,0,0), (1,0,0)... (8,5,0)
        squares = (9, 6)
        objp = np.zeros((np.prod(squares), 3), np.float32)
        objp[:, :2] = np.mgrid[0:squares[0], 0:squares[1]].T.reshape(-1, 2)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

        if img == None:

            self.load_video()
            i = 0
            while True:
                print("Render Cube: Frame {}".format(i))
                i += 1
                ret, frame = self.cap.read()

                if ret == False:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret2, corners = cv2.findChessboardCorners(gray, squares, None)

                if ret2:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.expand_dims(objp, axis=1), corners2, self.mtx,
                                                                       self.dist)

                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)

                    frame = draw_cube(frame, corners2, imgpts)  # Renders cube

                cv2.imshow('show', frame)
                cv2.waitKey(1)

        self.cap.release()

        cv2.destroyAllWindows()

    def get_markers(self):
        """Processs the video to get the IR markers
        Currently tracking a mobile phone light. May need to get a visible light filter to only allow IR light through"""

        marker_file = "marker_data.csv"
        f = io.open(marker_file, 'w')
        self.load_video()

        cv2.namedWindow("HSV TRACKBARS")
        cv2.createTrackbar("Hue low", "HSV TRACKBARS", 0, 255, nothing)
        cv2.createTrackbar("Hue high", "HSV TRACKBARS", 255, 255, nothing)
        cv2.createTrackbar("Sat low", "HSV TRACKBARS", 0, 255, nothing)
        cv2.createTrackbar("Sat high", "HSV TRACKBARS", 255, 255, nothing)
        cv2.createTrackbar("Val low", "HSV TRACKBARS", 0, 255, nothing)
        cv2.createTrackbar("Val high", "HSV TRACKBARS", 255, 255, nothing)
        cv2.createTrackbar("Radius Low", "HSV TRACKBARS", 0, 50, nothing)
        cv2.createTrackbar("Radius High", "HSV TRACKBARS", 50, 100, nothing)

        i = 0
        next_frame = True

        while True:
            print(i)

            if next_frame:
                next_frame = False
                ret, frame = self.cap.read()

                if ret == False:
                    break
            frame2 = frame.copy()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            hue_low = cv2.getTrackbarPos("Hue low", "HSV TRACKBARS")
            hue_high = cv2.getTrackbarPos("Hue high", "HSV TRACKBARS")
            sat_low = cv2.getTrackbarPos("Sat low", "HSV TRACKBARS")
            sat_high = cv2.getTrackbarPos("Sat high", "HSV TRACKBARS")
            val_low = cv2.getTrackbarPos("Val low", "HSV TRACKBARS")
            val_high = cv2.getTrackbarPos("Val high", "HSV TRACKBARS")
            r_low = cv2.getTrackbarPos("Radius Low", "HSV TRACKBARS")
            r_high = cv2.getTrackbarPos("Radius High", "HSV TRACKBARS")

            lower_blue = np.array([hue_low, sat_low, val_low])
            upper_blue = np.array([hue_high, sat_high, val_high])

            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None

            if len(cnts) > 0:

                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                if r_low < radius and radius < r_high:

                    cv2.circle(frame2, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                    cv2.circle(frame2, (int(x), int(y)), 1, (255, 0, 255), 1)

                    cv2.circle(mask, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                    cv2.circle(mask, (int(x), int(y)), 1, (255, 0, 255), 1)

                    f.write("{},{}\n".format(x, y))

                else:
                    f.write("{},{}\n".format(-9999, -9999))

            cv2.imshow("Raw", frame2)
            cv2.imshow("Mask", mask)

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord("n"):
                break

            if key_press == ord("p"):
                next_frame = True
            i += 1

        f.close()
    #


# def calibrate_cam():
#    
#    myCam = posturalCam()        
#    myCam.init_camera()
#    time.sleep(5)    
#    myCam.record_video(5, preview = True, v_fname = "temp.h264")
#    myCam.destroy_camera()
#
#


class myThread(threading.Thread):
    """A simple thread class. Pass an ID and name for the thread. Pass a function and any arguments to execute in thread"""

    def __init__(self, threadID, name, func, *args, **kwargs):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)


def cam_server():
    """Test the server"""
    myCam = posturalCam()  # Create a camera class
    myCam.start_UDP_server()  # Start a UDP server. This camera will now wait to be instructed to start recording
    myCam.start_TCP_server()


def cam_client():
    """Test the client"""
    myCam = posturalCam()
    myCam.start_UDP_client(1)  # Command server to record for 5 seconds
    myCam.start_TCP_client()


def main():
    # Check who we are
    global networked
    networked = True  # If True will run networked protocol. First checks IP address. If Server run server program. If client run client program

    #    pdb.set_trace()
    if networked:
        ip_addr = subprocess.check_output(['hostname', '-I']).decode().strip(" \n")  # Get IP address

        if ip_addr == '192.168.0.2':
            mode = 'client'
            #            print("I am a {}".format(mode))

            cam_client()  # Start the client camera


        elif ip_addr == '192.168.0.3':
            mode = 'server'
            print("I am a {}".format(mode))
            cam_server()

    else:

        thread1 = myThread(1, 'server_cam', cam_server)
        thread2 = myThread(2, 'client_cam', cam_client)
        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()


def check_synchrony():
    server_cam = posturalProc(0, v_fname='temp_out_server.h264')
    client_cam = posturalProc(0, v_fname='temp_out.h264')

    server_cam.load_video()
    client_cam.load_video()

    ret1, sf = server_cam.cap.read()
    ret2, cf = client_cam.cap.read()

    while True:

        if ret1 and ret2:
            cv2.imshow('serverCam', np.fliplr(sf))
            cv2.imshow('clientCam', np.fliplr(cf))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            ret1, sf = server_cam.cap.read()
            ret2, cf = client_cam.cap.read()


if __name__ == '__main__':
    main()
    check_synchrony()

###############################################################
# -------------------------------------------------------------#
# -------------------------------------------------------------#
# -----------------------NOTES---------------------------------#
# -------------------------------------------------------------#
# -------------------------------------------------------------#
# -------------------------------------------------------------#
# -------------------------------------------------------------#
# -------------------------------------------------------------#

# thread1 = myThread(1, 'server_cam', cam1_server)nnnnnn
# thread2 = myThread(2, 'client_cam', cam2_client)
# thread1.start()
# thread2.start()
#
#
# thread1.join()
# thread2.join()


# proc = posturalProc(v_fname = "temp.h264")
# proc.get_calibration_frames()
# proc.camera_calibration()
# proc.play_video()
# proc.get_markers()
# proc.check_camera_calibration()

# proc.cube_render()

# im = np.load(os.path.join(os.getcwd(), "calibration/calib_img_6.npy"))
#
# dst = proc.undistort(im)
#
# fig, ax = plt.subplots(1,2, sharex= True, sharey = True)
#
# ax[0].imshow(im[:,:,::-1])
# ax[1].imshow(dst[:,:,::-1])


# calibrate_cam()


# myCam = posturalCam()
# myCam.init_camera()
##time.sleep(5)
##tsarray = np.empty(90*15)
# myCam.record_video(10, preview = True)
# myCam.destroy_camera()


# proc = posturalProc()
##proc.load_video()
# proc.get_markers()
##plt.plot(1000/ np.diff((tsarray - tsarray[0]) / 1000), 'o-') 
#
#
##proc.play_video()
# proc.check_v_framerate()
