# -*- coding: utf-8 -*-
"""
"""

from __future__ import division, print_function

import sys
import time, io, os, glob, pickle, socket, threading, subprocess
import multiprocessing

if sys.version_info[0] == 2:
    import Queue
else:
    import queue
#import paramikomultiprocessing

if os.name != 'nt':
    import picamera #Import picamera if we are not on windows (fudge to check if on RPi's)
import cv2
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
from pykalman import KalmanFilter #For kalman filtering
from natsort import natsort
import pdb, itertools


def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext
    
def nothing(x):
    """A pass function for some openCV files"""    
    pass

class tsOutput(object):
    """Object to write to. Saves camera time stamps to disk"""
    def __init__(self, camera, video_filename, ts_filename):
        
        
        self.camera = camera
        self.video_output = open(video_filename, 'wb')
        self.ts_output = open(ts_filename, 'w')
        self.start_time = None
        
    def write(self, buf):
        
        self.video_output.write(buf) #Write the buffer to the video_output stream
        
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
        
        #File subdirectories
        self.calibration_subdir = "calibration_images"
        self.video_file_subdir = "video_files"
        
    def write(self, buf):
        
        self.video_output.write(buf) #Write the buffer to the video_output stream
        
        if self.camera.frame.complete and self.camera.frame.timestamp:            
            
            tsarray[self.i] = self.camera.frame.timestamp
            
            self.i +=1 
            
                
    def flush(self):

        self.video_output.flush()


    def close(self):
        self.video_output.close()


class posturalCam:
    """Postural Camera instance"""

    def __init__(self, cam_no = 0, kind = 'None', resolution = (1280, 720), framerate = 60, vflip = False):
        """Initialise the postural cam"""
        
        self.cam_no = cam_no
        self.resolution = resolution
        self.framerate = framerate
        self.vflip = vflip
        
        self.start_time_delta = 0.5 #Delay between each stage of camera recording sequence when networked
        
        if networked:
            self.host_addr = '192.168.0.3' #Server address
        else:
            self.host_addr = 'localhost' #Server address
        
        #File subdirectories
        self.video_fname = None
        self.video_server_fname = None
        self.calibration_subdir = "calibration_images"
        self.video_file_subdir = "video_files"
    
    
                
    def UDP_server_start(self):
        """Postural cam server. It will wait for a message telling it to initialise the camera. Then wait for another telling it to start recording"""
        
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind the socket to the port
        
        host, port = '192.168.0.3', 9999        
        server_address = (host, port) #The server's address
        print('UDP Server: starting server up on {} port {}'.format(server_address[0], server_address[1])) 
        
        sock.bind(server_address)       
            
        print('UDP Server: waiting for data')
        data, address = sock.recvfrom(1024)        
        server_init_time = time.time() #Time server recieved time message
        
        
        data_decoded = data.decode()     
        print('UDP Server: {} bytes recieved from {}: {}'.format(len(data), address, data_decoded)) 
        data_decoded = data_decoded.split(',') #Split the message into components. The first is the time on the client. The second is the time to record for.
        
        #THe decoded message is a string with two sections (comma seperated). The first indicates the time on the client. The second indicates the required recording time
        print("UDP Server: Recording video for {} seconds".format(data_decoded[1]))
        while True:
            
            if time.time() >= server_init_time + self.start_time_delta:
            
                self.init_camera()
#                time.sleep(1)
                self.record_video(float(data_decoded[1]), v_fname = 'server_video.h264')
                self.destroy_camera()
                print("UDP Server: Recording video finished")
                break
        

        if data:
            message = 'recordingSuccess' #Signal that everything worked.
            sent = sock.sendto(message.encode(), address)
            print('UDP Server: sent {} bytes message back to {}: {}'.format(sent, address, message))
    
    def UDP_client_record(self,t):
        
        init_time = self.start_time + self.start_time_delta  - 0.02  #Minus a constant to try and get rid of asynchrony in cameras
        
        while True:
            
            if time.time() >= init_time:
                             
                self.init_camera()
            
            
#                time.sleep(1)        
                self.record_video(t, preview = True, v_fname = self.video_fname)   
                self.destroy_camera()
                print('UDP Client: Recording finished')
                break
            
            
    
    def UDP_client_start(self, t):
        
        recording_Thread = threading.Thread(target = self.UDP_client_record, args = (t,))

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host, port = '192.168.0.3', 9999        
        server_address = (host, port) #The server's address
        
        print('UDP Client: Sending message to UDP Server {} on port {}'.format(host, port))        
        self.start_time = time.time()
        
        sent = sock.sendto('{},{}'.format(self.start_time, t).encode(), server_address) #Send the current time and t)
        
        
        print("UDP Client: Recording video for {} seconds".format(t))
        recording_Thread.start() #Start recording Thread. May have constructer overhead
        
        print("UDP Client: Waiting on server response")
       
        data, server = sock.recvfrom(1024) #Wait for a response
        
        print('UDP Client: Message from UDP Server: {}'.format(data.decode())) #Decode data and print it
        
        print('UDP Client: Closing UDP socket')
        sock.close()
        
        recording_Thread.join() #Wait for recording on Client to end
        print("UDP Client: Recording video finished")
   
    def TCP_server_start(self):
        """A TCP server to manage the serverCam
        
        Wait for a connection from the client. 
        
        Note: After sending video the client will be disconnected. 
        
        Except the following commands from the client:
                'start_UDP': Start a UDP server to wait for recording commands
                #Check UDP server: Check UDP server is live
                'send_video': Request the latest recorded video
                'KILL': Kill the TCP connection. 
        """
        
        #Create a TCP socket
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #Create a socket object
    
        host = self.host_addr #Get local machine name
        port = 8888 #Choose a port
        
        print('TCP Server: starting on {} port {}'.format(host, port))
        
        #Bind to the port
        serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #Allow the address to be used again
        serversocket.bind((host, port)) 
        
        #Queue up 1 request    
        serversocket.listen(1)      
            
        #Establist connection    
            
        while True:
            ##ACCEPT CONNECTIONS IN A CONTINUOUS LOOP.
            
            print("\nTCP Server: Waiting for New connection")
            clientsocket, addr = serversocket.accept()
            print("TCP Server: Connection established with {}".format(addr))
            
            server_live = True #The server is up
            
            while server_live:
                
                print("TCP Server: Waiting for data (Instruction)")
                data = clientsocket.recv(1024)
                
                #Check for data. If not the connection has probably gone down
                if not data:
                    print("TCP Server: Something went wrong with the connection. Please reconnect")
                    server_live = False
                    
                data_decoded = data.decode()
                print("TCP Server: {} Bytes recived from {}: {}".format(len(data), addr, data_decoded))
                
                if data_decoded == 'start_UDP':
                    self.UDP_server_start() #Start a UDP server and wait for recording instructions
                    #We will now wait here until UDP server closes, hopefully with a video recorded
               
                elif data_decoded == 'send_video':   
        
                    #Send over video file       
                    f_vid = open('server_video.h264', 'rb') #Open file to send
                    print("TCP Server: Sending file")
                    
                    while True:
                           
                        l = f_vid.read(1024)    
                        if not l:
                            clientsocket.close()
                            server_live = False 
                            break 
                        
                        clientsocket.send(l) 
                       
                elif data_decoded == 'send_IRPoints':
                    
                    print("TCP Server: Sending END message")
                    
                    #GET THE IR DATA POINTS. THEN SEND THEM 
                    self.proc = posturalProc(v_fname='server_video.h264', kind='server')
                    markers = self.proc.get_ir_markers_parallel() #Get the markers in parallel
                    
                    #Send the marker data
                    marker_bytes = pickle.dumps(markers) #Get bytes object of markers
                    
                    clientsocket.send(marker_bytes) #Send the marker data                  
                    clientsocket.send("END".encode())
                          
                elif data_decoded == 'KILL':  #Kill the connection              
                    
                    clientsocket.close()
                    server_live = False
                    print("TCP Server: TCP connection closed")
                    
                else:
                    print("TCP Server: Message not recognised: {}".format(data_decoded))
                   
                
#                elif data_decoded == 'KILL_SERVER':
#                    clientsocket.close()
#                    serversocket.close()
#                    server_live = False #Change this flag to kill the server
                    
            
  
    def TCP_client_start(self):
        """Start a TCP Client up.
        When done call self.TCP_client_close()"""
        print("") #Print a blank line
        self.TCP_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
        host = self.host_addr
        port = 8888
        
        #Try to make connection 100 times
        i = 0
        while True:
            try:    
                self.TCP_client_socket.connect((host, port))
                break
            except ConnectionRefusedError as exp:
                print("TCP Client: Could not connect to server. Trying again in 0.5 seconds")
                i += 0
                if i > 100:
                    print("TCP Client: Failed to conned 100 times")
                    raise ConnectionRefusedError 
                
                time.sleep(0.25)
                
        print("TCP Client: Connected to {} on port {}".format(host, port))        
        
    def TCP_client_start_UDP(self, t, video_fname):
        """Tell the server to start the UDP server. Wait 0.25 seconds. Then tell the server to record a video of t seconds
        t: The number of seconds the camera will record for
        video_fname: The name of the file to record to. The server video will have _server appended to it.
        """
        
        
        self.video_fname = video_fname #The name of the last video recorded
        split_name = self.video_fname.split('.')
        self.video_server_fname = '{}_server.{}'.format(split_name[0], split_name[1])

            
        self.TCP_client_socket.send('start_UDP'.encode())
        time.sleep(0.25)
        self.UDP_client_start(t)
        
    def TCP_client_request_video(self):
        """Request and recieve the last video from the TCP server. 
        
        Note: Always request the video last! It closes the connection. If you dont want this to happen you will need to reprogram to send information about how big the video file is and then send a end transfer command"""
        
        print("TCP Client: Requesting video from server")
        
        self.TCP_client_socket.send('send_video'.encode()) #Request video 
        
        f_video_recv = open(self.video_server_fname, 'wb')
        
        print("TCP Client: Recieving data")
        
        termination = 'vsc'.encode() #If this byte stream appears close the video. If this byte stream appeared by accident it would cause a problem!!!
        
        while True:
            
            data = self.TCP_client_socket.recv(1024)  

            if not data:
                break
            
            f_video_recv.write(data)    
            
        print("TCP Client: Video Recieved")        
    
    def TCP_client_request_IRPoints(self):
        """Request an IR points file from the server and get IR points from the client. The server will analyse the points data and then pass it to here
        
        Notes: May want to do this as a thread else it will block processing on the client"""
        
        #Start client processing
        self.proc = posturalProc(v_fname=self.video_fname, kind='client') 
        q = queue.Queue(1) #Queue to place the results into
        IR_Thread = threading.Thread(target = self.proc.get_ir_markers_parallel, kwargs = {'out_queue': q})    
        
        IR_Thread.start() #Start the marker thread
        
        ###Get server data
        
        END = 'END'.encode() #If this appears the message is over
        
        print("TCP Client: Requesting IR Points data")
        self.TCP_client_socket.send('send_IRPoints'.encode())
        
        IR_points_byte = [] #Buffer for data
        
        print("TCP Client: Revieving data")
        while True:
            
            data = self.TCP_client_socket.recv(1024)
            
                
            if END in data:         
            
                IR_points_byte.append(data[:data.find(END)])
                print("TCP Client: END notification recieved")
                break
            
            else:
                
                IR_points_byte.append(data)
            
            if len(IR_points_byte) > 1:
                
                last_pair = IR_points_byte[-2] + IR_points_byte[-1] #Join the last two messages
                
                if END in last_pair:
                    IR_points_byte[-2] = last_pair[:last_pair.find(END)] #If END is in the last two messages remove the END message
                    IR_points_byte.pop() #Now get rid of the remaining end message
                    print("TCP Client: END notification recieved")                 
                    break
            
                
         
        IR_points_byte = b''.join(IR_points_byte)
       
        
        self.IR_points_server = pickle.loads(IR_points_byte)
        print("TCP Client: IR Points Data recieved")
        
        if IR_Thread.is_alive():
            IR_Thread.join() #Wait for IR Thread to finish 
        
        
        ##
        return  q.get(), self.IR_points_server
        
      
    
    def TCP_client_close(self):
        """Close the TCP client socket"""
        
        print("TCP Client: Closing connection to server")
        self.TCP_client_socket.send('KILL'.encode()) #CLose the current connection to the TCP server. This is redundant because the video transfer kills it.
        self.TCP_client_socket.close()
                      
           
    def init_camera(self):
        """Initialise the PiCamera"""
        self.camera = picamera.PiCamera(resolution = self.resolution, framerate = self.framerate) 
        self.camera.vflip = self.vflip
#        self.camera.shutter_speed = 700
        
    def record_video(self, t, preview = True, v_fname = "temp_out.h264", v_format = "h264"):
        """record video for t seconds and save to file. Save a second file with the time stamps
        Optionally pass the name and format to save the file to.        
        """        
       
        if preview:
            self.camera.start_preview()
        
        try:
            self.camera.start_recording(tsOutput(self.camera, v_fname, "points.csv"), format = v_format, 
                                        level ="4.2")      
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
    
    def __init__(self, cam_no= 0, v_format = 'h264', v_fname = "temp_out.h264", kind = 'client'):        
        """Initialise the processing object. Pass the format and name of the video file to be analysed"""
        
        self.cam_no = cam_no        
        self.v_format = v_format
        self.v_fname = v_fname
        self.v_loaded = False
        
        self.kind = kind
        self.resolution = (1280, 720)
        
        
        ##If possible load the latest calibration parameters
        if self.kind == 'client':
            calib_fname = 'client_camera_calib_params.pkl'
        elif self.kind == 'server':
            calib_fname = 'server_camera_calib_params.pkl'
        try:
            with open(os.path.join(os.getcwd(), "calibration", calib_fname), 'rb') as f:
                if sys.version_info[0] == 2:
                    self.calib_params = pickle.load(f) #Pickle is different in python 3 vs 2
                else:
                    self.calib_params = pickle.load(f, encoding = "Latin-1") #Load the calib parameters. [ret, mtx, dist, rvecs, tvecs]     
                self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = self.calib_params
                print('Initialisation: Camera calibration parameters were found')
        except:            
            print("Initialisation: No camera calibration parameters were found")
            
        
    def load_video(self):
        """Load the video. The video filename was specified on initialisation"""
        
        if self.v_loaded == False:
            print("Loading video file: {}".format(self.v_fname))
           
            if self.v_fname in glob.glob('*.h264'):
                print("FILE EXISTS")
            self.cap = cv2.VideoCapture(self.v_fname)
#            pdb.set_trace()
            if self.cap.isOpened() != True:                
        
                raise IOError("Video could not be loaded. Check file exists")
            
            
            self.v_loaded = True
            print("Video Loading: Video Loaded from file: {}".format(self.v_fname))
            
        else:
            print("Video Loading: Video already loaded")
    
        
    def play_video(self, omx = True):
        """Play back the video. If omx == True and we are on the raspberry Pi it will play the video via the omxplayer"""
        
       
        if os.name != 'nt':
            RPi = True
        else:
            Rpi = False
            
        if not (Rpi and omx):
            self.load_video()
            cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
            
            
            while True:
                
                ret, frame = self.cap.read()
                
                if ret == False:
                    #If we are at the end of the video reload it and start again. 
                    self.v_loaded = False
                    self.cap.release() #Release the capture. 
                    self.load_video()
                    continue  #Skip to next iteration of while loop
                
                cv2.imshow("video", frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
            
            self.cap.release()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            self.v_loaded = False
        
        elif Rpi:
            subprocess.call(['omx', self.v_fname]) #Call omx player
    
    def average_background(self):
        """Not working"""
        
        self.load_video()
        
        
        ret, frame = self.cap.read()
        avg1 = np.float32(frame)
        i = 0
        while True:
            print(i)
            i +=1
            ret, frame = self.cap.read()
            
            if ret == False:
                #If we are at the end of the video reload it and start again. 
                break
            
            cv2.accumulateWeighted(frame, avg1, 0.05)
            res1 = cv2.convertScaleAbs(avg1)       
            
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        
        plt.imshow(res1)
        
        self.cap.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        self.v_loaded = False
        
    def background_MOG(self):
        """Not working"""
        self.load_video()
        
        fgbg = cv2.createBackgroundSubtractorMOG2()
        
        
        while True:
           
            ret, frame = self.cap.read()
            
            if ret == False:
                #If we are at the end of the video reload it and start again. 
                break
            
            fgmask = fgbg.apply(frame)           
            
            thresh = cv2.threshold(fgmask, 2, 255, cv2.THRESH_BINARY)[1]
            
#            frame[thresh == 0] = [0,0,0]
#            plt.plot(fgmask.flatten())
#            plt.show()
#            pdb.set_trace()
            
            cv2.imshow("fgmask", fgmask)
            cv2.imshow("thresh", thresh)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
    
    def check_v_framerate(self, f_name = "points.csv"):
        """Plot the camera framerate. Optionally pass the framerate file name"""
        
        frame_data = pd.read_csv(f_name)     
        
        fps = 1000 / frame_data.diff()
       
        print(fps['0.0'].mean(), fps['0.0'].std())
        plt.plot(fps)
        plt.show()    

    
    def get_calibration_frames(self):
        """Allows us to choose images for calibrating the RPi Camera. The more the merrier (get lots of images from different angles"""
        
        i = 0
#        Check for previous files
        master_dir = os.getcwd()
        os.chdir(os.path.join(master_dir, "calibration", self.kind)) #Go to the calibration images directory
        file_list = glob.glob("calib*")
        
        if file_list != []:
            
            cont = input("Calibration images exist. Press 'c' to overwrite or 'a' to append. Any other key to cancel ")
            
            if cont.lower() == 'c':
                
                for f in file_list:
                    os.remove(f)
                    
            elif cont.lower() == 'a':
                
             
                last_num = natsort.humansorted(file_list)[-1].split('.tiff')[0].split("_")[-1] #Gets the last image number
                last_num = int(last_num)
                
                i = last_num + 1
                    
            else:
                print("Escaping calibration")
                return 
        
        os.chdir(master_dir)
        
        
        print("Camera Calibration: Loading Video")
        self.load_video() #Make sure the calibration video is loaded      
        
                 
        
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
                print(ret)
                cv2.imwrite(os.path.join(os.getcwd(), 'calibration', self.kind, 'calib_img_{}.tiff'.format(i)), frame)
#                np.save(os.path.join(os.getcwd(), 'calibration', 'calib_img_{}'.format(i)), frame)
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
        
        
        #Find all the images
        
        master_dir = os.getcwd()
        
        os.chdir(os.path.join(master_dir, "calibration", self.kind)) #Go to the calibration images directory
        
        debug_dir = 'output'
        if not os.path.isdir(debug_dir):
            os.mkdir(debug_dir)
        
  
        img_names = glob.glob("*tiff")
        
        square_size = float(24.5)

        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        
        self.obj_points = []
        self.img_points = []
        h, w = 0, 0
        img_names_undistort = []
        for fn in img_names:
            print('processing %s... ' % fn, end='')
            img = cv2.imread(fn, 0)
            if img is None:
                print("Failed to load", fn)
                continue
    
            h, w = img.shape[:2]
            found, corners = cv2.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    
            if debug_dir:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                path, name, ext = splitfn(fn)
               
                outfile = os.path.join(debug_dir,name+'_chess.png')
                cv2.imwrite(outfile, vis)
                if found:
                    img_names_undistort.append(outfile)
    
            if not found:
                print('chessboard not found')
                continue
    
            self.img_points.append(corners.reshape(-1, 2))        
            self.obj_points.append(pattern_points)
    
            print('ok')
    
        # calculate camera distortion
        self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, (w, h), None, None)
    
        print("\nRMS:", self.rms)
        print("camera matrix:\n", self.camera_matrix)
        print("distortion coefficients: ", self.dist_coefs.ravel())
        
        
        ##Save distortion parameters to file
        if self.kind == 'server':
            fname = 'server_camera_calib_params.pkl'
            
        elif self.kind == 'client':
            fname = 'client_camera_calib_params.pkl'
            
        calib_params = self.rms, self.camera_matrix, self.dist_coefs, self.rvecs, self.tvecs
        
        if self.kind == 'server':
            fname = os.path.join(os.path.dirname(os.getcwd()), 'server_camera_calib_params.pkl')
            
        elif self.kind == 'client':
            fname =  os.path.join(os.path.dirname(os.getcwd()),'client_camera_calib_params.pkl')
            
        with open(fname, 'wb') as f:
            pickle.dump(calib_params, f)        #THIS MAY BE BETTER AS A TEXT FILE. NO IDEA IF THIS WILL LOAD ON THE RASPBERRY PI BECAUSE IT IS A BINARY FILE
        
        # undistort the image with the calibration
        print('')
        for img_found in img_names_undistort:
           
            img = cv2.imread(img_found)
    
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h))
    
            dst = cv2.undistort(img, self.camera_matrix, self.dist_coefs, None, newcameramtx)
    
            # crop and save the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            outfile = img_found + '_undistorted.png'
            print('Undistorted image written to: %s' % outfile)
            cv2.imwrite(outfile, dst)
    
        cv2.destroyAllWindows()
        
        os.chdir(master_dir)
       
            

        
    def camera_calibration2(self):
        """DEPRECIATED
        Run after self.camera_calibration. This does the actual optimisizing and saves the parameters to file"""
        
        master_dir = os.getcwd()
        os.chdir(os.path.join(master_dir, "calibration")) #Go to the calibration images directory
        ##Run the calibration
        mtx = np.array([[400, 0, self.resolution[0]/2],
                        [0, 400, self.resolution[1]/2],
                        [0,0,1]])
        print(mtx)                
        print("Camera Calibration: Calibrating Camera. Please wait... This may take several minutes or longer")
        self.calib_params = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.resolution, mtx, None, 
                                                flags =  cv2.CALIB_ZERO_TANGENT_DIST)
        
        
        print(self.calib_params[1])
#        self.calib_params = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.resolution, proc.calib_params[1], proc.calib_params[2], 
#                                                flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST)

        self.calib_params = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.resolution, self.calib_params[1], self.calib_params[2], 
                                                flags = cv2.CALIB_USE_INTRINSIC_GUESS)                                   

     
        
        print("Reprojection Error: {}".format(self.calib_params[0]))
        self.calib_params = list(self.calib_params)
        
        self.calib_params.append(self.objpoints)
        self.calib_params.append(self.imgpoints)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs, self.objpoints, self.imgpoints = self.calib_params
        print("Camera Calibration: Calibration complete")
        
        if self.kind == 'server':
            fname = 'server_camera_calib_params.pkl'
            
        elif self.kind == 'client':
            fname = 'client_camera_calib_params.pkl'
            
        with open(fname, 'wb') as f:
            pickle.dump(self.calib_params, f)
            
        os.chdir(master_dir) #Set directory back to master directory. 
    
    
    def camera_calibration_circle(self):
        """DEPRECIATED
        Use the camera calibration images to try and calibrate the camera"""
        
        #termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

        #Prepare object points, like (0,0,0), (1,0,0)... (8,5,0)
        
        squares = (11,4)     
        
#        ##Square number format
        self.objp = np.zeros((np.prod(squares), 3), np.float32)
        objp[:,:2] = np.mgrid[0:squares[0], 0:squares[1]].T.reshape(-1,2)
#        
        #mm format
        square_size = 24.5 #mm
                
#        objp = np.zeros((np.prod(squares), 3), np.float32)
        
#        p = itertools.product(np.arange(0,square_size*squares[0],square_size), np.arange(0,square_size*squares[1],square_size))
#        objp[:,:2] = np.array([i for i in p])[:,::-1]
            
        #Arrays to store object points and image points for all the images
        self.objpoints = [] #3d points in real world space
        self.imgpoints = [] #2d points in image plane
        
        master_dir = os.getcwd()
        os.chdir(os.path.join(master_dir, "calibration")) #Go to the calibration images directory
        
        images = glob.glob("*npy")
        
        len_images = len(images)
        i = 0
        for fname in images:
            
            img = np.load(fname)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#            gray
#            plt.imshow(gray, cmap = 'gray')
#            plt.show()
#            pdb.set_trace()
            ret, corners = cv2.findCirclesGrid(gray, squares, None, cv2.CALIB_CB_ASYMMETRIC_GRID)
            
            if ret == True:                
                print("Camera Calibration: Processing Image: {} of {}".format(i, len_images))    
                self.objpoints.append(objp)                
#                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                
                self.imgpoints.append(corners)
                
                cv2.drawChessboardCorners(img, squares, corners, ret)
                cv2.imshow("img", img)
                cv2.waitKey(250)
            
            
            i += 1        
        cv2.destroyAllWindows()
        cv2.waitKey(1)
#        pdb.set_trace()    
        ##Run the calibration
        
#        print("Camera Calibration: Calibrating Camera. Please wait... This may take several minutes or longer")
#        self.calib_params = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#        print("Reprojection Error: {}".format(self.calib_params[0]))
#        self.calib_params = list(self.calib_params)
#        self.calib_params.append(objpoints)
#        self.calib_params.append(imgpoints)
#        print("Camera Calibration: Calibration complete")
#        
#        if self.kind == 'server':
#            fname = 'server_camera_calib_params.pkl'
#            
#        elif self.kind == 'client':
#            fname = 'client_camera_calib_params.pkl'
#            
#        with open(fname, 'wb') as f:
#            pickle.dump(self.calib_params, f)
#            
#        os.chdir(master_dir) #Set directory back to master directory. 
        
        
    def check_camera_calibration(self):
        """DEPRECIATED
        Check the reprojection error"""
        
        tot_error = 0
        
        for i in range(len(self.objpoints)):
            
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error
        
        print("Check Camera Calibration: Total Error: {}".format(tot_error/len(self.objpoints)))
    
    
    def undistort(self, img):
        """undistort an image"""
        h,w = img.shape[:2]
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        
      
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        
        return dst
        
    def undistort_points(self, p):
       """Undistort points p where p is a 1XNx2 array or an NX1X2 array""" 
       
       
  
       dst = cv2.undistortPoints(p, self.camera_matrix, self.dist_coefs, P = self.camera_matrix)
                
       return dst       


    def cube_render(self, img = None):
        """DEPRECIATED
        Render a cube over an image
        
        DOESNT WORK AT THE MOMENT FOR SOME REASON! WONT SHOW IMAGE"""
        
        
        def draw(img, corners, imgpts):
            
            corner= tuple(corners[0].ravel())           
            
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)            
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
            
            return img
            
        def draw_cube(img, corners, imgpts):
            
            imgpts = np.int32(imgpts).reshape(-1,2)
            
            img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), -3)
            
            for i,j in zip(range(4), range(4,8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
            
            img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)
            
            return img

        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

        #Prepare object points, like (0,0,0), (1,0,0)... (8,5,0) 
        squares = (9,6)
        objp = np.zeros((np.prod(squares), 3), np.float32)
        objp[:,:2] = np.mgrid[0:squares[0], 0:squares[1]].T.reshape(-1,2)
        
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        
        axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                           [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3]])
    
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
                    
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    
                    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.expand_dims(objp, axis = 1), corners2, self.mtx, self.dist)
                    
                    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)
                    
                    frame = draw_cube(frame, corners2, imgpts) #Renders cube
                
                
                cv2.imshow('show', frame) 
                cv2.waitKey(1)                
           
        
        self.cap.release()

        cv2.destroyAllWindows()
        
    def get_markers_blob(self):
        
        self.load_video()
        
        
        
        i = 0
        next_frame = True
        
#        cv2.namedWindow("TB")
#        cv2.createTrackbar("minThresh", "TB", 0,255, nothing)    
#        cv2.createTrackbar("maxThresh", "TB", 0,255, nothing)             
#        
#        cv2.createTrackbar("minArea", "TB", 0,255, nothing)         
#        cv2.createTrackbar("maxArea", "TB", 500,1500, nothing)         
#        cv2.createTrackbar("minCircularity", "TB", 8, 10, nothing)         
#        
#        cv2.createTrackbar("minConvex", "TB", 87,255, nothing) 
#        cv2.createTrackbar("minIntRatio", "TB", 5,255, nothing) 



        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255
        
        params.filterByColor = True
        params.blobColor = 255
        
        params.filterByArea = True
        params.minArea = 0
        params.maxArea = 1500
        
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.1
            
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        
        detector = cv2.SimpleBlobDetector_create(params)

        
        while True:
            
           print(i)  
           
           if next_frame:
               next_frame = False
               ret, frame = self.cap.read()
            
               if ret == False:
               
                   break 
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
           
#           cv2.imshow("gray", gray)
           
           
           keypoints = detector.detect(gray) 
           im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#           cv2.imshow("Raw", frame)    
           cv2.imshow("blobs", im_with_keypoints)
           key_press = cv2.waitKey(1) & 0xFF
           if  key_press == ord("n"):
               break
           next_frame= True
           i += 1
           if key_press == ord("p"):
               next_frame = True
               i += 1
        
    def get_markers(self):
        """Processs the video to get the IR markers
        Currently tracking a mobile phone light. May need to get a visible light filter to only allow IR light through"""        
        
        
        marker_file = "marker_data.csv"        
#        f = io.open(marker_file, 'w')           
        self.load_video()
        
        cv2.namedWindow("HSV TRACKBARS")
        cv2.createTrackbar("Hue low", "HSV TRACKBARS", 0,255, nothing)    
        cv2.createTrackbar("Hue high", "HSV TRACKBARS", 255, 255, nothing)  
        cv2.createTrackbar("Sat low", "HSV TRACKBARS", 0, 255, nothing)    
        cv2.createTrackbar("Sat high", "HSV TRACKBARS", 255, 255, nothing)  
        cv2.createTrackbar("Val low", "HSV TRACKBARS", 0, 255, nothing)    
        cv2.createTrackbar("Val high", "HSV TRACKBARS", 255, 255, nothing)        
        cv2.createTrackbar("Radius Low", "HSV TRACKBARS", 0, 50, nothing)  
        cv2.createTrackbar("Radius High", "HSV TRACKBARS", 50, 500, nothing)  
        
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
           
#           lower_blue = np.array([0,0,255])
#           upper_blue = np.array([255,255,255])
           
           mask = cv2.inRange(hsv, lower_blue, upper_blue)
           mask = cv2.erode(mask, None, iterations = 2)
           mask = cv2.dilate(mask, None, iterations = 2)          
           
           cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
           center = None
           
           if len(cnts) > 0:
               
               c = max(cnts, key = cv2.contourArea)
               ((x,y), radius) = cv2.minEnclosingCircle(c)
               M = cv2.moments(c)
               center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
               
               if r_low < radius and radius < r_high:
                   
                   cv2.circle(frame2, (int(x), int(y)), int(radius), (0,0,255), 2 )
                   cv2.circle(frame2, (int(x), int(y)), 1, (255,0,255), 1 )
                   
                   cv2.circle(mask, (int(x), int(y)), int(radius), (0,0,255), 2 )
                   cv2.circle(mask, (int(x), int(y)), 1, (255,0,255), 1 )
                   
#                   f.write("{},{}\n".format(x,y))
                   
               else:
#                   f.write("{},{}\n".format(-9999, -9999))
                   pass
           
           cv2.imshow("Raw", frame2)
           cv2.imshow("Mask", mask)
           
           
           
           key_press = cv2.waitKey(1) & 0xFF
           if  key_press == ord("n"):
               break
           
           if key_press == ord("p"):
               next_frame = True
               i += 1
         
        f.close()   

    def ir_marker(self, img):
        """A function to get a single IR marker from and images. Returns the first marker it finds (So terrible if other IR light sources in)"""
                      
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        t_val = 100
        t, thresh = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)
       
        thresh = cv2.erode(thresh, None, iterations = 2)
        thresh = cv2.dilate(thresh, None, iterations = 2)       
       
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None    

        if len(cnts) > 0:
            
           
            c = max(cnts, key = cv2.contourArea)
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
           
#                   if r_low < radius and radius < r_high:

            
            return (x,y), radius, center
        
        else:
            return None #Return NaN if no marker found
    
    
        
        
    def get_ir_markers(self, thresh_track = False, display = True):
            """Processs the video to get the IR markers
            Currently tracking a mobile phone light. May need to get a visible light filter to only allow IR light through"""        
            
            resolution = (1280, 720)
#            marker_file = "marker_data.csv"        
    #        f = io.open(marker_file, 'w')           
            self.load_video()
            
            if thresh_track:
                cv2.namedWindow("GRAY TRACKBARS")
                cv2.createTrackbar("THRESH", "GRAY TRACKBARS", 100, 255, nothing)    
            
#            cv2.namedWindow("Raw", cv2.WINDOW_NORMAL)

            i = 0
            all_markers = []
            
            
            while True:                                              
               
               
                ret, frame = self.cap.read()
                
                if ret == False:
                                     
                    break
                
                i += 1    
                frame2 = frame.copy()   
                
                out = self.ir_marker(frame)
                
                if out != None:
                    
                   
                    (x,y), radius, center = self.ir_marker(frame)
                  
                   
                    if display:         
                       
                       cv2.circle(frame2, (int(x), int(y)), int(radius), (0,0,255), 2)
                       cv2.circle(frame2, (int(x), int(y)), 1, (255,0,255), 1 )
                       
                elif out == None:
           
                    (x,y), radius, center = (np.NaN, np.NaN), np.NaN, np.NaN
                    
                    
                all_markers.append([(x,y), radius, center])    
                
                if display:
                    
                    cv2.imshow("Raw", frame2)
                    key_press = cv2.waitKey(1) & 0xFF

            self.v_loaded = False
            return all_markers
    
            
    def get_ir_markers_parallel(self, thresh_track = False, out_queue = None):                      
        """"Get IR markers from a video file. 
        out_queue is an optionally queue argument. If included the data will be put into the queue
        """
            
#        if __name__ == '__main__':
        print("IN MAIN")
        
        self.load_video() #Load the video
        
        #Create the Workers
        
        max_jobs = 5 #Maximum number of items that can go in queue (may have to be small on RPi)    
        jobs = multiprocessing.Queue(max_jobs)
        results = multiprocessing.Queue() #Queue to place the results into
        
        n_workers = multiprocessing.cpu_count()
        workers = []
        
        for i in range(n_workers):
            print("Starting Worker {}".format(i))
            tmp = multiprocessing.Process(target=get_ir_marker_process, args=(jobs,results))
            tmp.start()
            workers.append(tmp)
        
            print("Worker started")
            
        print("There are {} workers".format(len(workers)))
        i = 0
        while self.cap.isOpened():  
            
            
            if not jobs.full():   
                ret, frame = self.cap.read()                
                if not ret:
                    self.cap.release()
                    break
                ID = i
                jobs.put([ID, frame]) #Put a job in the Queue (the frame is the job)
                if i%10 == 0:
                    print("Get Markers Parallel: Job {} put in Queue".format(i))
                i += 1
                
        print("All jobs complete")
        
        ##Tell all workers to Die
        for worker in workers:        
            jobs.put("KILL")
        
        #Wait for workers to Die
        for worker in workers:    
            worker.join()
        
        
        
        #Get everything out the results queue into a list
        output = []
        print("HERE")
        while results.empty() != True:
            output.append(results.get())
        
        self.v_loaded = False
        
        
        output.sort()
#        pdb.set_trace()
        if not np.alltrue(np.diff(np.array([i[0] for i in output])) == 1):
            raise Exception
        #If a queue object exists put it into it. Useful if function is passed as a thread
        if out_queue != None:
            out_queue.put(output)
                        
            
        return output
    
    def find_ir_markers(self, frame, n_markers = 2, it = 0, tval = 150):
        """A wrapper for the find_ir_markers function. Filter markers by area and adaptively call find_ir_markers"""
        
        
        plot = False #Save results 
        ax = None
        if plot:
            f, ax = plt.subplots(1,1)
            ax.imshow(frame[:,:,::-1])
            plt.suptitle("Frame: {}".format(it))
            
            
        #Initial guess        
        ir = self.find_ir_markers_nofilt(frame, it = i, tval = 150, gamma = None, plot_ax = None)
            
        
        #####IF NO MARKER FOUND DO SOMETHING HERE
        
        
        ##IF MARKERS FOUND BUT ONE IS SUPER LARGE THIS PROBABLY MEANS THERE IS GLARE
        
        max_area = 65 #Maximum area of marker acceptable
        min_area = 7 #Minimum area of marker acceptable
        
        if ir != None:

            rads_max = [mark['radius'] <= max_area for mark in ir] #Check radius of all markers is less than or equal area max
            rads_min = [min_area <= mark['radius'] for mark in ir] #Check radius of all markers is greater than or equal marker min
            
            
            #If marker area is too big it likely indicates either glare or external IR sources 
            if False in rads_max: #
                print("Marker glare. Trying again")
               
                ####LETS TRY TO RUN IT AGAIN WITH A HIGH GAMMA CORRECTION
                ir = proc.find_ir_markers_nofilt(frame, it = i, tval = 150, gamma = 5, plot_ax = None) #Try IR with a very high threshold
                rads_max = [mark['radius'] <= max_area for mark in ir] #Check radius of all markers is less than or equal area max
                rads_min = [min_area <= mark['radius'] for mark in ir] #Check radius of all markers is greater than or equal marker min

        ##FILTER LARGE AND SMALL IR DOTS                
        if ir != None:
            len_ir = len(ir)
            if len_ir > 0:
                mask = rads_max
                ir = np.extract(mask, ir)               
                
                
                #IF there are more than n_markers only get the largest radius markers 
                if len(ir) > n_markers:
                    ir = sorted(ir, key = lambda x: x['radius'], reverse = True) #Order by IR radius size                  
                    ir = ir[:n_markers] #Try to get the number of markers that should be visable
        
        len_ir = 0
        if ir != None:
            len_ir = len(ir)
            
        if len_ir != n_markers:
            print("WARNING: {} markers found".format(len_ir))
            FOUND = False
        else: 
            FOUND = True
            
            
        if plot:
            
            
            if ir != None:
                
                for mark in ir:
                    
                    circle1 = plt.Circle(mark['pos'], mark['radius'] , color='b', fill = False, linewidth = 2)
                    ax.add_artist(circle1)  
                    
            else:
                len_ir = 0
            
            ax.set_title("N Markers: {}".format(len_ir))
    
            if FOUND:
#                 pass
                plt.savefig(".\\markers\\found\\frame_{}.png".format(i), dpi = 50) 
            else:
                plt.savefig(".\\markers\\not_found\\frame_{}.png".format(i)) 
            plt.close()
        
     
        return self.order_ir_markers(ir) #Order the markers and return
            
    def gamma_correction(self, img, correction = 10):

        img = img/255.0
        img = cv2.pow(img, correction)
        return np.uint8(img*255)

    
    def find_ir_markers_nofilt(self, frame, n_markers = 2, it = 0, tval = 150, gamma = None, plot_ax = None):
        """find all IR markers in a single frame. nofilt because it does not filter by any marker properties (e.g. Area, circularity)
        
        frame: numpy array of image
        n_markers: The number of IR markers that should be in image
        it: integer step counter. just for plotting/debugging
        
        To do:
            We need a smarter way of finding markers. The threshold and dilation/errosion may not account well for glare. 
            Glare leads to a large blob being found. In which case the threshold should increase (untill there are only 2 markers).
            Record lots of marker frames in various lighting conditions and work out what is a normal area of the actual markers (after checking that the true markers are being tracked)
        """
        
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        if plot_ax != None:
            plot_ax[1].imshow(gray, cmap = "gray")
            
        if gamma != None:
                
            gray = self.gamma_correction(gray, correction = gamma)
            
            if plot_ax != None:
                plot_ax[2].imshow(gray, cmap = "gray")

         
#            plt.show()

        t_val = tval #Threshold value 
        t, thresh = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)
       
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None)
#        thresh = cv2.erode(thresh, None, iterations = 2) #Are these necessary? Erode and dilate
#        thresh = cv2.dilate(thresh, None, iterations = 2)     
        
        if plot_ax != None:
                plot_ax[3].imshow(gray, cmap = "gray")
       
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None    
        
        n_pos_markers = len(cnts) #Number of possible markers
      
        if n_pos_markers == 0:
            return None
        
        else:
        
            out = []
            
                
            for i in range(n_pos_markers):            
           
                c = cnts[i]
                ((x,y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            
                dat = {'pos': (x,y), 'radius': radius, 'center': center}
                out.append(dat)                    
           

            return out #Return a list of dictionaries. One dict per marker
        
    def order_ir_markers(self, markers):
        """Given a list of 2 IR markers (dicts) make the first item the left most marker. If None is passed also return None"""
        
        
        if markers == None:
            
            return None
            
        else:
    
            x_pos = np.array([i['pos'][0]  for i in markers]) #Get the x position of the markers
            
            left = x_pos.argmin()
            
            if left == 1:
                markers = markers[::-1]
                
            return markers

    def markers2numpy(self, all_markers, n_markers = 2):
        """Takes a list of all markers in a video and converts the positions to a numpy array"""
        
        markers = np.empty((len(all_markers), n_markers, 2))
        
        for i in range(len(markers)):
            
            if all_markers[i] == None:
                markers[i] = np.NaN
                
            else:
                
                for j in range(len(markers[i])):
                    
                    markers[i,j] = all_markers[i][j]['pos'] 
        
        return markers
        
def get_ir_marker_process(jobs, results, parent = None):
            
            
        while True:
            
            in_job = jobs.get() #Get the job ID and the job
            
            
            if isinstance(in_job, str):
                if in_job == "KILL":
                    
                    return 
                    
            
            
            else:         #
                ID, j = in_job[0], in_job[1]
                (x,y), radius, center = (np.NaN, np.NaN), np.NaN, np.NaN
                ##Analyse the image
                gray = cv2.cvtColor(j, cv2.COLOR_BGR2GRAY)
        
                t_val = 100
                t, thresh = cv2.threshold(gray, t_val, 255, cv2.THRESH_BINARY)
               
                thresh = cv2.erode(thresh, None, iterations = 2)
                thresh = cv2.dilate(thresh, None, iterations = 2)       
               
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
                center = None    
   
                if len(cnts) > 0:
                    
                   
                    c = max(cnts, key = cv2.contourArea)
                    ((x,y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                   
        #                   if r_low < radius and radius < r_high:
        
                    
                out = ID, (x,y), radius, center #Place in results queue with the ID


            results.put(out)
            print("hi")
        return #End function  
        
class stereo_process:
    
    def __init__(self, cam_client, cam_serv):
        """Pass two camera processing objects"""
        
        self.cam_serv = cam_serv
        self.cam_client = cam_client
        
        self.resolution = self.cam_client.resolution
        
        self.rectif = None #None if stereorectify has not been called
        
        try:
            with open(os.path.join(os.getcwd(), "calibration", 'stereo_camera_calib_params.pkl'), 'rb') as f:
                self.stereo_calib_params = pickle.load(f, encoding = "Latin-1") #Load the calib parameters. [ret, mtx1, dist1, mtx2, dts2, R, T, E, F    
#                pdb.set_trace()
                self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F, self.P1, self.P2= self.stereo_calib_params
                print('Initialisation: Stereo Camera calibration parameters were found')
        except:            
            print("Initialisation: No stereo camera calibration parameters were found")
            
    def get_calibration_frames(self):
        """Allows us to choose images for calibrating the RPi Camera. The more the merrier (get lots of images from different angles"""
        
        resolution = (1280, 720) #Image resolution
        ##Check for previous files
        
        i = 0
#        Check for previous files
        master_dir = os.getcwd()
        os.chdir(os.path.join(master_dir, "calibration", 'stereo')) #Go to the calibration images directory
        file_list = glob.glob("calib*")
        
        if file_list != []:
            
            cont = input("Calibration images exist. Press 'c' to overwrite or 'a' to append. Any other key to cancel ")
            
            if cont.lower() == 'c':
                
                for f in file_list:
                    os.remove(f)
                    
            elif cont.lower() == 'a':
                
             
                last_num = natsort.humansorted(file_list)[-1].split('.tiff')[0].split("_")[-1] #Gets the last image number
                last_num = int(last_num)
                
                i = last_num + 1
                    
            else:
                print("Escaping calibration")
                return 
        
        os.chdir(master_dir)       
        

        print("Camera Calibration: Loading Video")
        
        #Load both videos        
        self.cam_serv.load_video() #Make sure the calibration video is loaded      
        self.cam_client.load_video()           

        
        #Get first frame from both videos
        ret_serv, frame_serv = self.cam_serv.cap.read()
        ret_client, frame_client = self.cam_client.cap.read()
        
            

        cv2.namedWindow("Server", cv2.WINDOW_NORMAL)     
        cv2.namedWindow("Client", cv2.WINDOW_NORMAL)
#               cv2.resizeWindow("Raw", int(resolution[0]/2), int(resolution[1]/2)) 
        while True:
                                    
            
            if ret_serv == False and ret_client == False:
                print("Camera Calibration: No more Frames")
                self.cam_serv.cap.release()
                self.cam_client.cap.release()
                cv2.destroyAllWindows()
                
                self.cam_serv.v_loaded = False
                self.cam_client.v_loaded = False
                break
            
            cv2.imshow("Server", frame_serv)
            cv2.imshow("Client", frame_client)
            
            cv2.resizeWindow("Server", int(resolution[0]/2), int(resolution[1]/2)) 
            cv2.resizeWindow("Client", int(resolution[0]/2), int(resolution[1]/2)) 
            
            key_press = cv2.waitKey(1) & 0xFF
            
            if key_press == ord("n"):
                
                ret_serv, frame_serv = self.cam_serv.cap.read()
                ret_client, frame_client = self.cam_client.cap.read()
                continue
           
            elif key_press == ord("s"):         
                
                cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo', 'calib_img_serv_{}.tiff'.format(i)), frame_serv)
                cv2.imwrite(os.path.join(os.getcwd(), 'calibration', 'stereo', 'calib_img_client_{}.tiff'.format(i)), frame_client)
                
#                np.save(os.path.join(os.getcwd(), 'calibration', 'calib_img_serv_{}'.format(i)), frame_serv)
#                np.save(os.path.join(os.getcwd(), 'calibration', 'calib_img_client_{}'.format(i)), frame_client)
                
                
                i += 1
                ret_serv, frame_serv = self.cam_serv.cap.read()
                ret_client, frame_client = self.cam_client.cap.read()#               
                continue
            
            elif key_press == ord("q"):
                self.cam_serv.cap.release()
                self.cam_client.cap.release()                
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                self.cam_serv.v_loaded = False    
                self.cam_client.v_loaded = False
                break
        
    def stereo_calibrate(self):
        
        print("Running Stereo Calibration")
        master_dir = os.getcwd()
        
        os.chdir(os.path.join(master_dir, "calibration", 'stereo')) #Go to the calibration images
                
        client_dir = 'client'
        server_dir = 'server'

        img_names_client = glob.glob("{}/*tiff".format(client_dir))
        img_names_server = glob.glob("{}/*tiff".format(server_dir))
        
        img_names_client = natsort.humansorted(img_names_client)
        img_names_server = natsort.humansorted(img_names_server)
        
        
        square_size = 24.5

        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        
        
        self.obj_points = []
        self.img_points_client = []
        self.img_points_server = []
        h, w = 0, 0
        
        for fn in range(len(img_names_client)):
            print('processing %s... ' % img_names_client[fn], end='')
            print('processing %s... ' % img_names_server[fn], end='')
            
            img_client = cv2.imread(img_names_client[fn], 0)
            img_server = cv2.imread(img_names_server[fn], 0)
            
            if img_client is None:
                print("Failed to load", img_names_client[fn])
                continue
            
            if img_server is None:
                print("Failed to load", img_names_server[fn])
                continue
            
            h, w = img_client.shape[:2]
            
            found_client, corners_client = cv2.findChessboardCorners(img_client, pattern_size)
            found_server, corners_server = cv2.findChessboardCorners(img_server, pattern_size)
            
            if found_client and found_server:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)                
                cv2.cornerSubPix(img_client, corners_client, (5, 5), (-1, -1), term)
                cv2.cornerSubPix(img_server, corners_server, (5, 5), (-1, -1), term)
                
            if not (found_client and found_server):
                print('chessboard not found in both images')
                continue  
           
            self.img_points_client.append(corners_client.reshape(-1,2))
            self.img_points_server.append(corners_server.reshape(-1,2))
            self.obj_points.append(pattern_points)
            
            print('ok')
    

     
        self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(self.obj_points, self.img_points_client, self.img_points_server, 
                                                                                                     self.cam_client.camera_matrix, self.cam_client.dist_coefs,
                                                                                                     self.cam_serv.camera_matrix, self.cam_serv.dist_coefs, (w,h), flags = cv2.CALIB_FIX_INTRINSIC)
        
        self.P1 = np.dot(self.cameraMatrix1, np.hstack((np.identity(3),np.zeros((3,1)))))  #Projection Matrix for client cam
        self.P2 = np.dot(self.cameraMatrix2, np.hstack((self.R,self.T))) #Projection matrix for server cam
        print("\nRMS:", self.retval)     
        
        self.stereo_calib_params =  self.retval, self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2, self.R, self.T, self.E, self.F, self.P1, self.P2
        
        fname = os.path.join(os.path.dirname(os.getcwd()), 'stereo_camera_calib_params.pkl')
        
        with open(fname, 'wb') as f:
            pickle.dump(self.stereo_calib_params, f)    
        
        os.chdir(master_dir)
    
    def stereo_rectify(self):
        """
        DEPRECIATED
        This function only works when the image has resolution (1280, 720) """
        
#        self.T[0,0] = -100
        self.rectif = cv2.stereoRectify(self.cameraMatrix1, self.distCoeffs1, self.cameraMatrix2, self.distCoeffs2,
                                        (1280, 720), 
                                        self.R, self.T,
                                        flags = cv2.CALIB_ZERO_DISPARITY,
                                        alpha = 1, newImageSize=(0,0))
                                        
        self.R1, self.R2, self.P1, self.P2, self.Q, _o, _oo = self.rectif
        
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cameraMatrix1, self.distCoeffs1, self.R1, self.P1, (1280, 720), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cameraMatrix2, self.distCoeffs2, self.R2, self.P2, (1280, 720), cv2.CV_32FC1)


        
    def triangulate(self, points1, points2):
        """NOT WORKING YET"""
        
        
        z = cv2.triangulatePoints(self.P1, self.P2, points1, points2)  
        
        z = (z / z[-1]).T
        z = z[:,:3]
        return z

    
    def test_triangulate(self, img1, img2):
        
        square_size = 24.5

        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        
        
        self.obj_points = []
        self.img_points_client = []
        self.img_points_server = []        
        h, w = 0, 0     

        
        img_client = img1
        img_server = img2  
        
        h, w = img_client.shape[:2]
        
        found_client, corners_client = cv2.findChessboardCorners(img_client, pattern_size)
        found_server, corners_server = cv2.findChessboardCorners(img_server, pattern_size)
        
        if found_client and found_server:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)                
            cv2.cornerSubPix(img_client, corners_client, (5, 5), (-1, -1), term)
            cv2.cornerSubPix(img_server, corners_server, (5, 5), (-1, -1), term)
            
        if not (found_client and found_server):
            print('chessboard not found in both images')
            return None
              
        print('ok')
        return corners_client, corners_server     
        
    def triangulate_all_get_PL(self, client_markers = None, server_markers = None):
        """Get points from both videos and triangulate to 3d. Return path length measure.
        I should break this into smaller functions.
        This function is for code testing and will be very slow on the RPi. Use on desktop"""
        
        if client_markers != None:
            
            all_markers1 = client_markers
            all_markers2 = server_markers
            
            pos1 = np.array([i[1] for i in all_markers1])
            pos2 = np.array([i[1] for i in all_markers2])
        
        else:
        
            all_markers1 = self.cam_client.get_ir_markers(display = True)
       
            all_markers2 = self.cam_serv.get_ir_markers(display = True)
    
#        all_markers1 = proc.get_ir_markers_parallel()
#       all_markers2 = proc2.get_ir_markers_parallel()
           
            pos1 = np.array([i[0] for i in all_markers1])
            pos2 = np.array([i[0] for i in all_markers2])
        
        pos1_und = self.cam_client.undistort_points(np.expand_dims(pos1, 0)).squeeze()
        pos2_und = self.cam_serv.undistort_points(np.expand_dims(pos2, 0)).squeeze()
    
        pos3d = self.triangulate(pos1_und.T, pos2_und.T)
        
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=3)
        
        dt = 1/60
        transition_M = np.array([[1, 0, 0, dt, 0, 0],
                              [0, 1, 0, 0, dt, 0],
                              [0, 0, 1, 0, 0, dt],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
                  
        observation_M = np.array([[1,0,0,0,0,0], 
                                  [0,1,0,0,0,0], 
                                  [0,0,1,0,0,0]])
    
        measurements = np.ma.masked_invalid(pos3d)  
    
    
        initcovariance=100*np.eye(6)
        transistionCov=0.5*np.eye(6)
        observationCov=2*np.eye(3)
                 
                            
        kf = KalmanFilter(transition_matrices = transition_M,  observation_matrices = observation_M, initial_state_covariance=initcovariance, transition_covariance=transistionCov,
            observation_covariance=observationCov)    

        (filtered_state_means, filtered_state_covariances) = kf.smooth(measurements)  
        
        first_point = np.where(np.isfinite(measurements.data[:,0]))[0][0]
        last_point = np.where(np.isfinite(measurements.data[:,0]))[0][-1]
        
       
        #Index of first IR marker point
        filtered_state_means2 = filtered_state_means[first_point:last_point]    
        pos3d2 = pos3d[first_point:last_point]
#        pdb.set_trace()
             
        
    
        plt.plot(pos3d2[:,0], 'ro')
        plt.plot(filtered_state_means2[:,0],'r-')
        plt.plot(pos3d2[:,1], 'bo')
        plt.plot(filtered_state_means2[:,1],'b-')
        plt.plot(pos3d2[:,2], 'go')
        plt.plot(filtered_state_means2[:,2],'g-')
    
        
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], color = 'r')
        ax.plot(filtered_state_means[:,0], filtered_state_means[:,1], filtered_state_means[:,2], color = 'b')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        ax.set_xlim([-500,500])
        ax.set_ylim([-500,500])
        ax.set_zlim([0, 2000])
        
        distance = np.sum(np.sqrt(np.sum(np.square(np.diff(filtered_state_means2[:,:3], axis = 0)), axis = 1))) #distance travelled in mm
        print("Distance: {}".format(distance))        
        plt.show()
        return distance, filtered_state_means[:,:3]       
                
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
    """Run a standard server program"""
    
    myCam = posturalCam() #Create a camera class
    myCam.TCP_server_start()

def cam_client(t):
    """Run a standard client program"""
    
    myCam = posturalCam()    
    myCam.TCP_client_start()
    myCam.TCP_client_start_UDP(t, 'testIR.h264') 
    
#    t0 = time.time()
#    points = myCam.TCP_client_request_IRPoints()
#    t1 = time.time()
#    print("TIME: {}".format(t1-t0))
    
    myCam.TCP_client_request_video()   
    myCam.TCP_client_close()
    
    
    
    
    
    proc = posturalProc(v_fname = 'testIR.h264', kind = 'client')
    proc.check_v_framerate()
#    proc2 = posturalProc(v_fname = 'testIR_server.h264', kind = 'server')    
    

#    stereo = stereo_process(proc, proc2)
    
#    PL, points2 = stereo.triangulate_all_get_PL(points[0], points[1])
    
    

#    pdb.set_trace()
    #Ana


def cam_get_x_vids(x):
    """x is the number of videos to get
    
    DEPRECIATED"""
    
    fnames = ['test_video_{}.h264'.format(i) for i in range(x)]
    
    myCam = posturalCam()    
    
    
    for f in fnames:
        myCam.TCP_client_start()
        myCam.TCP_client_start_UDP(15, f) 
        myCam.TCP_client_request_video()      
        myCam.TCP_client_close()
    
def main(t):
    
   # Check who we are
    global networked
    networked = True #If True will run networked protocol. First checks IP address. If Server run server program. If client run client program
     
    
#    pdb.set_trace()
    if networked:
        ip_addr = subprocess.check_output(['hostname', '-I']).decode().strip(" \n") #Get IP address
        
        if ip_addr == '192.168.0.2':
            mode = 'client'     
#            print("I am a {}".format(mode))       
           
            cam_client(t)
#            cam_get_x_vids(1) #Start the client camera 
            
            
        elif ip_addr == '192.168.0.3':
            mode = 'server'
            print("I am a {}".format(mode))
            cam_server()
   
    

def check_synchrony():
    """Test the synchrony between the two RPi's. 
    Take two videos of a timer (a fast clock) and name them 'test_video_0.h264' and 'test_video_server_0.h264'
    This function will play the frames back frame by frame so you can assess the camera synchrony"""
    
    
    
    fnames = ['test_video_{}.h264'.format(i) for i in range(1)]
    
    for video_fname in fnames:
        split_name = video_fname.split('.')
        video_server_fname = '{}_server.{}'.format(split_name[0], split_name[1])
       
        server_cam = posturalProc(0, v_fname = video_server_fname)
        client_cam = posturalProc(0, v_fname = video_fname)
        
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
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
  


def test_stereo_calibration():
    """Test the calibration of the stereo cameras.
    Requires two videos names as in the function. 
    This will grab images of a chessboard and reconstruct it in 3D space. Then check the distance between points and plot a histogram"""
    
    proc = posturalProc(v_fname = 'calib_vid_0.h264', kind = 'client')
    proc2 = posturalProc(v_fname = 'calib_vid_0_server.h264', kind = 'server')    
    

    stereo = stereo_process(proc, proc2)

    img1 = cv2.imread('calibration\\stereo\\client\\calib_img_client_12.tiff',0) #the zero argument makes grayscale (wont work otherwise)
    img2 = cv2.imread('calibration\\stereo\\server\\calib_img_serv_12.tiff',0)
    
    img_p1, img_p2 = stereo.test_triangulate(img1, img2)  
    
    
    img_p1 = proc.undistort_points(img_p1) #Remove extra dimension    
    img_p2 = proc.undistort_points(img_p2) #Remove extra dimension
    plt.figure()
    plt.imshow(img1, cmap = 'gray')
    
    img_p1_o, img_p2_o = cv2.correctMatches(stereo.F, img_p1.transpose(1,0,2), img_p2.transpose(1,0,2)) #Must be in format 1XNX2
#    print(img_p1.transpose(1,0,2).shape)
    img_p1, img_p2 = img_p1.squeeze(), img_p2.squeeze()
    img_p1_o, img_p2_o = img_p1_o.squeeze(), img_p2_o.squeeze()
    plt.figure()
    plt.scatter(img_p1[:,0], img_p1[:,1], color = 'b', s = 100)
    plt.scatter(img_p2[:,0], img_p2[:,1], color = 'r', s = 100)
    plt.scatter(img_p1[:,0], img_p1[:,1], color = 'g')
    plt.scatter(img_p2[:,0], img_p2[:,1], color = 'y')
    
    P1 = np.dot(stereo.cameraMatrix1, np.hstack((np.identity(3),np.zeros((3,1))))) 
    P2 = np.dot(stereo.cameraMatrix2, np.hstack((stereo.R,stereo.T)))
    
    
  
    z = cv2.triangulatePoints(P1, P2, img_p1_o.T, img_p2_o.T)  
    z = (z / z[-1]).T
    z = z[:,:3]
    

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z[:,0], z[:,1], z[:,2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    
    
    p_diff = np.sqrt(np.square(np.diff(z, axis = 0)).sum(1))        
    p_diff = np.append(p_diff, -999)
    
    
    plt.figure()
    sns.distplot(p_diff.reshape((6,9))[:,:-1].flatten())
    
#proc.get_calibration_frames()
#proc.camera_calibration()
#proc.play_video()
#proc.get_markers()
#proc.check_camera_calibration()
    
def calibration_protocol():
    """A function to calibrate all the cameras"""
    
    proc = posturalProc(v_fname = 'testIR.h264', kind = 'client')
    proc2 = posturalProc(v_fname = 'testIR_server.h264', kind = 'server')    
    
#    proc.get_calibration_frames()
#    proc2.get_calibration_frames()
#    
#    proc.camera_calibration()    
#    proc2.camera_calibration()
    
    stereo = stereo_process(proc, proc2)
#    stereo.get_calibration_frames() #You will need to then place the calibration images into the correct folders (Set this up to do automatically)
    stereo.stereo_calibrate()
    print(stereo.R)
    print(stereo.T)
    
if __name__ == '__main__':

    proc = posturalProc(v_fname = 'testIR.h264', kind = 'client')
    proc.load_video()
    
    
    max_area = 80
    all_rad = []
    all_markers = []
    
    i = 0
    while True:
        print(i)
        ret, frame = proc.cap.read()
        if ret == False:
            break
        
        ir = proc.find_ir_markers(frame, it = i, tval = 150)
        all_markers.append(ir)
        if ir != None:
            
#            rads = [mark['radius'] > max_area for mark in ir]
#            
#            if True in rads:
#                ir = proc.find_ir_markers(frame, it = i, tval = 220)
#
            [all_rad.append(mark['radius']) for mark in ir]
        
        
#        f = plt.figure() 
#        plt.imshow(frame[:,:,::-1])
#        print(ir)
#        if ir != None:
#            for mark in ir:
#                
#                circle1 = plt.Circle(mark['pos'], mark['radius'] , color='b', fill = False, linewidth = 2)
#                plt.gca().add_artist(circle1)
#                        
#        
#            plt.title("N markers: {}".format(len(ir)))
#       
#        plt.savefig("frame_{}.png".format(i))
#        plt.show()
#        plt.close()
        
        
        i += 1
        
    plt.figure()
    plt.plot(all_rad)
    plt.show()
    
    all_pos = proc.markers2numpy(all_markers)
        
#        if ir != None:
#            ir = proc.order_ir_markers(ir)
    
#    main(t = 10) #Record video    
 
 
 
#    process()
#    check_synchrony()    
#    calibration_protocol() #Calibrate the cameras
#    test_stereo_calibration() 
#    
#    proc = posturalProc(v_fname = 'testIR.h264', kind = 'client')
#    proc2 = posturalProc(v_fname = 'testIR_server.h264', kind = 'server')    
##    
###    p2 = proc2.get_ir_markers_parallel()
###    p1 = proc.get_ir_markers_parallel()   
##    
##
###    pos1 = np.array([i[0] for i in p1])
##   
#    stereo = stereo_process(proc, proc2)
#    stereo.get_calibration_frames()
#    
##    PL, points = stereo.triangulate_all_get_PL()
    
#    np.save("points.npy", points)
    
#    points = np.load("points.npy")


#    import mpl_toolkits.mplot3d.axes3d as p3
#    import matplotlib.animation as animation
#
#
#
#    def update_plot(i):
#        print(i)
#        ax.plot(points[:i,0], points[:i,1], points[:i,2], color = 'k')
##        ax.plot(filtered_state_means[:,0], filtered_state_means[:,1], filtered_state_means[:,2], color = 'b')
#
#    # Attaching 3D axis to the figure
#    fig = plt.figure()
#    ax = p3.Axes3D(fig)
#
# 
#    # Setting the axes properties
#    
#    ax.set_xlabel('X')
#    
#   
#    ax.set_ylabel('Y')
#    
#
#    ax.set_zlabel('Z')
#    
#    ax.set_title('3D Test')
#    
#    # Creating the Animation object
#    line_ani = animation.FuncAnimation(fig, update_plot, len(points), 
#                                       interval=10, blit=False)
#    
#    plt.show()
