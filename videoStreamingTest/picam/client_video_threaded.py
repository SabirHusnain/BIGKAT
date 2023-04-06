# -*- coding: utf-8 -*-
"""

"""

import socket
import time
import cv2
import io
import struct
import numpy as np
import threading

class PiVideoClient:
    """A object to recieve video from the surver and provide methods to access the most recent frame"""
    def __init__(self):
        
        self.data = None
        # initialize the frame and the variable used to indicate
	  # if the thread should be stopped
        self.frame = None
        self.stopped = False
        
    def connect(self):
        """Attempt to connect tot the surver"""
        
        # Connect a client socket to my_server:8000 (change my_server to the
        # hostname of your server)
        self.client_socket = socket.socket()
        self.client_socket.connect((socket.gethostname(), 8000))
        self.connection = self.client_socket.makefile('rb')
        
    def start(self):
        """start the thread to read frames from the video stream"""
        threading.Thread(target=self.update, args=()).start()
        
        return self
        
    def update(self):
        
        
        while True:
            image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
            
            if not image_len:
                break        
        
            self.data = self.connection.read(image_len) #Read frame from the network stream
            
            if self.stopped:
                
                self.connection.close()
                return

    def read(self):
        """Read and decode a frame"""
        
        if self.data != None:
            frame = np.fromstring(self.data, dtype=np.uint8) 
            
            image = cv2.imdecode(frame, 1)
             
            return image
 
    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True

client = PiVideoClient()

client.connect()

client.start()

t0 = time.time()

while True:
    t0_0 = time.time()
    frame = client.read()

  
    if frame != None:
#        print(image.shape)
        cv2.imshow("WEB", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    t1 = time.time()
    
    fps = 1/ (t1 - t0_0)
    
    print("FPS: {}".format(fps))
    if (t1-t0) > 30:
        print("stop")
        client.stop()
        break
  