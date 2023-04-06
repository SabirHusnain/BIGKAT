# -*- coding: utf-8 -*-
"""

"""
import sys
import socket
import time
import cv2
import io
import struct
import numpy as np
from PyQt4 import QtCore, QtGui  #Note that some code will need to be changed if this is used (Signal has different call signiture)

Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot


class MainWindow(QtGui.QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__(None)  
        self.initUI()
        
    def initUI(self):
        
     
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))
        
        self.central_widget = QtGui.QWidget()        
        grid = QtGui.QGridLayout()
        grid.setSpacing(10)      
        
        self.label1 = QtGui.QLabel(self)
        grid.addWidget(self.label1, 1, 1, 1, 1)
        
        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)
        
        self.connect_client()
        
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.video)
#        self._timer.setInterval(1)         
        self._timer.start(1000/100)
        
        self.show()  
    
    def connect_client(self):
        self.client_socket = socket.socket()
        self.client_socket.connect((socket.gethostname(), 8000))

        self.connection = self.client_socket.makefile('rb')
        
        print("CONNECTION MADE")
        
    def say_hi(self):
        
        print("HI")
        
    def video(self):
        ###Camera feeds
      
        image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
        
    
        frame = self.connection.read(image_len)
        
        # Construct a numpy array from the stream
    
        data = np.fromstring(frame, dtype=np.uint8)
    
        # "Decode" the image from the array, preserving colour
        f1 = cv2.imdecode(data, 1)
        
        
        
        image_scale_factor = 1

         
        
        image_size = (int(f1.shape[1] * image_scale_factor), int(f1.shape[0] * image_scale_factor))
        
      
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        f1 = cv2.resize(f1, image_size)
       
        
        image1 = QtGui.QImage(f1, f1.shape[1], f1.shape[0], 
                       f1.strides[0], QtGui.QImage.Format_RGB888)

                       
        self.label1.setPixmap(QtGui.QPixmap.fromImage(image1))
        
      
        
        
if __name__ == '__main__':
    
    app = QtGui.QApplication(sys.argv)

    ex = MainWindow()
    
    sys.exit(app.exec_())
    
    