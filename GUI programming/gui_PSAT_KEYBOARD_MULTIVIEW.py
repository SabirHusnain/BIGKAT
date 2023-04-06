# -*- coding: utf-8 -*-
"""

"""

from PyQt5 import QtGui, QtCore, QtWidgets
Signal = QtCore.pyqtSignal
Slot = QtCore.pyqtSlot

import cv2
import sys


def load_demo_images():
    """For demo purposes we can load 2 images"""
    frame1 = cv2.imread('calib_img_0.tiff')
    frame2 = cv2.imread('calib_img_1.tiff')
    
    return frame1, frame2
 
class virtualKeyboard(QtWidgets.QDialog):
    """"A virtual Keyboard"""
    
    def __init__(self, parent = None):
        
        super(virtualKeyboard, self).__init__(parent)    

        
        self.create_keys()
        
        
        
    def create_keys(self):
  
        self.lower_key_labels  =  [['1','2','3','4','5','6','7','8','9','0'],
                                   ['q','w','e','r','t','y','u','i','o','p'],
                                   ['a','s','d','f','g','h','j','k','l',''],
                                   ['CAP','z','x','c','v','b','n','m','<-',''],
                                   ['','Spacebar','']]       
        
        self.upper_key_labels  =  [['1','2','3','4','5','6','7','8','9','0'],
                                   ['Q','W','E','R','T','Y','U','I','O','P'],
                                   ['A','S','D','F','G','H','J','K','L',''],
                                   ['CAP','Z','X','C','V','B','N','M','<-',''],
                                   ['','Spacebar','']] 
        
        self.alt_key_labels  =  [['1','2','3','4','5','6','7','8','9','0'],
                                   ['@','£','&','_','(',')',':',';','"',''],
                                   ['','!','#','=','*','/','+','-','*',''],
                                   ['',',','.','','','','','','<-',''],
                                   ['','Spacebar','']] 
                                   
                                   
        self.key_buttons = [[Text_Button(k) for k in row] for row in self.lower_key_labels] #Create all the buttons


                      
        master_grid = QtWidgets.QGridLayout() #Master layout
        
        row_grids = [QtWidgets.QGridLayout() for r in range(len(self.lower_key_labels))] #Row Layouts
    
        
        ##Layout each row
        
        for key in range(len(self.key_buttons[0])):
            row_grids[0].addWidget(self.key_buttons[0][key], 0, key) #Add the first row of buttons to the first grid
            
        for key in range(len(self.key_buttons[1])):
            row_grids[1].addWidget(self.key_buttons[1][key], 0, key) #Add the second row of buttons to the first grid
        
        for key in range(len(self.key_buttons[2])):
            row_grids[2].addWidget(self.key_buttons[2][key], 0, key) #Add the third row of buttons to the first grid
         
        for key in range(len(self.key_buttons[3])):
            row_grids[3].addWidget(self.key_buttons[3][key], 0, key) #Add the third row of buttons to the first grid
            
        row_grids[4].addWidget(self.key_buttons[4][0], 0, 0, 1,3)
        row_grids[4].addWidget(self.key_buttons[4][1], 0, 3, 1,5)
        row_grids[4].addWidget(self.key_buttons[4][2], 0, 8, 1,4)


        master_grid.addLayout(row_grids[0], 0, 0)
        master_grid.addLayout(row_grids[1], 1, 0)
        master_grid.addLayout(row_grids[2], 2, 0)
        master_grid.addLayout(row_grids[3], 3, 0)
        master_grid.addLayout(row_grids[4], 4, 0)
        
        self.setLayout(master_grid)           
     
#        self.set_case('lower')
       
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100%);")
        
#        self.hide() #Make the keyboard invisible
    
    def connect_to_buttons(self, obj):
        """connect all the buttons to a suitable Qt object"""
        
        [[k.pressed.connect(obj) for k in row] for row in self.key_buttons] #Connect all the buttons to the object

    def set_case(self, case = 'lower'):
        """Toggle the letter case"""
        
        if case == 'lower':
            
            for row in range(len(self.key_buttons)):
                for k in range(len(self.key_buttons[row])):
                    
                    self.key_buttons[row][k].change_text(self.lower_key_labels[row][k]) 
            
        elif case == 'upper':
            
            for row in range(len(self.key_buttons)):
                for k in range(len(self.key_buttons[row])):
                    
                    self.key_buttons[row][k].change_text(self.upper_key_labels[row][k])      
                    
            
        elif case == 'alt':
            
            for row in range(len(self.key_buttons)):
                for k in range(len(self.key_buttons[row])):
                    
                    self.key_buttons[row][k].change_text(self.alt_key_labels[row][k])    
        
   
class Text_Button(QtWidgets.QToolButton):
    """A button class for the keyboard"""
    
    pressed = Signal(str)

    def __init__(self, text, parent=None):
        super(Text_Button, self).__init__(parent)

        self.text = text
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Preferred)
        self.setText(text)
        
        
        
        self.setStyleSheet("""background-color: rgba(87, 87, 87, 70%); 
                                    border: 0px;
                                    color: rgb(255,255,255);""")

    def sizeHint(self):
        size = super(Text_Button, self).sizeHint()
        size.setHeight(size.height() + 20)
        size.setWidth(max(size.width(), size.height()))
        return size
      
    def mousePressEvent(self, e):
        
        if self.text != '':
       
            self.pressed.emit(self.text)   

                
            self.setStyleSheet("""background-color: yellow; 
                                        border: 0px;
                                        color: rgb(255,255,255);""")
            print("Key: {}".format(self.text))
        
    def mouseReleaseEvent(self, e):
        if self.text != '':
            self.setStyleSheet("""background-color: rgba(87, 87, 87, 70%); 
                                        border: 0px;
                                        color: rgb(255,255,255);""")
                                        
    def change_text(self, text):
        self.text = text
        self.setText(text)
           

       
class MyLineEdit(QtWidgets.QLineEdit):
    """A child of the QLineEdit that emits a signal when it is pressed"""
 
    pressed = Signal() #Signal to emit when mousepress focused    
    end_focus = Signal()    
    
    def __init__(self, parent = None):
        super(MyLineEdit, self).__init__(parent)
        
        self.text = ''
        
        self.in_focus = None
       
    def mousePressEvent(self, e):
        
        self.pressed.emit() #Emit a signal when the key is pressed
        print("Key Pressed")
    
    def focusInEvent(self, e):
        
        QtWidgets.QLineEdit.focusInEvent(self, QtGui.QFocusEvent(QtCore.QEvent.FocusIn)) #Call the default In focus event
        
        self.in_focus = True
        
        print("IN")
    
    def focusOutEvent(self, e):
        
        QtWidgets.QLineEdit.focusOutEvent(self, QtGui.QFocusEvent(QtCore.QEvent.FocusOut)) #Call the default Outfocus event
           
        self.in_focus = False
        self.end_focus.emit() #Emit signal that focus was lost
        print("OUT")
        
    @Slot(str)    
    def recieve_input(self, inp):

        if self.in_focus:        
        
            print("Recieved key {}".format(inp))
            
            if inp =='Spacebar':
                self.text += ' '
            elif inp == '<-':
                self.text = self.text[:-1]
            else:
                self.text += inp
                
                
            self.setText(self.text)
        
        
class QLED(QtWidgets.QWidget):

    def __init__(self, parent = None):   
        super(QLED, self).__init__(parent)
        
        palette = QtCore.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)
        
        
        
class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__(None)  
        self.initUI()
        
    def initUI(self):
        
        self.setGeometry(400, 240, 800, 480)
        self.setWindowTitle('Postural Sway Assessment Tool')    
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Cleanlooks'))
        
        
        self.create_widgets()
        
        
        exitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Shutdown', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Shutdown PSAT')
        exitAction.triggered.connect(QtWidgets.qApp.quit)
        
        
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        
        self.show()  
    
    def create_widgets(self):
        
        self.central_widget = QtWidgets.QWidget()
        
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)      
           
        grid.addWidget(self.create_demographics_group(), 0,0, 2,1)

        grid.addWidget(self.create_camera_group(), 0,0, 1,1)
        grid.addWidget(self.create_recording_group(), 0, 0, 1, 1)
        
        #        
       
        
        #Exit button
#        Shutdown = QtWidgets.QPushButton("Shut down")
#        Shutdown.clicked.connect(self.shutdown_event) #New style shutdown function
#        
#        grid.addWidget(Shutdown, 4, 4, 1, 2)        
#        Preview_btn = QtWidgets.QPushButton("Start Camera Preview")
#        grid.addWidget(Preview_btn, 4, 0, 1, 1)
 
#        QtCore.QObject.connect(Shutdown, QtCore.SIGNAL('clicked()'), self.shutdown_event) #Call shutdown function Old style
        grid.addWidget(self.create_keyboard(), 2, 0, 4, 1) 
        
        self.show_window_1()      
        
        self.central_widget.setLayout(grid)
        self.setCentralWidget(self.central_widget)
        
        #Demographics box

    def shutdown_event(self):
        """Shut down button action. Opens a question box to ask if you want to power down"""
        shutdown_msg = "Are you sure you want to power down PSAT?"
        
        reply = QtWidgets.QMessageBox.question(self, "Shutdown", shutdown_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        
        if reply == QtWidgets.QMessageBox.Yes:
            QtCore.QCoreApplication.instance().quit() #Quit Application. This will close down the RPis in future
        else:
            print("Not closing")
    

    def show_keyboard(self):
        
        print("SHOW")
        self.keyboard.show()
        
    def hide_keyboard(self):
        
        self.keyboard.hide()
        print("HIDE")
        
    def load_file(self):
        
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 
                '/home')
                
        print(fname)
        
    def show_window_1(self):
        """Show the demographics window"""
        self.demographicsBox.show()
        self.camera_box.hide()
        self.recording_box.hide()
        self.show_keyboard()
        
        
    
    def show_window_2(self):        
        """Show the check camera window"""
        
        self.demographicsBox.hide()
        self.camera_box.show()
        self.recording_box.hide()
        self.hide_keyboard()
 
    
    def show_window_3(self):        
        """Show the recording window"""
        pass
    
    def show_window_4(self):
        """Show data analysis window"""
        
        pass
      
               
    def create_demographics_group(self):
        
 
        self.demographicsBox = QtWidgets.QGroupBox("Demographics")       
#        self.demographicsBox.setStyleSheet("QGroupBox{background-image: url('calib_img_0.tiff');}") #Add a background image
        #Labels and buttons
        Forename_label = QtWidgets.QLabel('First Name(s)')     
        self.Forename_edit = MyLineEdit(self)
#        self.Forename_edit.pressed.connect(self.show_keyboard)    
#        self.Forename_edit.end_focus.connect(self.hide_keyboard)  
        
        Surname_label = QtWidgets.QLabel('Surname')
        self.Surname_edit = MyLineEdit(self)
#        self.Surname_edit.pressed.connect(self.show_keyboard)    
#        self.Surname_edit.end_focus.connect(self.hide_keyboard)  
        
        Gender = QtWidgets.QLabel('Gender')
        self.Gender_edit = QtWidgets.QComboBox()
        self.Gender_edit.addItem("Male")
        self.Gender_edit.addItem("Female")    
        
        ID = QtWidgets.QLabel("ID")
        self.ID_edit = MyLineEdit(self)
#        self.ID_edit.pressed.connect(self.show_keyboard)    
#        self.ID_edit.end_focus.connect(self.hide_keyboard)
        
        
        #DOB should be a date format or a box to select the date
        DOB = QtWidgets.QLabel('DOB')
        self.DOB_edit =  QtWidgets.QDateEdit()      
        
        
        load_IDs = QtWidgets.QPushButton("Load Participants")
        load_IDs.clicked.connect(self.load_file)
        
        check_ID = QtWidgets.QPushButton("Verify Participant")
        
        
        start_Rec = QtWidgets.QPushButton("Start Recording")
        start_Rec.clicked.connect(self.show_window_2)
        
        
        
        
        
        
        demographics_grid = QtWidgets.QGridLayout()        
        
        demographics_grid.addWidget(Forename_label, 1,0, 1,1)
        demographics_grid.addWidget(self.Forename_edit, 1, 1, 1, 1)
        
        demographics_grid.addWidget(Surname_label, 1, 2, 1, 1)
        demographics_grid.addWidget(self.Surname_edit, 1, 3, 1, 1)
        
        demographics_grid.addWidget(Gender, 1, 4, 1, 1)
        demographics_grid.addWidget(self.Gender_edit, 1, 5, 1, 1)
        
        demographics_grid.addWidget(ID, 2, 0, 1, 1)
        demographics_grid.addWidget(self.ID_edit, 2, 1, 1, 1)
        
        demographics_grid.addWidget(DOB, 2, 2, 1, 1)
        demographics_grid.addWidget(self.DOB_edit, 2, 3, 1, 1)
        
        demographics_grid.addWidget(load_IDs, 3, 0, 1, 1)
        demographics_grid.addWidget(check_ID, 3, 1, 1, 1)
        
        demographics_grid.addWidget(start_Rec, 3, 3, 1, 1)
        
        demographics_grid.setSpacing(10)
        
        self.demographicsBox.setLayout(demographics_grid)
        
        return self.demographicsBox
    
    def create_camera_group(self):
        
        self.camera_box = QtWidgets.QGroupBox("Camera Status") 
        
        box_grid = QtWidgets.QGridLayout()     
#        box_grid.addWidget(sc)
        
        
        
        ###Camera feeds
        image_size = (220, 150)
        label1 = QtWidgets.QLabel(self)
        label2 = QtWidgets.QLabel(self)
        
        f1, f2 =  load_demo_images()
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        f1 = cv2.resize(f1, image_size)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        f2 = cv2.resize(f2, image_size)
        
        image1 = QtGui.QImage(f1, f1.shape[1], f1.shape[0], 
                       f1.strides[0], QtGui.QImage.Format_RGB888)
        
        image2 = QtGui.QImage(f2, f2.shape[1], f2.shape[0], 
                       f2.strides[0], QtGui.QImage.Format_RGB888)
                       
        label1.setPixmap(QtGui.QPixmap.fromImage(image1))
        label2.setPixmap(QtGui.QPixmap.fromImage(image2))
        
        
        #Buttons      
        
        back = QtWidgets.QPushButton("Back")
        back.clicked.connect(self.show_window_1)
        
        start = QtWidgets.QPushButton("Start Recording")
        start.clicked.connect(self.show_window_3)
       
        

        box_grid.addWidget(label1, 0, 0, 1, 2)
        box_grid.addWidget(label2, 0, 2, 1, 2)
        
        box_grid.addWidget(back, 2, 0, 1, 1)
        
        
        box_grid.setSpacing(20) 
        self.camera_box.setLayout(box_grid)
        
        return self.camera_box
        
        
    def create_recording_group(self):
        
        self.recording_box = QtWidgets.QGroupBox("Recording")         
        box_grid = QtWidgets.QGridLayout()
        
        Record_time_label = QtWidgets.QLabel('Record Time')
        Record_time = QtWidgets.QDoubleSpinBox()
        Record_time.setValue(10)
        Record_time.setMinimum(0.000001)
        
        #Start recording button
        Start = QtWidgets.QPushButton("Start Recording")
        
        #Progress bar
        progress = QtWidgets.QProgressBar(self)   
    
        box_grid.addWidget(Start,               0, 2, 1, 2)    
        box_grid.addWidget(Record_time_label,   0, 0, 1, 1)    
        box_grid.addWidget(Record_time,         0, 1, 1, 1)   
        box_grid.addWidget(progress,            1, 0, 1, 4)  
        
        box_grid.setSpacing(1) 
        self.recording_box.setLayout(box_grid)
        
        return self.recording_box


    def create_keyboard(self):
        """Virtual Keyboard"""
        
        self.keyboard = virtualKeyboard()
        
        #NOW CONNECT ALL THE KEYS TO THEIR OUTPUTS
        self.keyboard.connect_to_buttons(self.Forename_edit.recieve_input) #Connect this to the buttons
        self.keyboard.connect_to_buttons(self.Surname_edit.recieve_input) #Connect this to the buttons
        self.keyboard.connect_to_buttons(self.ID_edit.recieve_input) #Connect this to the buttons
     
#         spacebar.pressed.connect(self.Forename_edit.recieve_input)
        return self.keyboard
                
        

if __name__ == '__main__':
    
    app = QtWidgets.QApplication(sys.argv)
#    app.autoSipEnabled() 
    ex = MainWindow()

    sys.exit(app.exec_())
    
#    f1, f2 =  load_demo_images()
