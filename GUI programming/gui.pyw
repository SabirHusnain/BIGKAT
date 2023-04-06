# -*- coding: utf-8 -*-
"""

"""

#!/usr/bin/env python
import sys
from PyQt4 import QtGui, QtCore
import main as game

class Example(QtGui.QWidget):
    
    def __init__(self):
        super(Example, self).__init__()
        
        self.initUI()
        
    def initUI(self):
        
        Forename = QtGui.QLabel('First Name')
        Surname = QtGui.QLabel('Surname')
        
        DOB = QtGui.QLabel('DOB')
        RT = QtGui.QLabel('Gender')
        Exit = QtGui.QPushButton("Exit")
        Exit.clicked.connect(QtCore.QCoreApplication.instance().quit)
        
        start_game = QtGui.QPushButton("Start Game")
        start_game.clicked.connect(game.run) 

        NameEdit = QtGui.QLineEdit()
        NameEdit2 = QtGui.QLineEdit()
        DOBEdit = QtGui.QLineEdit()
        RTEdit = QtGui.QTextEdit()
        
      
        combo = QtGui.QComboBox(self)
        combo.addItem("Male")
        combo.addItem("Female")       
        
        self.pbar = QtGui.QProgressBar(self)
        self.btn = QtGui.QPushButton('Start', self)
        self.btn.clicked.connect(self.doAction)

        self.timer = QtCore.QBasicTimer()
        self.step = 0
        

        grid = QtGui.QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(Forename, 1, 0, 1, 1)
        grid.addWidget(NameEdit, 1, 1, 1, 1)
        
        grid.addWidget(Surname, 1, 2, 1, 1 )
        grid.addWidget(NameEdit2, 1, 3, 1, 1)
        

        grid.addWidget(DOB, 3, 0, 1, 1)
        grid.addWidget(DOBEdit, 3, 1, 1, 1)

        grid.addWidget(RT, 3, 2, 1, 1)
        grid.addWidget(combo, 3, 3, 1, 1)
        
        grid.addWidget(Exit, 4, 3, 1, 1)
        
        grid.addWidget(self.pbar, 4, 1, 1, 2)
        grid.addWidget(self.btn, 4, 0, 1, 1)
        
        grid.addWidget(start_game, 5,0,1,1)
        
        self.setLayout(grid) 
        
        self.setGeometry(300, 300, 350, 100)
        self.setWindowTitle('PyHeadTrack')    
        self.show()       
        
        
   
        
    def timerEvent(self, e):
  
        if self.step >= 100:
        
            self.timer.stop()
            self.btn.setText('Finished')
            return
            
        self.step = self.step + 1
        self.pbar.setValue(self.step)

    def doAction(self):
      
        if self.timer.isActive():
            self.timer.stop()
            self.btn.setText('Start')
            
        else:
            self.timer.start(100, self)
            self.btn.setText('Stop')
        
def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()