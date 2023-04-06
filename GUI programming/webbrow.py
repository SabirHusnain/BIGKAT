# -*- coding: utf-8 -*-
"""

"""

import sys
from PyQt4 import QtCore, QtGui, QtWebKit
from ui_mainwindow import Ui_MainWindow

class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    # Maintain the list of browser windows so that they do not get garbage
    # collected.
    _window_list = []

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        MainWindow._window_list.append(self)

        self.setupUi(self)
        
        self.lblAddress = QtGui.QLabel("", self.tbAddress)
        self.tbAddress.insertWidget(self.actionGo, self.lblAddress)
        self.addressEdit = QtGui.QLineEdit(self.tbAddress)
        self.tbAddress.insertWidget(self.actionGo, self.addressEdit)
        
        self.addressEdit.setFocusPolicy(QtCore.Qt.StrongFocus)
        
        self.connect(self.addressEdit, QtCore.SIGNAL("returnPressed()"),
                     self.actionGo, QtCore.SLOT("trigger()"))
                     
        self.connect(self.actionBack, QtCore.SIGNAL("triggered()"),
                     self.WebBrowser, QtCore.SLOT("back()"))
        
        self.connect(self.actionForward, QtCore.SIGNAL("triggered()"),
                     self.WebBrowser, QtCore.SLOT("forward()"))
        
        self.connect(self.actionStop, QtCore.SIGNAL("triggered()"),
                     self.WebBrowser, QtCore.SLOT("stop()"))
        
        self.connect(self.actionRefresh, QtCore.SIGNAL("triggered()"),
                     self.WebBrowser, QtCore.SLOT("reload()"))

        self.pb = QtGui.QProgressBar(self.statusBar())
        self.pb.setTextVisible(False)
        self.pb.hide()
        self.statusBar().addPermanentWidget(self.pb)
        self.WebBrowser.load(QtCore.QUrl("http://www.google.com"))

        
    @QtCore.pyqtSignature("")
    def on_actionHome_triggered(self):
        self.WebBrowser.load(QtCore.QUrl("http://www.google.com"))

    def on_WebBrowser_urlChanged(self, url):
        self.addressEdit.setText(url.toString())
        
    def on_WebBrowser_titleChanged(self, title):
        #print 'titleChanged',title.toUtf8()
        self.setWindowTitle(title)

    def on_WebBrowser_loadStarted(self):
        #print 'loadStarted'
        #self.misc.keyboard_show()
        
        self.pb.show()
        self.pb.setRange(0, 100)
        self.pb.setValue(1)

    def on_WebBrowser_loadFinished(self, flag):
        #print 'loadFinished'
        if flag is True:
            self.pb.hide()
            self.statusBar().removeWidget(self.pb)
            
    def on_WebBrowser_loadProgress(self, status):
        self.pb.show()
        self.pb.setRange(0, 100)
        self.pb.setValue(status)

    @QtCore.pyqtSignature("")
    def on_actionGo_triggered(self):
        #print "on_actionGo_triggered"
        
        self.WebBrowser.load(QtCore.QUrl(self.addressEdit.text()))
        self.addressEdit.setText(self.addressEdit.text())

    @QtCore.pyqtSignature("")
    def on_actionHome_triggered(self):
        #print "on_actionHome_triggered"
        self.WebBrowser.load(QtCore.QUrl("http://www.google.com"))
        self.addressEdit.setText("http://www.google.com")


if __name__ == "__main__":
    a = QtGui.QApplication(sys.argv)
    w = MainWindow()

    w.show()
    sys.exit(a.exec_())