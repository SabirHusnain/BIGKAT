import sys
from PyQt5 import QtGui, QtCore, QtWidgets
#from PySide import QtGui, QtCore
#import main as game


class Example(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """The Postural Rig GUI layout"""

        # Frames

#        topleft = QtGui.QFrame(self)
#        topleft.setFrameShape(QtGui.QFrame.StyledPanel)

        #Labels and buttons
        Forename_label = QtWidgets.QLabel('First Name(s)')
        self.Forename_edit = QtWidgets.QLineEdit()

        Surname_label = QtWidgets.QLabel('Surname')
        self.Surname_edit = QtWidgets.QLineEdit()

        ID = QtWidgets.QLabel("ID")
        self.ID_edit = QtWidgets.QLineEdit()

        # DOB should be a date format or a box to select the date
        DOB = QtWidgets.QLabel('DOB')
        self.DOB_edit = QtWidgets.QDateEdit()

        # Gender is a box selections
        Gender = QtWidgets.QLabel('Gender')
        self.Gender_edit = QtWidgets.QComboBox()
        self.Gender_edit.addItem("Male")
        self.Gender_edit.addItem("Female")

        self.DOB_w = QtWidgets.QCalendarWidget()

        Record_time_label = QtWidgets.QLabel('Record Time')
        self.Record_time = QtWidgets.QDoubleSpinBox()
        self.Record_time.setValue(10)
        self.Record_time.setMinimum(0.000001)
        # Buttons

        # New Participant buttong

        New_part = QtWidgets.QPushButton("New Participant")
        # Start recording button
        Start = QtWidgets.QPushButton("Start Recording")

        # Preview Button
        Preview = QtWidgets.QPushButton("Preview Cameras")

        # Process Data Button
        Process = QtWidgets.QPushButton("Process Data")

        # Exit button
        Exit = QtWidgets.QPushButton("Shut down")

        # Progress bar
        self.progress = QtWidgets.QProgressBar(self)

        ##--------------------------##
        ##--------------------------##
        ##-----------Actions--------##
        ##--------------------------##
        ##--------------------------##
        ##--------------------------##

        Exit.clicked.connect(QtCore.QCoreApplication.instance().quit)

        # Start button
        # This will happen in order
        Start.clicked.connect(self.get_demographics)
        Start.clicked.connect(self.action_show_rec_progress)


#        self.timer = QtCore.QBasicTimer()
#        self.step = 0

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(New_part, 0, 0, 1, 1)

        grid.addWidget(Forename_label, 1, 0, 1, 1)
        grid.addWidget(self.Forename_edit, 1, 1, 1, 1)

        grid.addWidget(Surname_label, 1, 2, 1, 1)
        grid.addWidget(self.Surname_edit, 1, 3, 1, 1)

        grid.addWidget(Gender, 1, 4, 1, 1)
        grid.addWidget(self.Gender_edit, 1, 5, 1, 1)

        grid.addWidget(ID, 2, 0, 1, 1)
        grid.addWidget(self.ID_edit, 2, 1, 1, 1)

        grid.addWidget(DOB, 2, 2, 1, 1)
        grid.addWidget(self.DOB_edit, 2, 3, 1, 1)

#        grid.addWidget(DOB_w, 2, 5, 1, 1)

        blank = QtWidgets.QLabel('')  # Add a blank row to the GUI

        grid.addWidget(blank,   4, 0, 1, 1)
        grid.addWidget(Start,   5, 0, 1, 1)
        grid.addWidget(Record_time_label,   5, 1, 1, 1)
        grid.addWidget(self.Record_time,   5, 2, 1, 1)
        grid.addWidget(Preview, 5, 5, 1, 1)
        grid.addWidget(Process, 5, 3, 1, 1)
        grid.addWidget(Exit,    6, 5, 1, 1)

        grid.addWidget(self.progress, 6, 0, 1, 4)

        self.setLayout(grid)

        self.setGeometry(500, 300, 500, 100)
        self.setWindowTitle('Postural Sway Assessment Tool')
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Cleanlooks'))
        self.show()

    def get_demographics(self):
        """Retrieve the participant demographics data input into the main GUI"""

        # Empty dictionary of demographic information for participants
        self.participant_demographics = {}
        self.participant_demographics['Forename'] = self.Forename_edit.text()
        self.participant_demographics['Surname'] = self.Surname_edit.text()
        self.participant_demographics['ID'] = self.ID_edit.text()
        self.participant_demographics['Gender'] = self.Gender_edit.currentText(
        )

        self.participant_demographics['DOB'] = self.DOB_edit.date().toPython()

        self.participant_demographics['record_time'] = self.Record_time.value()

        print(self.participant_demographics)

    def action_show_rec_progress(self):
        self.completed = 0

        while self.completed < 100:
            self.completed += 0.00001
            self.progress.setValue(self.completed)

    def timerEvent(self, e):

        if self.step >= 100:

            self.timer.stop()
            self.btn.setText('Finished')
            return

        self.step = self.step + 1
        self.pbar.setValue(self.step)


def main():
    app = QtWidgets.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
