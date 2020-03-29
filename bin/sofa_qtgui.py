import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import random
import sofa_config

cfg = None;

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'SOFA - https://github.com/cyliustack/sofa'
        self.width = 640
        self.height = 400
        self.initUI()
        self.setWindowTitle('SOFA GUI')  
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        m = PlotCanvas(self, width=5, height=4)
        m.move(0,0)

        button = QPushButton('Process', self)
        button.setToolTip('To process trace data')
        button.move(500,100)
        button.resize(120,40)
        self.show()

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()


    def plot(self):
        global cfg
        print(cfg)
        data = pd.read_csv('./sofalog/mpstat.csv')
        #data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.set_title('mpstat dynamic changes')
        ax.plot(data['timestamp'], data['duration'], 'r-')
        self.draw()

def sofa_qtgui(cfg_in):
    app = QApplication(sys.argv)
    ex = App()
    global cfg
    cfg = cfg_in
    sys.exit(app.exec_())
