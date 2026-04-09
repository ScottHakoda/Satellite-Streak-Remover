import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QProgressBar, QTextEdit
from PyQt6.QtCore import QThread, pyqtSignal

class Worker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)

    def run(self):
        for i in range(100):
            # ... your processing ...
            self.progress.emit(i+1)
            self.log.emit(f"Step {i+1} completed")
        self.log.emit("Done!")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My PyQt App")
        self.setGeometry(100, 100, 800, 600)

        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.button = QPushButton("Select File", self)
        self.button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.button)

        self.label = QLabel("No file selected", self)
        layout.addWidget(self.label)

        # Matplotlib Figure and Canvas
        self.figure = Figure(figsize=(8, 6))
        self.axes = self.figure.subplots(2,2)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        #progress bar
        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)
        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_path:
            self.label.setText(f"Selected: {file_path}")
            self.start_long_task()
            self.plot_four_figures()
        else:
            self.label.setText("No file selected")

    def plot_four_figures(self):
        # Example data for each subplot
        titles = ["Plot 1", "Plot 2", "Plot 3", "Plot 4"]
        for i in range(2):
            for j in range(2):
                ax = self.axes[i, j]
                ax.clear()
                ax.plot([0, 1, 2, 3], [i*2+j, (i*2+j)+1, (i*2+j)+2, (i*2+j)+3])
                ax.set_title(titles[i*2+j])
        self.canvas.draw()

    def start_long_task(self):
        self.worker = Worker()
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.console.append)
        self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())