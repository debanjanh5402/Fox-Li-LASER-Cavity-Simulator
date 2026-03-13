# -*- coding: utf-8 -*-
import sys 
import os
import time
import numpy as np

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QMessageBox, QTabWidget, QFileDialog, QInputDialog, QFormLayout, QComboBox) 
from PyQt5.QtCore import Qt, QTimer 
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.figure import Figure 

from gain_processor import load_and_process_gain
from physics_engine import run_iteration_np, calc_far_field_np, create_circle, create_mirror
import viz_utils

class FoxLiGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fox-Li LASER Cavity Simulator (NumPy)")
        self.showFullScreen()
        self.simulation_running = False
        self.gain_filepath = None
        
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        self.tabs = QTabWidget()

        self.tab_visual = QWidget()
        self.tab_simulation = QWidget()
        self.tab_results = QWidget()
        self.tab_far_field = QWidget()

        self.tabs.addTab(self.tab_visual, "Setup Visualisation")
        self.tabs.addTab(self.tab_simulation, "Cavity Simulation")
        self.tabs.addTab(self.tab_results, "Simulation Results")
        self.tabs.addTab(self.tab_far_field, "Far-Field analysis")

        self.init_param_panel()
        self.init_visual_tab()
        self.init_simulation_tab()
        self.init_results_tab()
        self.init_far_field_tab()

        main_layout.addLayout(self.control_layout, 25)
        main_layout.addWidget(self.tabs, 75)
        self.setLayout(main_layout)

        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.run_iteration)

    def init_param_panel(self):
        self.control_layout = QVBoxLayout()
        self.control_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.control_layout.addSpacing(10)

        self.inputs = {}

        simulation_label = QLabel("Simulation Parameters")
        font = simulation_label.font()
        font.setPointSize(15)
        font.setBold(True)
        simulation_label.setFont(font)
        simulation_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(simulation_label)
        self.control_layout.addSpacing(10)

        labels_general = ["Grid size (N)", "Wavelength (μm)", "Pixel size (μm)", "Propagation distance z (m)", "Max Iter"]
        defaults_general = [1501, 1.315, 20.0, 1.212352, 500]
        keys_general = ['N', 'wav', 'p', 'z', 'max_iter']

        general_layout = QFormLayout()
        general_layout.setLabelAlignment(Qt.AlignLeft)
        general_layout.setFormAlignment(Qt.AlignLeft)

        for label, default, key in zip(labels_general, defaults_general, keys_general):
            le = QLineEdit(str(default))
            self.inputs[key] = le
            general_layout.addRow(QLabel(label), le)

        self.control_layout.addLayout(general_layout)
        self.control_layout.addSpacing(10)

        mirror_label = QLabel("Mirror Parameters")
        font = mirror_label.font()
        font.setPointSize(15)
        font.setBold(True)
        mirror_label.setFont(font)
        mirror_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(mirror_label)
        self.control_layout.addSpacing(10)

        mirror_param_layout = QGridLayout()
        mirror_param_layout.setAlignment(Qt.AlignCenter)
        
        mirror_param_layout.addWidget(QLabel("R1 (m)"), 0, 0)
        le_r1 = QLineEdit("5.8378044")
        self.inputs['R1'] = le_r1
        mirror_param_layout.addWidget(le_r1, 0, 1)
        
        mirror_param_layout.addWidget(QLabel("R2 (m)"), 0, 2)
        le_r2 = QLineEdit("3.4378044")
        self.inputs['R2'] = le_r2
        mirror_param_layout.addWidget(le_r2, 0, 3)

        mirror_param_layout.addWidget(QLabel("D1 (mm)"), 1, 0)
        le_d1 = QLineEdit("20.0")
        self.inputs['D1'] = le_d1
        mirror_param_layout.addWidget(le_d1, 1, 1)
        
        mirror_param_layout.addWidget(QLabel("D2 (mm)"), 1, 2)
        le_d2 = QLineEdit("11.0")
        self.inputs['D2'] = le_d2
        mirror_param_layout.addWidget(le_d2, 1, 3)

        mirror_param_layout.addWidget(QLabel("k1"), 2, 0)
        le_k1 = QLineEdit("-1.0")
        self.inputs['k01'] = le_k1
        mirror_param_layout.addWidget(le_k1, 2, 1)
        
        mirror_param_layout.addWidget(QLabel("k2"), 2, 2)
        le_k2 = QLineEdit("-1.03360")
        self.inputs['k02'] = le_k2
        mirror_param_layout.addWidget(le_k2, 2, 3)

        self.control_layout.addLayout(mirror_param_layout)
        self.control_layout.addSpacing(10)

        misalign_label = QLabel("Misalignment Parameters")
        font = misalign_label.font()
        font.setPointSize(15)
        font.setBold(True)
        misalign_label.setFont(font)
        misalign_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(misalign_label)
        self.control_layout.addSpacing(10)

        mirror_layout = QGridLayout()
        mirror_layout.setAlignment(Qt.AlignCenter)

        mis_labels = ["x (μm)", "y (μm)", "θx (μ rad)", "θy (μ rad)"]
        mis_keys_m1 = ['x1', 'y1', 'theta_x1', 'theta_y1']
        mis_keys_m2 = ['x2', 'y2', 'theta_x2', 'theta_y2']
        mis_defaults = [0.0, 0.0, 0.0, 0.0]

        mirror_layout.addWidget(QLabel("<b>Mirror 1</b>"), 0, 0, 1, 2, Qt.AlignCenter)
        for i, (label, key, default) in enumerate(zip(mis_labels, mis_keys_m1, mis_defaults), start=1):
            mirror_layout.addWidget(QLabel(label), i, 0)
            le = QLineEdit(str(default))
            self.inputs[key] = le
            mirror_layout.addWidget(le, i, 1)

        mirror_layout.addWidget(QLabel("<b>Mirror 2</b>"), 0, 2, 1, 2, Qt.AlignCenter)
        for i, (label, key, default) in enumerate(zip(mis_labels, mis_keys_m2, mis_defaults), start=1):
            mirror_layout.addWidget(QLabel(label), i, 2)
            le = QLineEdit(str(default))
            self.inputs[key] = le
            mirror_layout.addWidget(le, i, 3)

        self.control_layout.addLayout(mirror_layout)
        self.control_layout.addSpacing(10)

        gain_label = QLabel("Gain Profile")
        gain_label.setFont(font)
        gain_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(gain_label)
        self.control_layout.addSpacing(10)
        
        gain_layout = QFormLayout()
        self.gain_combo = QComboBox()
        self.gain_combo.addItems(["No Gain", "Load from File..."])
        gain_layout.addRow(QLabel("Gain Mode:"), self.gain_combo)
        
        self.btn_browse_gain = QPushButton("Browse...")
        self.btn_browse_gain.clicked.connect(self.select_gain_file)
        self.gain_filepath_label = QLabel("No file selected.")
        self.gain_filepath_label.setWordWrap(True)
        gain_layout.addRow(self.btn_browse_gain, self.gain_filepath_label)
        self.control_layout.addLayout(gain_layout)
        self.control_layout.addSpacing(10)

        self.btn_visualize = QPushButton("Visualize Setup")
        self.btn_visualize.clicked.connect(self.visualize_setup)
        self.control_layout.addWidget(self.btn_visualize, alignment=Qt.AlignTop)

        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.clicked.connect(self.initialize_simulation)
        self.control_layout.addWidget(self.btn_run, alignment=Qt.AlignTop)

        self.btn_save = QPushButton("Save Results")
        self.btn_save.clicked.connect(self.save_results)
        self.control_layout.addWidget(self.btn_save, alignment=Qt.AlignTop)

    def init_visual_tab(self):
        layout = QVBoxLayout()
        self.visual_figure = Figure(figsize=(16, 12))
        self.visual_canvas = FigureCanvas(self.visual_figure)
        layout.addWidget(self.visual_canvas)
        self.tab_visual.setLayout(layout)

    def init_simulation_tab(self):
        layout = QVBoxLayout()
        self.sim_figure = Figure(figsize=(16, 12))
        self.sim_canvas = FigureCanvas(self.sim_figure)
        layout.addWidget(self.sim_canvas)
        self.tab_simulation.setLayout(layout)

    def init_results_tab(self):
        layout = QVBoxLayout()
        self.result_figure = Figure(figsize=(16, 16))
        self.result_canvas = FigureCanvas(self.result_figure)
        layout.addWidget(self.result_canvas)
        self.tab_results.setLayout(layout)

    def init_far_field_tab(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        self.btn_calculate_ff = QPushButton("Calculate Far-Field & M²")
        self.btn_calculate_ff.clicked.connect(self.calculate_far_field)
        top_layout.addWidget(self.btn_calculate_ff, 1)
        
        results_layout = QFormLayout()
        self.m2_label = QLabel("Not calculated")
        self.dr_label = QLabel("Not calculated")
        self.dr_gauss_label = QLabel("Not calculated")
        self.drho_label = QLabel("Not calculated")
        self.drho_gauss_label = QLabel("Not calculated")
        
        results_layout.addRow("<b>M² Factor:</b>", self.m2_label)
        results_layout.addRow("<b>Dr (simulated):</b>", self.dr_label)
        results_layout.addRow("<b>D_rho (simulated):</b>", self.drho_label)
        results_layout.addRow("<b>Dr_gauss (Gaussian):</b>", self.dr_gauss_label)
        results_layout.addRow("<b>D_rho_gauss (Gaussian):</b>", self.drho_gauss_label)

        top_layout.addLayout(results_layout, 2)
        main_layout.addLayout(top_layout)
        
        self.far_field_figure = Figure(figsize=(16, 12))
        self.far_field_canvas = FigureCanvas(self.far_field_figure)
        main_layout.addWidget(self.far_field_canvas)
        self.tab_far_field.setLayout(main_layout)

    def select_gain_file(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Gain Profile Data File", "", 
                                                  "Text Files (*.txt);;All Files (*)", options=options)
        if filepath:
            self.gain_filepath = filepath
            self.gain_filepath_label.setText(os.path.basename(filepath))
            self.gain_combo.setCurrentIndex(1)

    def get_inputs(self):
        self.N = int(self.inputs['N'].text())
        self.wav = float(self.inputs['wav'].text()) * 1e-6    
        self.p = float(self.inputs['p'].text()) * 1e-6        
        self.z = float(self.inputs['z'].text())
        self.R1 = float(self.inputs['R1'].text())
        self.R2 = float(self.inputs['R2'].text())
        self.D1 = float(self.inputs['D1'].text()) * 1e-3      
        self.D2 = float(self.inputs['D2'].text()) * 1e-3
        self.max_iter = int(self.inputs['max_iter'].text())
        self.k01 = float(self.inputs['k01'].text())
        self.k02 = float(self.inputs['k02'].text())
        self.k = 2 * np.pi / self.wav

        self.x1 = float(self.inputs['x1'].text()) * 1e-6      
        self.y1 = float(self.inputs['y1'].text()) * 1e-6      
        self.x2 = float(self.inputs['x2'].text()) * 1e-6      
        self.y2 = float(self.inputs['y2'].text()) * 1e-6      

        self.theta_x1 = float(self.inputs['theta_x1'].text()) * 1e-6 
        self.theta_y1 = float(self.inputs['theta_y1'].text()) * 1e-6 
        self.theta_x2 = float(self.inputs['theta_x2'].text()) * 1e-6 
        self.theta_y2 = float(self.inputs['theta_y2'].text()) * 1e-6 

        x0 = np.linspace(-self.N/2+0.5, self.N/2-0.5, self.N, endpoint=True)
        x_coords, y_coords = np.meshgrid(x0, x0)
        self.x = x_coords * self.p
        self.y = y_coords * self.p

        fx0 = np.linspace(-0.5, 0.5, self.N, endpoint=True)
        fx_coords, fy_coords = np.meshgrid(fx0, fx0)
        self.fx = fx_coords / self.p
        self.fy = fy_coords / self.p
        
        self.k_sq = self.k**2
        self.four_pi_sq = 4 * np.pi**2
        self.f_sq_sum = self.fx**2 + self.fy**2

        self.circ0 = create_circle(x_grid=self.x, y_grid=self.y, diameter=(self.N*self.p))
        self.circ1, self.Mirror1 = create_mirror(x_grid=self.x, y_grid=self.y, wav_num=self.k,
                                                 diameter=self.D1, ROC=self.R1, kappa=self.k01,
                                                 xoff=self.x1, yoff=self.y1, angx=self.theta_x1, angy=self.theta_y1,
                                                 left_or_right_mirror="left", return_circ=True)
        self.circ2, self.Mirror2 = create_mirror(x_grid=self.x, y_grid=self.y, wav_num=self.k,
                                                 diameter=self.D2, ROC=self.R2, kappa=self.k02,
                                                 xoff=self.x2, yoff=self.y2, angx=self.theta_x2, angy=self.theta_y2,
                                                 left_or_right_mirror="right", return_circ=True)
        
        if self.gain_combo.currentText() == "Load from File..." and self.gain_filepath:
            time_start = time.perf_counter()
            self.exp_gain_profile, self.raw_gain_profile = load_and_process_gain(self.gain_filepath, self.x, self.y)
            time_end = time.perf_counter()
            print(f"Time duration for processing gain: {time_end-time_start} seconds")
        else:
            self.exp_gain_profile = np.ones((self.N, self.N))
            self.raw_gain_profile = None

        amp = np.random.uniform(size=(self.N, self.N))
        phase = np.random.uniform(size=(self.N, self.N))
        self.E0 = amp * np.exp(2j * np.pi * phase)
        self.iter = 0

    def visualize_setup(self):
        try:
            self.get_inputs()
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Error reading parameters: {e}")
            return
        
        Mirror1_nominal = create_mirror(x_grid=self.x, y_grid=self.y, wav_num=self.k,
                                        diameter=self.D1, ROC=self.R1, kappa=self.k01, 
                                        xoff=0.0, yoff=0.0, angx=0.0, angy=0.0,
                                        left_or_right_mirror="left",
                                        return_circ=False)
        Mirror2_nominal = create_mirror(x_grid=self.x, y_grid=self.y, wav_num=self.k,
                                        diameter=self.D2, ROC=self.R2, kappa=self.k02, 
                                        xoff=0.0, yoff=0.0, angx=0.0, angy=0.0,
                                        left_or_right_mirror="right",
                                        return_circ=False)
        
        viz_utils.plot_setup(self.visual_figure, Mirror1_nominal, Mirror2_nominal, 
                             self.Mirror1, self.Mirror2, 
                             self.circ1, self.circ2, 
                             self.raw_gain_profile, self.exp_gain_profile)
        
        self.visual_canvas.draw()
        self.tabs.setCurrentWidget(self.tab_visual)

    def initialize_simulation(self):
        try:
            self.get_inputs()
            QApplication.processEvents() 

            amp = np.random.uniform(size=(self.N, self.N))
            phase = np.random.uniform(size=(self.N, self.N))
            self.E0 = amp * np.exp(2j * np.pi * phase)
            self.iter = 0
            
            self.simulation_running = True
            self.tabs.setCurrentWidget(self.tab_simulation)
            self.sim_start_time = time.perf_counter()
            self.simulation_timer.start(5)

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))

    def run_iteration(self):
        if not self.simulation_running: return
        
        self.E0, self.last_E_out, intensity, phase = run_iteration_np(
            self.E0, self.Mirror1, self.Mirror2, self.exp_gain_profile, 
            self.z, self.circ2, self.circ0, 
            self.k_sq, self.four_pi_sq, self.f_sq_sum, 
            self.N, self.p, self.wav
        )
        
        viz_utils.plot_iteration(self.sim_figure, intensity, phase, self.iter + 1)
        self.sim_canvas.draw()
        
        self.iter += 1
        
        if self.iter >= self.max_iter:
            sim_end_time = time.perf_counter()
            print(f"Total time for {self.max_iter} iterations (including GUI rendering & delays): {sim_end_time - self.sim_start_time} seconds")
            self.simulation_running = False
            self.simulation_timer.stop()
            QMessageBox.information(self, "Simulation Complete", f"{self.max_iter} iterations completed!")
            self.plot_results()

    def plot_results(self):
        intensity = np.abs(self.last_E_out)**2 #* self.circ0 * (1 - self.circ2)
        phase = np.angle(self.last_E_out) * self.circ0 * (1 - self.circ2)
        
        viz_utils.plot_final_results(self.result_figure, intensity, phase, self.N // 2)
        
        self.result_canvas.draw()
        self.tabs.setCurrentWidget(self.tab_results)

    def calculate_far_field(self):
        if not hasattr(self, 'last_E_out'):
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return
        
        M2, Dr, Drho, Dr_gauss, Drho_gauss, I_out, I_far, I_gauss, I_far_gauss = calc_far_field_np(
            self.last_E_out, self.x, self.y, self.fx, self.fy, self.D1, self.D2, self.circ1, self.N)

        viz_utils.plot_far_field(self.far_field_figure, I_out, I_gauss, I_far, I_far_gauss, self.N)
        self.far_field_canvas.draw()
        
        self.m2_label.setText(f"{M2:.4f}")
        self.dr_label.setText(f"{Dr:.4e}")
        self.drho_label.setText(f"{Drho:.4f}")
        self.dr_gauss_label.setText(f"{Dr_gauss:.4e}")
        self.drho_gauss_label.setText(f"{Drho_gauss:.4f}")

        self.tabs.setCurrentWidget(self.tab_far_field)

    def save_results(self):
        options = ["Setup Visualisation", "Cavity Simulation", "Simulation Results", "Far-Field analysis", "All"]
        choice, ok = QInputDialog.getItem(self, "Select Figure", "Choose which plot to save:", options, 0, False)
        if ok and choice:
            if choice == "All":
                dir_path = QFileDialog.getExistingDirectory(self, "Select Folder to Save All Figures")
                if dir_path:
                    self.visual_figure.savefig(os.path.join(dir_path, "01_setup_visualisation.png"), dpi=300)
                    self.sim_figure.savefig(os.path.join(dir_path, "02_cavity_simulation.png"), dpi=300)
                    self.result_figure.savefig(os.path.join(dir_path, "03_simulation_results.png"), dpi=300)
                    self.far_field_figure.savefig(os.path.join(dir_path, "04_far_field_analysis.png"), dpi=300)
                    QMessageBox.information(self, "Saved", f"All figures saved in {dir_path}")
            else:
                file_path, _ = QFileDialog.getSaveFileName(self, f"Save {choice} Figure", "", "PNG Images (*.png)")
                if file_path:
                    figure_map = {
                        "Setup Visualisation": self.visual_figure, 
                        "Cavity Simulation": self.sim_figure,
                        "Simulation Results": self.result_figure, 
                        "Far-Field analysis": self.far_field_figure
                    }
                    figure_map[choice].savefig(file_path, dpi=300)

if __name__ == '__main__':
    print("Running in standard NumPy mode.")
    app = QApplication(sys.argv)
    window = FoxLiGUI()
    window.show()
    sys.exit(app.exec_())