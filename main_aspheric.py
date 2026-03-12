# -*- coding: utf-8 -*-
import sys 
import os
os.environ['JAX_ENABLE_X64'] = 'True'

import pandas as pd # type:ignore
try:
    import jax          
    import jax.numpy as jnp 
    from jax import jit     
    import numpy as onp     
except ImportError:
    print("Error: JAX or jaxlib not installed. Please install them to run this JAX-accelerated code.")
    sys.exit(1)

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QMessageBox, QTabWidget, QFileDialog, QInputDialog, QFormLayout, QComboBox) 
from PyQt5.QtCore import Qt, QTimer # type:ignore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # type:ignore
from matplotlib.figure import Figure # type:ignore
from matplotlib.gridspec import GridSpec # type:ignore
from mpl_toolkits.axes_grid1 import make_axes_locatable # type:ignore

from gain_processor import load_and_process_gain
from physics_engine import _angspec_prop_core, _run_iteration_core

class FoxLiGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fox-Li LASER Cavity Simulator (JAX-Accelerated)")
        self.showFullScreen()
        self.simulation_running = False
        self.gain_filepath = None
        
        self._angspec_prop_jax = jit(_angspec_prop_core, static_argnums=(5, 6, 7))
        self._run_iteration_jax = jit(_run_iteration_core, static_argnums=(10, 11, 12))
        
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        self.tabs = QTabWidget()

        self.tab_visual = QWidget(); self.tab_simulation = QWidget()
        self.tab_results = QWidget(); self.tab_far_field = QWidget()

        self.tabs.addTab(self.tab_visual, "Setup Visualisation"); self.tabs.addTab(self.tab_simulation, "Cavity Simulation")
        self.tabs.addTab(self.tab_results, "Simulation Results"); self.tabs.addTab(self.tab_far_field, "Far-Field analysis")

        self.init_param_panel()
        self.init_visual_tab(); self.init_simulation_tab(); self.init_results_tab()
        self.init_far_field_tab()

        main_layout.addLayout(self.control_layout, 25); main_layout.addWidget(self.tabs, 75)
        self.setLayout(main_layout)

        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.run_iteration)

    def init_param_panel(self):
        self.control_layout = QVBoxLayout(); self.control_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.control_layout.addSpacing(10)

        self.inputs = {}

        simulation_label = QLabel("Simulation Parameters")
        font = simulation_label.font(); font.setPointSize(15); font.setBold(True)
        simulation_label.setFont(font); simulation_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(simulation_label); self.control_layout.addSpacing(10)

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
        font = mirror_label.font(); font.setPointSize(15); font.setBold(True)
        mirror_label.setFont(font); mirror_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(mirror_label); self.control_layout.addSpacing(10)

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
        font = misalign_label.font(); font.setPointSize(15); font.setBold(True)
        misalign_label.setFont(font); misalign_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(misalign_label); self.control_layout.addSpacing(10)

        mirror_layout = QGridLayout(); mirror_layout.setAlignment(Qt.AlignCenter)

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

        self.control_layout.addLayout(mirror_layout); self.control_layout.addSpacing(10)

        gain_label = QLabel("Gain Profile"); gain_label.setFont(font); gain_label.setAlignment(Qt.AlignCenter)
        self.control_layout.addWidget(gain_label); self.control_layout.addSpacing(10)
        
        gain_layout = QFormLayout()
        self.gain_combo = QComboBox(); self.gain_combo.addItems(["No Gain", "Load from File..."])
        gain_layout.addRow(QLabel("Gain Mode:"), self.gain_combo)
        
        self.btn_browse_gain = QPushButton("Browse...")
        self.btn_browse_gain.clicked.connect(self.select_gain_file)
        self.gain_filepath_label = QLabel("No file selected."); self.gain_filepath_label.setWordWrap(True)
        gain_layout.addRow(self.btn_browse_gain, self.gain_filepath_label)
        self.control_layout.addLayout(gain_layout); self.control_layout.addSpacing(10)

        self.btn_visualize = QPushButton("Visualize Setup"); self.btn_visualize.clicked.connect(self.visualize_setup)
        self.control_layout.addWidget(self.btn_visualize, alignment=Qt.AlignTop)

        self.btn_run = QPushButton("Run Simulation"); self.btn_run.clicked.connect(self.initialize_simulation)
        self.control_layout.addWidget(self.btn_run, alignment=Qt.AlignTop)

        self.btn_save = QPushButton("Save Results"); self.btn_save.clicked.connect(self.save_results)
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
        main_layout = QVBoxLayout(); top_layout = QHBoxLayout()
        self.btn_calculate_ff = QPushButton("Calculate Far-Field & M²"); self.btn_calculate_ff.clicked.connect(self.calculate_far_field)
        top_layout.addWidget(self.btn_calculate_ff, 1)
        
        results_layout = QFormLayout()
        self.m2_label = QLabel("Not calculated"); self.dr_label = QLabel("Not calculated")
        self.dr_gauss_label = QLabel("Not calculated"); self.drho_label = QLabel("Not calculated")
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
        self.k = 2 * onp.pi / self.wav

        self.x1 = float(self.inputs['x1'].text()) * 1e-6      
        self.y1 = float(self.inputs['y1'].text()) * 1e-6      
        self.x2 = float(self.inputs['x2'].text()) * 1e-6      
        self.y2 = float(self.inputs['y2'].text()) * 1e-6      

        self.theta_x1 = float(self.inputs['theta_x1'].text()) * 1e-6 
        self.theta_y1 = float(self.inputs['theta_y1'].text()) * 1e-6 
        self.theta_x2 = float(self.inputs['theta_x2'].text()) * 1e-6 
        self.theta_y2 = float(self.inputs['theta_y2'].text()) * 1e-6 

        x0 = jnp.linspace(-self.N/2+0.5, self.N/2-0.5, self.N, endpoint=True)
        x_coords, y_coords = jnp.meshgrid(x0, x0)
        self.x = x_coords * self.p
        self.y = y_coords * self.p

        fx0 = jnp.linspace(-0.5, 0.5, self.N, endpoint=True)
        fx_coords, fy_coords = jnp.meshgrid(fx0, fx0)
        self.fx = fx_coords / self.p
        self.fy = fy_coords / self.p
        
        self.k_sq = self.k**2
        self.four_pi_sq = 4 * onp.pi**2
        self.f_sq_sum = self.fx**2 + self.fy**2
        
        self.circ0 = jnp.zeros((self.N, self.N)); self.circ0 = self.circ0.at[self.x**2 + self.y**2 < (self.N*self.p/2)**2].set(1)
        self.circ1 = jnp.zeros((self.N, self.N)); self.circ1 = self.circ1.at[self.x**2 + self.y**2 < (self.D1/2)**2].set(1)
        self.circ2 = jnp.zeros((self.N, self.N)); self.circ2 = self.circ2.at[self.x**2 + self.y**2 < (self.D2/2)**2].set(1)
        
        r21 = (self.x - self.x1)**2 + (self.y - self.y1)**2
        sag1_phase = -2j * self.k * r21 / (self.R1 + jnp.sqrt(self.R1**2 - (1 + self.k01) * r21))
        tilt1 = jnp.exp(1j * self.k * (self.x * self.theta_x1 + self.y * self.theta_y1))
        self.Mirror1 = jnp.exp(sag1_phase) * tilt1 * self.circ1
        
        r22 = (self.x - self.x2)**2 + (self.y - self.y2)**2
        sag2_phase = 2j * self.k * r22 / (self.R2 + jnp.sqrt(self.R2**2 - (1 + self.k02) * r22))
        tilt2 = jnp.exp(1j * self.k * (self.x * self.theta_x2 + self.y * self.theta_y2))
        self.Mirror2 = jnp.exp(sag2_phase) * tilt2 * self.circ2
        
        if self.gain_combo.currentText() == "Load from File..." and self.gain_filepath:
            gain_prof, raw_prof = load_and_process_gain(self.gain_filepath, onp.array(self.x), onp.array(self.y))
            self.gain_profile = jnp.array(gain_prof)
            self.raw_gain_profile = raw_prof
        else:
            self.gain_profile = jnp.ones((self.N, self.N))
            self.raw_gain_profile = None

        key = jax.random.PRNGKey(42)
        key_amp, key_phase = jax.random.split(key)
        self.E0 = jax.random.uniform(key_amp, (self.N, self.N), dtype=jnp.float64) * jnp.exp(1j * 2 * onp.pi * jax.random.uniform(key_phase, (self.N, self.N), dtype=jnp.float64))
        self.iter = 0

    def visualize_setup(self):
        try:
            self.get_inputs()
        except Exception as e:
            QMessageBox.critical(self, "Input Error", f"Error reading parameters: {e}")
            return

        self.visual_figure.clf()
        gs = GridSpec(2, 4, figure=self.visual_figure)
        ax_m1_nom = self.visual_figure.add_subplot(gs[0, 0])
        ax_m2_nom = self.visual_figure.add_subplot(gs[0, 1])
        ax_m1_mis = self.visual_figure.add_subplot(gs[0, 2])
        ax_m2_mis = self.visual_figure.add_subplot(gs[0, 3])
        ax_gain_raw = self.visual_figure.add_subplot(gs[1, 0:2])
        ax_gain_interp = self.visual_figure.add_subplot(gs[1, 2:4])
        
        self.visual_figure.subplots_adjust(wspace=0.4, hspace=0.5)

        
        r2_nom = self.x**2 + self.y**2
        sag1_nom = -2j * self.k * r2_nom / (self.R1 + jnp.sqrt(self.R1**2 - (1 + self.k01) * r2_nom))
        Mirror1_nominal = jnp.exp(sag1_nom) * self.circ1
        sag2_nom = 2j * self.k * r2_nom / (self.R2 + jnp.sqrt(self.R2**2 - (1 + self.k02) * r2_nom))
        Mirror2_nominal = jnp.exp(sag2_nom) * self.circ2
        
        def plot_phase(ax, data, title, circ_mask):
            ax.set_title(title, fontsize=8, fontweight='bold')
            phase_data = onp.array(jnp.angle(data)) * onp.array(circ_mask)
            im = ax.imshow(phase_data, cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            self.visual_figure.colorbar(im, cax=cax)
            ax.axis('off')

        plot_phase(ax_m1_nom, Mirror1_nominal, "M1 Phase-Nominal", self.circ1)
        plot_phase(ax_m2_nom, Mirror2_nominal, "M2 Phase-Nominal", self.circ2)
        plot_phase(ax_m1_mis, self.Mirror1, "M1 Phase-Misaligned", self.circ1)
        plot_phase(ax_m2_mis, self.Mirror2, "M2 Phase-Misaligned", self.circ2)

        if hasattr(self, 'raw_gain_profile') and self.raw_gain_profile is not None:
            im4 = ax_gain_raw.imshow(self.raw_gain_profile/onp.max(self.raw_gain_profile), cmap='viridis', origin='lower', aspect='auto')
            ax_gain_raw.set_title("2D Projected Cavity Gain Map (Scaled)", fontsize=8, fontweight='bold')
            self.visual_figure.colorbar(im4, ax=ax_gain_raw)
        else:
            ax_gain_raw.text(0.5, 0.5, "No Gain Data Loaded", ha='center', va='center')
            ax_gain_raw.set_title("2D Projected Cavity Gain Map", fontsize=8, fontweight='bold')

        if hasattr(self, 'gain_profile') and self.gain_profile is not None and self.gain_combo.currentText() == "Load from File...":
            exp_gain_onp = onp.array(self.gain_profile)
            im5 = ax_gain_interp.imshow(exp_gain_onp/onp.max(exp_gain_onp), cmap='viridis', origin='lower', aspect='auto')
            ax_gain_interp.set_title(f"Final Transmission Multiplier (Exp)", fontsize=8, fontweight='bold')
            self.visual_figure.colorbar(im5, ax=ax_gain_interp)
        else:
            ax_gain_interp.text(0.5, 0.5, "No Gain Applied", ha='center', va='center')
            ax_gain_interp.set_title(f"Final Transmission Multiplier", fontsize=8, fontweight='bold')

        self.visual_canvas.draw()
        self.tabs.setCurrentWidget(self.tab_visual)

    def initialize_simulation(self):
        try:
            self.get_inputs()
            QApplication.processEvents() 

            E0_dummy = self.E0
            
            E0_next_dummy, E_out_dummy, intensity_dummy, phase_dummy = self._run_iteration_jax(
                E0_dummy, self.Mirror1, self.Mirror2, self.gain_profile, 
                self.z, self.circ2, self.circ0, 
                self.k_sq, self.four_pi_sq, self.f_sq_sum, 
                self.N, self.p, self.wav
            )
            
            E0_next_dummy.block_until_ready()

            key = jax.random.PRNGKey(42)
            key_amp, key_phase = jax.random.split(key)
            self.E0 = jax.random.uniform(key_amp, (self.N, self.N), dtype=jnp.float64) * jnp.exp(1j * 2 * onp.pi * jax.random.uniform(key_phase, (self.N, self.N), dtype=jnp.float64))
            self.iter = 0
            
            self.simulation_running = True
            self.tabs.setCurrentWidget(self.tab_simulation)
            self.simulation_timer.start(5)

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))

    def run_iteration(self):
        if not self.simulation_running: return
        
        self.E0, E_out_jax, intensity_jax, phase_jax = self._run_iteration_jax(
            self.E0, self.Mirror1, self.Mirror2, self.gain_profile, 
            self.z, self.circ2, self.circ0, 
            self.k_sq, self.four_pi_sq, self.f_sq_sum, 
            self.N, self.p, self.wav
        )
        
        E_out = onp.array(E_out_jax)
        intensity = onp.array(intensity_jax)
        phase = onp.array(phase_jax)
        
        self.last_E_out = E_out_jax 
        
        self.sim_figure.clf()
        axs = self.sim_figure.subplots(1, 2)
        self.sim_figure.subplots_adjust(wspace=0.5)

        divider0 = make_axes_locatable(axs[0]); cax0 = divider0.append_axes("right", size="2%", pad=0.05)
        intensity_norm = intensity / (onp.max(intensity) + 1e-16)
        im0 = axs[0].imshow(intensity_norm, cmap='hot'); axs[0].set_title("Intensity", fontsize=12, fontweight='bold')
        self.sim_figure.colorbar(im0, cax=cax0)

        divider1 = make_axes_locatable(axs[1]); cax1 = divider1.append_axes("right", size="2%", pad=0.05)
        im1 = axs[1].imshow(phase, cmap='jet'); axs[1].set_title("Phase", fontsize=12, fontweight='bold')
        self.sim_figure.colorbar(im1, cax=cax1)

        self.sim_figure.suptitle(f"Iteration {self.iter+1}", fontsize=20, fontweight='bold')
        self.sim_canvas.draw()
        
        self.iter += 1
        
        if self.iter >= self.max_iter:
            self.simulation_running = False
            self.simulation_timer.stop()
            self.last_E_out = onp.array(self.last_E_out)
            QMessageBox.information(self, "Simulation Complete", f"{self.max_iter} iterations completed!")
            self.plot_results()

    def plot_results(self):
        intensity = onp.abs(self.last_E_out)**2 * onp.array(self.circ0) * (1 - onp.array(self.circ2))
        phase = onp.angle(self.last_E_out) * onp.array(self.circ0) * (1 - onp.array(self.circ2))
        
        self.last_intensity = intensity; self.last_phase = phase; self.last_center_row = self.N // 2

        self.result_figure.clf()
        axs = self.result_figure.subplots(2, 2)
        self.result_figure.subplots_adjust(wspace=0.5, hspace=0.6)

        divider00 = make_axes_locatable(axs[0, 0]); cax00 = divider00.append_axes("right", size="2%", pad=0.05)
        im00 = axs[0, 0].imshow(intensity / onp.max(intensity), cmap='hot', interpolation='nearest')
        axs[0, 0].set_title("Final Intensity", fontsize=12, fontweight='bold')
        self.result_figure.colorbar(im00, cax=cax00)

        divider01 = make_axes_locatable(axs[0, 1]); cax01 = divider01.append_axes("right", size="2%", pad=0.05)
        im01 = axs[0, 1].imshow(phase, cmap='jet', vmin=-onp.pi, vmax=onp.pi, interpolation='nearest')
        axs[0, 1].set_title("Final Phase", fontsize=12, fontweight='bold')
        self.result_figure.colorbar(im01, cax=cax01)

        cr = self.last_center_row
        axs[1, 0].plot(intensity[cr, :]/onp.max(intensity), linewidth=1)
        axs[1, 0].plot(intensity[:, cr]/onp.max(intensity), linewidth=1)
        axs[1, 0].set_title("Central Intensities", fontsize=12, fontweight='bold')
        axs[1, 0].grid(True, linestyle='--', alpha=0.25)

        axs[1, 1].plot(phase[cr, :], linewidth=1)
        axs[1, 1].plot(phase[:, cr], linewidth=1)
        axs[1, 1].set_title("Central Phases", fontsize=12, fontweight='bold')
        axs[1, 1].grid(True, linestyle='--', alpha=0.25)

        self.result_canvas.draw()
        self.tabs.setCurrentWidget(self.tab_results)

    def calculate_far_field(self):
        if not hasattr(self, 'last_E_out'):
            QMessageBox.warning(self, "No Data", "Please run a simulation first.")
            return
        
        E_out = jnp.array(self.last_E_out)
        I_out = jnp.abs(E_out)**2
        
        E_far = jnp.fft.fftshift(jnp.fft.fft2(E_out))
        I_far = jnp.abs(E_far)**2
        
        x, y, fx, fy = self.x, self.y, self.fx, self.fy
        
        total_power_out = jnp.sum(I_out)
        x_c = jnp.sum(x * I_out) / total_power_out
        y_c = jnp.sum(y * I_out) / total_power_out
        r_c = jnp.sqrt(x_c**2 + y_c**2)
        Dr = jnp.sum(((jnp.sqrt(x**2 + y**2) - r_c)**2 * I_out)) / total_power_out

        total_power_far = jnp.sum(I_far)
        fx_c = jnp.sum(fx * I_far) / total_power_far
        fy_c = jnp.sum(fy * I_far) / total_power_far
        f_c = jnp.sqrt(fx_c**2 + fy_c**2)
        Drho = jnp.sum(((jnp.sqrt(fx**2 + fy**2) - f_c)**2) * I_far) / total_power_far

        w0 = jnp.mean(jnp.array([self.D1, self.D2]))
        E_gauss = jnp.exp(-(x**2 + y**2) / w0**2) * self.circ1
        I_gauss = jnp.abs(E_gauss)**2
        E_far_gauss = jnp.fft.fftshift(jnp.fft.fft2(E_gauss))
        I_far_gauss = jnp.abs(E_far_gauss)**2
        
        total_power_gauss = jnp.sum(I_gauss)
        Dr_gauss = jnp.sum((x**2 + y**2) * I_gauss) / total_power_gauss
        total_power_far_gauss = jnp.sum(I_far_gauss)
        Drho_gauss = jnp.sum((fx**2 + fy**2) * I_far_gauss) / total_power_far_gauss

        M2 = jnp.sqrt(Drho / Drho_gauss)
        
        M2_onp, Dr_onp, Drho_onp = onp.array(M2), onp.array(Dr), onp.array(Drho)
        Dr_gauss_onp, Drho_gauss_onp = onp.array(Dr_gauss), onp.array(Drho_gauss)
        I_out_onp, I_far_onp = onp.array(I_out), onp.array(I_far)
        I_gauss_onp, I_far_gauss_onp = onp.array(I_gauss), onp.array(I_far_gauss)

        self.far_field_figure.clf(); axs = self.far_field_figure.subplots(2, 2)
        self.far_field_figure.subplots_adjust(wspace=0.5, hspace=0.5)

        divider00 = make_axes_locatable(axs[0, 0]); cax00 = divider00.append_axes("right", size="2%", pad=0.05)
        im00 = axs[0, 0].imshow(I_out_onp/onp.max(I_out_onp), cmap='hot')
        axs[0, 0].set_title("Simulated Beam Near-Field (Intensity)", fontsize=5, fontweight='bold')
        self.far_field_figure.colorbar(im00, cax=cax00)

        divider01 = make_axes_locatable(axs[0, 1]); cax01 = divider01.append_axes("right", size="2%", pad=0.05)
        im01 = axs[0, 1].imshow(I_gauss_onp/onp.max(I_gauss_onp), cmap='hot')
        axs[0, 1].set_title("Reference Gaussian Near-Field (Intensity)", fontsize=5, fontweight='bold')
        self.far_field_figure.colorbar(im01, cax=cax01)
        
        divider10 = make_axes_locatable(axs[1, 0]); cax10 = divider10.append_axes("right", size="2%", pad=0.05)
        I_far_slice = I_far_onp[self.N//2-50:self.N//2+50, self.N//2-50:self.N//2+50]
        im10 = axs[1, 0].imshow(I_far_slice**(0.5), cmap='jet')
        axs[1, 0].set_title("Simulated Beam Far-Field (amplitude)", fontsize=5, fontweight='bold')
        self.far_field_figure.colorbar(im10, cax=cax10)

        divider11 = make_axes_locatable(axs[1, 1]); cax11 = divider11.append_axes("right", size="2%", pad=0.05)
        I_far_gauss_slice = I_far_gauss_onp[self.N//2-50:self.N//2+50, self.N//2-50:self.N//2+50]
        im11 = axs[1, 1].imshow(I_far_gauss_slice**(0.5), cmap='jet')
        axs[1, 1].set_title("Reference Gaussian Far-Field (amplitude)", fontsize=5, fontweight='bold')

        self.far_field_figure.colorbar(im11, cax=cax11); self.far_field_canvas.draw()
        
        self.m2_label.setText(f"{M2_onp:.4f}")
        self.dr_label.setText(f"{Dr_onp:.4e}")
        self.drho_label.setText(f"{Drho_onp:.4f}")
        self.dr_gauss_label.setText(f"{Dr_gauss_onp:.4e}")
        self.drho_gauss_label.setText(f"{Drho_gauss_onp:.4f}")

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
    print(f"JAX backend: {jax.default_backend()}")
    if jax.config.read('jax_enable_x64'):
        print(">> JAX is running in **64-bit precision** (float64/complex128). Expect maximum accuracy.")
    else:
        print(">> JAX is running in 32-bit precision. Results may differ from NumPy.")
        
    if jax.default_backend() == 'gpu':
        print("Using GPU acceleration via JAX/XLA. Expect significantly faster simulation iterations!")
    elif jax.default_backend() == 'cpu':
        print("Using CPU via JAX/XLA. The JIT compilation still provides good speedup over standard NumPy.")
    else:
        print("Unknown JAX backend. Continuing with default setup.")
        
    app = QApplication(sys.argv)
    window = FoxLiGUI()
    window.show()
    sys.exit(app.exec_())