# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_phase_map(ax, data, title, circ_mask, figure):
    ax.set_title(title, fontsize=8, fontweight='bold')
    phase_data = np.angle(data) * circ_mask
    im = ax.imshow(phase_data, cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    figure.colorbar(im, cax=cax)
    ax.axis('off')


def plot_setup(fig, M1_nom, M2_nom, M1_mis, M2_mis, circ1, circ2, raw_gain, exp_gain):
    fig.clf()
    gs = GridSpec(2, 4, figure=fig)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_gain_raw = fig.add_subplot(gs[1, 0:2])
    ax_gain_interp = fig.add_subplot(gs[1, 2:4])
    fig.subplots_adjust(wspace=0.4, hspace=0.5)

    plot_phase_map(axs[0], M1_nom, "M1 Phase-Nominal", circ1, fig)
    plot_phase_map(axs[1], M2_nom, "M2 Phase-Nominal", circ2, fig)
    plot_phase_map(axs[2], M1_mis, "M1 Phase-Misaligned", circ1, fig)
    plot_phase_map(axs[3], M2_mis, "M2 Phase-Misaligned", circ2, fig)

    if raw_gain is not None:
        #im4 = ax_gain_raw.imshow(raw_gain/np.max(raw_gain), cmap='viridis', origin='lower', aspect='auto')
        im4 = ax_gain_raw.imshow(raw_gain, cmap='viridis', origin='lower', aspect='auto')
        ax_gain_raw.set_title("2D Projected Cavity Gain Map (Scaled)", fontsize=8, fontweight='bold')
        fig.colorbar(im4, ax=ax_gain_raw)
    else:
        ax_gain_raw.text(0.5, 0.5, "No Gain Data Loaded", ha='center', va='center')
        ax_gain_raw.set_title("2D Projected Cavity Gain Map", fontsize=8, fontweight='bold')

    if exp_gain is not None:
        #im5 = ax_gain_interp.imshow(exp_gain/np.max(exp_gain), cmap='viridis', origin='lower', aspect='auto')
        im5 = ax_gain_interp.imshow(exp_gain, cmap='viridis', origin='lower', aspect='auto')
        ax_gain_interp.set_title("Final Transmission Multiplier (Exp)", fontsize=8, fontweight='bold')
        fig.colorbar(im5, ax=ax_gain_interp)
    else:
        ax_gain_interp.text(0.5, 0.5, "No Gain Applied", ha='center', va='center')
        ax_gain_interp.set_title("Final Transmission Multiplier", fontsize=8, fontweight='bold')


def plot_iteration(fig, intensity, phase, iteration):
    fig.clf()
    axs = fig.subplots(1, 2)
    fig.subplots_adjust(wspace=0.5)

    intensity_norm = intensity / (np.max(intensity) + 1e-16)
    
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="2%", pad=0.05)
    im0 = axs[0].imshow(intensity_norm, cmap='hot')
    axs[0].set_title("Intensity", fontsize=12, fontweight='bold')
    fig.colorbar(im0, cax=cax0)

    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="2%", pad=0.05)
    im1 = axs[1].imshow(phase, cmap='jet')
    axs[1].set_title("Phase", fontsize=12, fontweight='bold')
    fig.colorbar(im1, cax=cax1)
    fig.suptitle(f"Iteration {iteration}", fontsize=20, fontweight='bold')


def plot_final_results(fig, intensity, phase, center_row):
    fig.clf()
    axs = fig.subplots(2, 2)
    fig.subplots_adjust(wspace=0.5, hspace=0.6)

    intensity_norm = intensity / (np.max(intensity) + 1e-16)

    divider00 = make_axes_locatable(axs[0, 0])
    cax00 = divider00.append_axes("right", size="2%", pad=0.05)
    im00 = axs[0, 0].imshow(intensity_norm, cmap='hot', interpolation='nearest')
    axs[0, 0].set_title("Final Intensity", fontsize=12, fontweight='bold')
    fig.colorbar(im00, cax=cax00)

    divider01 = make_axes_locatable(axs[0, 1])
    cax01 = divider01.append_axes("right", size="2%", pad=0.05)
    im01 = axs[0, 1].imshow(phase, cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
    axs[0, 1].set_title("Final Phase", fontsize=12, fontweight='bold')
    fig.colorbar(im01, cax=cax01)

    axs[1, 0].plot(intensity_norm[center_row, :], linewidth=1)
    axs[1, 0].plot(intensity_norm[:, center_row], linewidth=1)
    axs[1, 0].set_title("Central Intensities", fontsize=12, fontweight='bold')
    axs[1, 0].grid(True, linestyle='--', alpha=0.25)

    axs[1, 1].plot(phase[center_row, :], linewidth=1)
    axs[1, 1].plot(phase[:, center_row], linewidth=1)
    axs[1, 1].set_title("Central Phases", fontsize=12, fontweight='bold')
    axs[1, 1].grid(True, linestyle='--', alpha=0.25)


def plot_far_field(fig, I_out, I_gauss, I_far, I_far_gauss, N):
    fig.clf()
    axs = fig.subplots(2, 2)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    divider00 = make_axes_locatable(axs[0, 0])
    cax00 = divider00.append_axes("right", size="2%", pad=0.05)
    im00 = axs[0, 0].imshow(I_out / (np.max(I_out) + 1e-16), cmap='hot')
    axs[0, 0].set_title("Simulated Beam Near-Field (Intensity)", fontsize=5, fontweight='bold')
    fig.colorbar(im00, cax=cax00)

    divider01 = make_axes_locatable(axs[0, 1])
    cax01 = divider01.append_axes("right", size="2%", pad=0.05)
    im01 = axs[0, 1].imshow(I_gauss / (np.max(I_gauss) + 1e-16), cmap='hot')
    axs[0, 1].set_title("Reference Gaussian Near-Field (Intensity)", fontsize=5, fontweight='bold')
    fig.colorbar(im01, cax=cax01)
    
    divider10 = make_axes_locatable(axs[1, 0])
    cax10 = divider10.append_axes("right", size="2%", pad=0.05)
    I_far_slice = I_far[N//2-50:N//2+50, N//2-50:N//2+50]
    im10 = axs[1, 0].imshow(I_far_slice**(0.5), cmap='jet')
    axs[1, 0].set_title("Simulated Beam Far-Field (amplitude)", fontsize=5, fontweight='bold')
    fig.colorbar(im10, cax=cax10)

    divider11 = make_axes_locatable(axs[1, 1])
    cax11 = divider11.append_axes("right", size="2%", pad=0.05)
    I_far_gauss_slice = I_far_gauss[N//2-50:N//2+50, N//2-50:N//2+50]
    im11 = axs[1, 1].imshow(I_far_gauss_slice**(0.5), cmap='jet')
    axs[1, 1].set_title("Reference Gaussian Far-Field (amplitude)", fontsize=5, fontweight='bold')
    fig.colorbar(im11, cax=cax11)