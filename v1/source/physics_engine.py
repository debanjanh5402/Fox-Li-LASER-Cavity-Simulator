# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


# --- NumPy Physics Functions---
def create_circle(x_grid, y_grid, diameter, xoff=0.0, yoff=0.0):
    r2 = (x_grid-xoff)**2 + (y_grid-yoff)**2
    circ = np.where(r2 < (diameter/2)**2, 1.0, 0.0)
    return circ
        
def create_mirror(x_grid, y_grid, wav_num, 
                  diameter, ROC, kappa, 
                  xoff, yoff, angx, angy, 
                  left_or_right_mirror:str, 
                  return_circ:bool=False):
    circ = create_circle(x_grid, y_grid, diameter, xoff, yoff)
            
    if left_or_right_mirror.lower() == "left": 
        factor = -1
    elif left_or_right_mirror.lower() == "right": 
        factor = +1
    else: 
        print(f"Wrong argument for left_or_right_mirror: {left_or_right_mirror}")

    r2 = (x_grid-xoff)**2 + (y_grid-yoff)**2
    sag_phase = factor * 2j * wav_num * r2 / (ROC + np.sqrt(ROC**2 - (1+kappa) * r2))
    tilt = np.exp(1j* wav_num * (x_grid*angx + y_grid*angy))
    mirror = np.exp(sag_phase) * tilt * circ

    if return_circ: 
        return circ, mirror
    else:
        return mirror
            
def angspec_prop_np(u, dz, k_sq, four_pi_sq, f_sq_sum, N, p, wav):
    alpha = np.sqrt(k_sq - four_pi_sq * f_sq_sum) 
    f0 = (1/wav) * 1/np.sqrt(1 + (2*dz/(N*p))**2)
    LP = np.where(f_sq_sum <= f0**2, 1.0, 0.0)
    H = np.exp(1j * dz * alpha) * LP
    U = fftshift(fft2(ifftshift(u)))
    return fftshift(ifft2(ifftshift(U * H)))


def run_iteration_np(E0, Mirror1, Mirror2, gain_profile, z, circ2, circ0, k_sq, four_pi_sq, f_sq_sum, N, p, wav):
    E0 /= np.max(np.abs(E0))
    E0 = E0 * Mirror1
    E0 = angspec_prop_np(E0, z, k_sq, four_pi_sq, f_sq_sum, N, p, wav)
    #E0 = E0 * gain_profile
    E_out = E0 * (1 - circ2)
    E0 = E0 * Mirror2
    E0 = angspec_prop_np(E0, z, k_sq, four_pi_sq, f_sq_sum, N, p, wav)
    E0 = E0 * gain_profile
    
    intensity = np.abs(E_out)**2 #* circ0 #* (1 - circ2)
    phase = np.angle(E_out) * circ0 * (1 - circ2)
    
    return E0, E_out, intensity, phase


def calc_far_field_np(E_out, x, y, fx, fy, D1, D2, circ1, N):
    I_out = np.abs(E_out)**2
    E_far = fftshift(fft2(ifftshift(E_out)))
    I_far = np.abs(E_far)**2
    
    total_power_out = np.sum(I_out)
    x_c = np.sum(x * I_out) / total_power_out
    y_c = np.sum(y * I_out) / total_power_out
    r_c = np.sqrt(x_c**2 + y_c**2)
    Dr = np.sum(((np.sqrt(x**2 + y**2) - r_c)**2 * I_out)) / total_power_out

    total_power_far = np.sum(I_far)
    fx_c = np.sum(fx * I_far) / total_power_far
    fy_c = np.sum(fy * I_far) / total_power_far
    #f_c = np.sqrt(fx_c**2 + fy_c**2) * 0
    #Drho = np.sum(((jnp.sqrt(fx**2 + fy**2) - f_c)**2) * I_far) / total_power_far
    Drho = np.sum(((fx-fx_c)**2 + (fx-fy_c)**2) * I_far)/total_power_far

    E_gauss = np.exp(-2.0 * (x**2 + y**2) / (D1 + D2)**2) * circ1
    I_gauss = np.abs(E_gauss)**2
    E_far_gauss = fftshift(fft2(ifftshift(E_gauss)))
    I_far_gauss = np.abs(E_far_gauss)**2
    
    total_power_gauss = np.sum(I_gauss)
    Dr_gauss = np.sum((x**2 + y**2) * I_gauss) / total_power_gauss
    total_power_far_gauss = np.sum(I_far_gauss)
    Drho_gauss = np.sum((fx**2 + fy**2) * I_far_gauss) / total_power_far_gauss

    M2 = np.sqrt(Drho / Drho_gauss)
    return M2, Dr, Drho, Dr_gauss, Drho_gauss, I_out, I_far, I_gauss, I_far_gauss