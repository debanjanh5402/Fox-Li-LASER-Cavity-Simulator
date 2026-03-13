# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift


# --- NumPy Physics Functions---
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
    E0 = E0 * gain_profile
    E_out = E0 * (1 - circ2)
    E0 = E0 * Mirror2
    E0 = angspec_prop_np(E0, z, k_sq, four_pi_sq, f_sq_sum, N, p, wav)
    E0 = E0 * gain_profile
    
    intensity = np.abs(E_out)**2 * circ0 * (1 - circ2)
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
    f_c = np.sqrt(fx_c**2 + fy_c**2)
    Drho = np.sum(((np.sqrt(fx**2 + fy**2) - f_c)**2) * I_far) / total_power_far

    w0 = np.mean([D1, D2])
    E_gauss = np.exp(-(x**2 + y**2) / w0**2) * circ1
    I_gauss = np.abs(E_gauss)**2
    E_far_gauss = fftshift(fft2(ifftshift(E_gauss)))
    I_far_gauss = np.abs(E_far_gauss)**2
    
    total_power_gauss = np.sum(I_gauss)
    Dr_gauss = np.sum((x**2 + y**2) * I_gauss) / total_power_gauss
    total_power_far_gauss = np.sum(I_far_gauss)
    Drho_gauss = np.sum((fx**2 + fy**2) * I_far_gauss) / total_power_far_gauss

    M2 = np.sqrt(Drho / Drho_gauss)
    return M2, Dr, Drho, Dr_gauss, Drho_gauss, I_out, I_far, I_gauss, I_far_gauss


# --- JAX accelerated Physics Functions ---
try:
    import jax.numpy as jnp
    from jax.numpy.fft import fft2 as jfft2
    from jax.numpy.fft import ifft2 as jifft2
    from jax.numpy.fft import fftshift as jfftshift
    from jax.numpy.fft import ifftshift as jifftshift


    def angspec_prop_jax(u, dz, k_sq, four_pi_sq, f_sq_sum, N, p, wav):
        alpha = jnp.sqrt(k_sq - four_pi_sq * f_sq_sum) 
        f0 = (1/wav) * 1/jnp.sqrt(1 + (2*dz/(N*p))**2)
        LP = jnp.where(f_sq_sum <= f0**2, 1.0, 0.0)
        H = jnp.exp(1j * dz * alpha) * LP
        U = jfftshift(jfft2(jifftshift(u)))
        return jfftshift(jifft2(jifftshift(U * H)))
        

    def run_iteration_jax(E0, Mirror1, Mirror2, gain_profile, z, circ2, circ0, k_sq, four_pi_sq, f_sq_sum, N, p, wav):
        E0 /= jnp.max(jnp.abs(E0[:]))
        E0 = E0 * Mirror1
        E0 = angspec_prop_jax(E0, z, k_sq, four_pi_sq, f_sq_sum, N, p, wav)
        E0 = E0 * gain_profile
        E_out = E0 * (1 - circ2)
        E0 = E0 * Mirror2
        E0 = angspec_prop_jax(E0, z, k_sq, four_pi_sq, f_sq_sum, N, p, wav)
        E0 = E0 * gain_profile
        
        intensity = jnp.abs(E_out)**2 * circ0 * (1 - circ2)
        phase = jnp.angle(E_out) * circ0 * (1 - circ2)
        
        return E0, E_out, intensity, phase


    def calc_far_field_jax(E_out, x, y, fx, fy, D1, D2, circ1, N):
        I_out = jnp.abs(E_out)**2
        E_far = jfftshift(jfft2(jifftshift(E_out)))
        I_far = jnp.abs(E_far)**2
        
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

        w0 = jnp.mean(jnp.array([D1, D2]))
        E_gauss = jnp.exp(-(x**2 + y**2) / w0**2) * circ1
        I_gauss = jnp.abs(E_gauss)**2
        E_far_gauss = jfftshift(jfft2(jifftshift(E_gauss)))
        I_far_gauss = jnp.abs(E_far_gauss)**2
        
        total_power_gauss = jnp.sum(I_gauss)
        Dr_gauss = jnp.sum((x**2 + y**2) * I_gauss) / total_power_gauss
        total_power_far_gauss = jnp.sum(I_far_gauss)
        Drho_gauss = jnp.sum((fx**2 + fy**2) * I_far_gauss) / total_power_far_gauss

        M2 = jnp.sqrt(Drho / Drho_gauss)
        return M2, Dr, Drho, Dr_gauss, Drho_gauss, I_out, I_far, I_gauss, I_far_gauss

except ImportError:
    pass