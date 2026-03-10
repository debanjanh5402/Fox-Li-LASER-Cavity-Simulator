import jax.numpy as jnp

def _angspec_prop_core(u, dz, k_sq, four_pi_sq, f_sq_sum, N, p, wav):
    alpha = jnp.sqrt(k_sq - four_pi_sq * f_sq_sum) 
    f0 = (1/wav) * 1/jnp.sqrt(1 + (2*dz/(N*p))**2)
    LP = jnp.where(f_sq_sum <= f0**2, 1.0, 0.0)
    H = jnp.exp(1j * dz * alpha) * LP
    U = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(u)))
    u_out = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(U * H)))
    return u_out
    
def _run_iteration_core(E0, Mirror1, Mirror2, gain_profile, z, circ2, circ0, k_sq, four_pi_sq, f_sq_sum, N, p, wav):
    E0 /= jnp.max(jnp.abs(E0[:]))
    
    prop_fn = lambda u, dz: _angspec_prop_core(u, dz, k_sq, four_pi_sq, f_sq_sum, N, p, wav)

    E1 = E0 * Mirror1
    E1_prop = prop_fn(E1, z)
    E2 = E1_prop * Mirror2
    E2_prop = prop_fn(E2, z)
    
    E0_next = E2_prop * gain_profile
    E_out = E1_prop * (1 - circ2)
    
    intensity = jnp.abs(E_out)**2 * circ0 * (1 - circ2)
    phase = jnp.angle(E_out) * circ0 * (1 - circ2)
    
    return E0_next, E_out, intensity, phase