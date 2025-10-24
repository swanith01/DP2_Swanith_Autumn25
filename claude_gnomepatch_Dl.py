#!/usr/bin/env python3
"""
Compute power spectrum from gnomonic patch and full sky
Extends to ℓ_max = 10^4 where possible
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.fft import fft2, fftshift, fftfreq
from scipy.ndimage import gaussian_filter

def compute_full_sky_power_spectrum(map_data, lmax=10000):
    """
    Compute power spectrum from full HEALPix map
    
    Parameters:
    -----------
    map_data : array
        Full HEALPix map
    lmax : int
        Maximum multipole (limited by NSIDE: lmax ≤ 3*NSIDE)
    
    Returns:
    --------
    ell : array
        Multipole values
    cl : array
        C_ℓ values
    dl : array
        D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) values in µK²
    """
    nside = hp.npix2nside(len(map_data))
    max_possible_lmax = 3 * nside - 1
    
    if lmax > max_possible_lmax:
        print(f"\nWarning: Requested lmax={lmax} exceeds maximum possible for NSIDE={nside}")
        print(f"         Maximum lmax = 3*NSIDE - 1 = {max_possible_lmax}")
        lmax = max_possible_lmax
    
    print(f"\nComputing full-sky power spectrum up to ℓ_max = {lmax}...")
    print(f"NSIDE = {nside}, Max possible ℓ = {max_possible_lmax}")
    
    # Compute C_ℓ using HEALPix
    cl = hp.anafast(map_data, lmax=lmax)
    ell = np.arange(len(cl))
    
    # Convert to D_ℓ
    dl = np.zeros_like(cl)
    dl[1:] = ell[1:] * (ell[1:] + 1) * cl[1:] / (2 * np.pi)
    
    print(f"Computed {len(ell)} multipoles")
    print(f"D_ℓ range: [{np.min(dl[2:]):.3e}, {np.max(dl):.3e}] µK²")
    
    return ell, cl, dl

def extract_gnomonic_patch(map_data, ra_center, dec_center, size_deg, npix):
    """
    Extract a gnomonic (flat-sky) patch from HEALPix map
    
    Parameters:
    -----------
    map_data : array
        Full HEALPix map
    ra_center, dec_center : float
        Center coordinates in degrees
    size_deg : float
        Size of patch in degrees
    npix : int
        Number of pixels on each side
    
    Returns:
    --------
    patch : 2D array
        Extracted patch (npix × npix)
    pixel_size_arcmin : float
        Size of each pixel in arcminutes
    """
    print(f"\nExtracting {size_deg}° × {size_deg}° patch...")
    print(f"Center: RA={ra_center}°, Dec={dec_center}°")
    print(f"Resolution: {npix} × {npix} pixels")
    
    nside = hp.npix2nside(len(map_data))
    
    # Convert to colatitude and longitude
    theta_center = np.radians(90 - dec_center)
    phi_center = np.radians(ra_center)
    
    # Create grid in tangent plane
    extent_deg = size_deg / 2
    extent_rad = np.radians(extent_deg)
    
    # Linear grid in radians
    x = np.linspace(-extent_rad, extent_rad, npix)
    y = np.linspace(-extent_rad, extent_rad, npix)
    X, Y = np.meshgrid(x, y)
    
    # Gnomonic projection (tangent plane)
    # For small angles: Δθ ≈ y, Δφ ≈ x/cos(θ)
    theta_grid = theta_center + Y
    phi_grid = phi_center + X / np.sin(theta_center)
    
    # Interpolate values from HEALPix map
    patch = np.zeros((npix, npix))
    for i in range(npix):
        for j in range(npix):
            if 0 <= theta_grid[i, j] <= np.pi:
                patch[i, j] = hp.get_interp_val(map_data, theta_grid[i, j], phi_grid[i, j])
            else:
                patch[i, j] = np.nan
    
    pixel_size_arcmin = (size_deg * 60) / npix
    
    print(f"Pixel size: {pixel_size_arcmin:.3f} arcmin")
    print(f"Patch statistics:")
    print(f"  Mean: {np.nanmean(patch):.3f} µK")
    print(f"  RMS:  {np.nanstd(patch):.3f} µK")
    
    return patch, pixel_size_arcmin

def compute_2d_power_spectrum(patch, pixel_size_arcmin):
    """
    Compute 2D power spectrum from flat-sky patch using FFT
    
    Parameters:
    -----------
    patch : 2D array
        Temperature map patch
    pixel_size_arcmin : float
        Pixel size in arcminutes
    
    Returns:
    --------
    ell_2d : 2D array
        Multipole values for each Fourier mode
    power_2d : 2D array
        2D power spectrum
    """
    print(f"\nComputing 2D power spectrum from patch...")
    
    # Remove NaN values by setting to mean
    patch_clean = np.copy(patch)
    patch_clean[np.isnan(patch)] = np.nanmean(patch)
    
    # Subtract mean (monopole)
    patch_clean -= np.mean(patch_clean)
    
    # Apply apodization window to reduce edge effects
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    patch_windowed = patch_clean * window
    
    # Compute 2D FFT
    fft_patch = fft2(patch_windowed)
    fft_patch = fftshift(fft_patch)
    
    # Compute power spectrum
    power_2d = np.abs(fft_patch)**2
    
    # Get frequency grid
    npix = patch.shape[0]
    pixel_size_rad = np.radians(pixel_size_arcmin / 60.0)
    
    # Frequency corresponds to multipole ℓ
    freq_x = fftshift(fftfreq(npix, d=pixel_size_rad))
    freq_y = fftshift(fftfreq(npix, d=pixel_size_rad))
    
    # Convert to ℓ (angular frequency)
    # ℓ = 2π * spatial_frequency
    ell_x, ell_y = np.meshgrid(freq_x, freq_y)
    ell_2d = np.sqrt(ell_x**2 + ell_y**2)
    
    # Normalize power spectrum
    # Account for pixel size and number of pixels
    power_2d *= pixel_size_rad**2
    
    print(f"2D power spectrum computed")
    print(f"ℓ range: [{np.min(ell_2d[ell_2d>0]):.1f}, {np.max(ell_2d):.1f}]")
    
    return ell_2d, power_2d

def azimuthal_average(ell_2d, power_2d, nbins=100):
    """
    Compute azimuthally averaged 1D power spectrum
    
    Parameters:
    -----------
    ell_2d : 2D array
        Multipole values
    power_2d : 2D array
        2D power spectrum
    nbins : int
        Number of bins for averaging
    
    Returns:
    --------
    ell_1d : array
        Binned multipole values
    cl_1d : array
        Azimuthally averaged C_ℓ
    dl_1d : array
        D_ℓ values
    """
    print(f"\nAzimuthally averaging to 1D power spectrum...")
    
    # Create bins in ℓ space
    ell_max = np.max(ell_2d)
    ell_min = np.min(ell_2d[ell_2d > 0])
    
    # Use logarithmic binning for better sampling
    ell_bins = np.logspace(np.log10(ell_min), np.log10(ell_max), nbins)
    
    ell_1d = []
    cl_1d = []
    
    for i in range(len(ell_bins) - 1):
        mask = (ell_2d >= ell_bins[i]) & (ell_2d < ell_bins[i+1])
        
        if np.sum(mask) > 0:
            ell_1d.append(np.mean(ell_2d[mask]))
            cl_1d.append(np.mean(power_2d[mask]))
    
    ell_1d = np.array(ell_1d)
    cl_1d = np.array(cl_1d)
    
    # Convert to D_ℓ
    dl_1d = ell_1d * (ell_1d + 1) * cl_1d / (2 * np.pi)
    
    print(f"Created 1D power spectrum with {len(ell_1d)} bins")
    print(f"ℓ range: [{ell_1d[0]:.1f}, {ell_1d[-1]:.1f}]")
    
    return ell_1d, cl_1d, dl_1d

def export_power_spectrum(filename, ell, dl, description):
    """Export power spectrum to text file"""
    header = f"""Angular Power Spectrum: D_ℓ vs ℓ
{description}
D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in µK²

Columns: ℓ    D_ℓ"""
    
    data = np.column_stack([ell, dl])
    np.savetxt(filename, data, fmt=['%12.6f', '%15.6e'], header=header)
    print(f"\nExported: {filename}")

def plot_comparison(ell_full, dl_full, ell_patch, dl_patch, 
                   ra_center, dec_center, size_deg):
    """Plot comparison of full sky and patch power spectra"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Linear scale
    ax1.plot(ell_full, dl_full, 'b-', linewidth=2, label='Full Sky', alpha=0.7)
    ax1.plot(ell_patch, dl_patch, 'r-', linewidth=2, 
            label=f'Patch ({size_deg}° × {size_deg}°)', alpha=0.7)
    ax1.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax1.set_ylabel(r'$D_\ell$ [µK$^2$]', fontsize=12)
    ax1.set_title(f'Power Spectrum Comparison (Linear)\n' + 
                 f'Patch center: RA={ra_center}°, Dec={dec_center}°', fontsize=14)
    ax1.set_xlim(0, min(5000, max(ell_full)))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Log scale
    mask_full = ell_full > 1
    mask_patch = ell_patch > 1
    
    ax2.loglog(ell_full[mask_full], dl_full[mask_full], 'b-', 
              linewidth=2, label='Full Sky', alpha=0.7)
    ax2.loglog(ell_patch[mask_patch], dl_patch[mask_patch], 'r-', 
              linewidth=2, label=f'Patch ({size_deg}° × {size_deg}°)', alpha=0.7)
    ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax2.set_ylabel(r'$D_\ell$ [µK$^2$]', fontsize=12)
    ax2.set_title('Power Spectrum Comparison (Log-Log)', fontsize=14)
    ax2.set_xlim(2, max(ell_full))
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('power_spectrum_fullsky_vs_patch.png', dpi=150, bbox_inches='tight')
    print("\nSaved: power_spectrum_fullsky_vs_patch.png")
    plt.close()

def main():
    print("="*70)
    print("Full Sky vs Patch Power Spectrum Analysis")
    print("="*70)
    
    # Load map
    fits_file = "4e3_2048_50_50_ksz.fits"
    print(f"\nLoading: {fits_file}")
    map_data = hp.read_map(fits_file, verbose=False)
    
    nside = hp.npix2nside(len(map_data))
    print(f"NSIDE: {nside}, NPIX: {len(map_data):,}")
    
    # ========================================
    # FULL SKY POWER SPECTRUM
    # ========================================
    print("\n" + "="*70)
    print("FULL SKY POWER SPECTRUM")
    print("="*70)
    
    lmax_full = 10000
    ell_full, cl_full, dl_full = compute_full_sky_power_spectrum(map_data, lmax=lmax_full)
    
    export_power_spectrum(
        'power_spectrum_fullsky_10k.txt',
        ell_full,
        dl_full,
        f'Full sky HEALPix map (NSIDE={nside}, lmax={len(ell_full)-1})'
    )
    
    # ========================================
    # PATCH POWER SPECTRUM
    # ========================================
    print("\n" + "="*70)
    print("PATCH POWER SPECTRUM")
    print("="*70)
    
    # Patch parameters (like Figure 7)
    ra_center = 180.0  # degrees
    dec_center = 30.0  # degrees
    size_deg = 32.0    # 32° × 32° patch
    npix_patch = 2048  # High resolution for reaching high ℓ
    
    # Extract patch
    patch, pixel_size_arcmin = extract_gnomonic_patch(
        map_data, ra_center, dec_center, size_deg, npix_patch
    )
    
    # Compute 2D power spectrum
    ell_2d, power_2d = compute_2d_power_spectrum(patch, pixel_size_arcmin)
    
    # Azimuthally average to 1D
    ell_patch, cl_patch, dl_patch = azimuthal_average(ell_2d, power_2d, nbins=200)
    
    export_power_spectrum(
        'power_spectrum_patch.txt',
        ell_patch,
        dl_patch,
        f'Gnomonic patch ({size_deg}° × {size_deg}°, {npix_patch}×{npix_patch} pixels)\n' +
        f'Center: RA={ra_center}°, Dec={dec_center}°\n' +
        f'Pixel size: {pixel_size_arcmin:.3f} arcmin'
    )
    
    # ========================================
    # COMPARISON PLOT
    # ========================================
    print("\n" + "="*70)
    print("CREATING COMPARISON")
    print("="*70)
    
    plot_comparison(ell_full, dl_full, ell_patch, dl_patch,
                   ra_center, dec_center, size_deg)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nGenerated files:")
    print("  1. power_spectrum_fullsky_10k.txt         - Full sky (ℓ up to ~12k)")
    print("  2. power_spectrum_patch.txt               - 32° patch (ℓ up to ~10k)")
    print("  3. power_spectrum_fullsky_vs_patch.png    - Comparison plot")
    
    print(f"\nFull sky power spectrum:")
    print(f"  ℓ_max = {len(ell_full)-1}")
    print(f"  Peak ℓ ~ {ell_full[np.argmax(dl_full[:100])]:.0f}")
    print(f"  D_ℓ(peak) ~ {np.max(dl_full[:100]):.2f} µK²")
    
    print(f"\nPatch power spectrum:")
    print(f"  ℓ_max ~ {ell_patch[-1]:.0f}")
    print(f"  ℓ_min ~ {ell_patch[0]:.0f}")
    print(f"  Number of ℓ bins: {len(ell_patch)}")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
