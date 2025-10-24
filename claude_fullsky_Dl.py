#!/usr/bin/env python3
"""
Compute, smooth, and export angular power spectrum D_ℓ vs ℓ
Creates smooth power spectrum and exports to text file
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

def compute_power_spectrum(map_data, lmax=3000):
    """
    Compute angular power spectrum
    
    Returns:
    --------
    ell : array
        Multipole values
    cl : array
        C_ℓ values
    dl : array
        D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) values in µK²
    """
    print(f"\nComputing angular power spectrum up to ℓ_max = {lmax}...")
    
    # Compute C_ℓ
    cl = hp.anafast(map_data, lmax=lmax)
    ell = np.arange(len(cl))
    
    # Convert to D_ℓ = ℓ(ℓ+1)C_ℓ/(2π)
    dl = np.zeros_like(cl)
    dl[1:] = ell[1:] * (ell[1:] + 1) * cl[1:] / (2 * np.pi)
    dl[0] = 0  # Monopole is undefined
    
    print(f"Computed {len(ell)} multipoles")
    
    return ell, cl, dl

def bin_power_spectrum(ell, dl, bin_width=10):
    """
    Bin the power spectrum for smoother appearance
    
    Parameters:
    -----------
    ell : array
        Multipole values
    dl : array
        D_ℓ values
    bin_width : int
        Width of bins in ℓ space
    
    Returns:
    --------
    ell_binned : array
        Binned multipole centers
    dl_binned : array
        Binned D_ℓ values
    dl_std : array
        Standard deviation in each bin
    """
    print(f"\nBinning power spectrum with bin width Δℓ = {bin_width}...")
    
    max_ell = len(ell) - 1
    n_bins = max_ell // bin_width
    
    ell_binned = []
    dl_binned = []
    dl_std = []
    
    for i in range(n_bins):
        ell_min = i * bin_width
        ell_max = (i + 1) * bin_width
        
        if ell_max > max_ell:
            ell_max = max_ell
        
        # Skip first few multipoles (usually noisy)
        if ell_min < 2:
            continue
        
        mask = (ell >= ell_min) & (ell < ell_max)
        
        if np.sum(mask) > 0:
            ell_binned.append(np.mean(ell[mask]))
            dl_binned.append(np.mean(dl[mask]))
            dl_std.append(np.std(dl[mask]))
    
    print(f"Created {len(ell_binned)} bins")
    
    return np.array(ell_binned), np.array(dl_binned), np.array(dl_std)

def smooth_power_spectrum_gaussian(ell, dl, sigma=10):
    """
    Smooth power spectrum using Gaussian filter
    
    Parameters:
    -----------
    ell : array
        Multipole values
    dl : array
        D_ℓ values
    sigma : float
        Width of Gaussian kernel in ℓ space
    
    Returns:
    --------
    dl_smooth : array
        Smoothed D_ℓ values
    """
    print(f"\nSmoothing with Gaussian filter (σ = {sigma})...")
    
    # Skip first few multipoles
    start_idx = 2
    dl_smooth = np.copy(dl)
    dl_smooth[start_idx:] = gaussian_filter1d(dl[start_idx:], sigma=sigma)
    
    return dl_smooth

def smooth_power_spectrum_spline(ell, dl, smoothing=None, k=3):
    """
    Smooth power spectrum using spline interpolation
    
    Parameters:
    -----------
    ell : array
        Multipole values
    dl : array
        D_ℓ values
    smoothing : float or None
        Smoothing factor (larger = smoother). If None, auto-determined
    k : int
        Degree of spline (3 = cubic)
    
    Returns:
    --------
    dl_smooth : array
        Smoothed D_ℓ values
    """
    print(f"\nSmoothing with spline interpolation (k={k})...")
    
    # Skip first few multipoles and use subset for fitting
    start_idx = 2
    ell_fit = ell[start_idx:]
    dl_fit = dl[start_idx:]
    
    # Remove any NaN or inf values
    mask = np.isfinite(dl_fit)
    ell_fit = ell_fit[mask]
    dl_fit = dl_fit[mask]
    
    # Create spline
    if smoothing is None:
        smoothing = len(ell_fit) * np.std(dl_fit) * 0.1
    
    spline = UnivariateSpline(ell_fit, dl_fit, s=smoothing, k=k)
    
    dl_smooth = np.copy(dl)
    dl_smooth[start_idx:] = spline(ell[start_idx:])
    
    return dl_smooth

def export_power_spectrum(filename, ell, dl, dl_smooth=None, header_info=None):
    """
    Export power spectrum to text file
    
    Parameters:
    -----------
    filename : str
        Output filename
    ell : array
        Multipole values
    dl : array
        Raw D_ℓ values
    dl_smooth : array or None
        Smoothed D_ℓ values
    header_info : dict
        Additional information for header
    """
    print(f"\nExporting to {filename}...")
    
    # Prepare header
    header_lines = [
        "Angular Power Spectrum: D_ℓ vs ℓ",
        "D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in µK²",
        ""
    ]
    
    if header_info:
        for key, value in header_info.items():
            header_lines.append(f"{key}: {value}")
        header_lines.append("")
    
    # Column headers
    if dl_smooth is not None:
        header_lines.append("Columns: ℓ    D_ℓ(raw)    D_ℓ(smooth)")
        data = np.column_stack([ell, dl, dl_smooth])
        fmt = ['%6d', '%15.6e', '%15.6e']
    else:
        header_lines.append("Columns: ℓ    D_ℓ")
        data = np.column_stack([ell, dl])
        fmt = ['%6d', '%15.6e']
    
    header = '\n'.join(['# ' + line for line in header_lines])
    
    # Save to file
    np.savetxt(filename, data, fmt=fmt, header=header)
    print(f"Saved {len(ell)} data points")

def plot_comparison(ell, dl_raw, dl_smooth_list, smooth_labels, 
                   dl_binned=None, ell_binned=None):
    """
    Plot comparison of raw and smoothed power spectra
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Linear scale
    ax1.plot(ell, dl_raw, 'k-', alpha=0.3, linewidth=0.5, label='Raw')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (dl_smooth, label) in enumerate(zip(dl_smooth_list, smooth_labels)):
        ax1.plot(ell, dl_smooth, color=colors[i % len(colors)], 
                linewidth=2, label=label)
    
    if dl_binned is not None:
        ax1.plot(ell_binned, dl_binned, 'o', markersize=4, 
                alpha=0.6, label='Binned')
    
    ax1.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax1.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell/2\pi$ [µK$^2$]', fontsize=12)
    ax1.set_title('Angular Power Spectrum (Linear Scale)', fontsize=14)
    ax1.set_xlim(0, max(ell))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Log scale
    ax2.plot(ell[ell>0], dl_raw[ell>0], 'k-', alpha=0.3, 
            linewidth=0.5, label='Raw')
    
    for i, (dl_smooth, label) in enumerate(zip(dl_smooth_list, smooth_labels)):
        ax2.plot(ell[ell>0], dl_smooth[ell>0], color=colors[i % len(colors)], 
                linewidth=2, label=label)
    
    if dl_binned is not None:
        ax2.plot(ell_binned, dl_binned, 'o', markersize=4, 
                alpha=0.6, label='Binned')
    
    ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax2.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell/2\pi$ [µK$^2$]', fontsize=12)
    ax2.set_title('Angular Power Spectrum (Log Scale)', fontsize=14)
    ax2.set_xlim(2, max(ell))
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('power_spectrum_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot: power_spectrum_comparison.png")
    plt.close()

def main():
    print("="*70)
    print("Angular Power Spectrum Analysis")
    print("="*70)
    
    # Load map
    fits_file = "4e3_2048_50_50_ksz.fits"
    print(f"\nLoading: {fits_file}")
    map_data = hp.read_map(fits_file, verbose=False)
    
    nside = hp.npix2nside(len(map_data))
    map_stats = {
        'NSIDE': nside,
        'NPIX': len(map_data),
        'Mean': f'{np.mean(map_data):.6e} µK',
        'RMS': f'{np.std(map_data):.6e} µK',
        'Min': f'{np.min(map_data):.6e} µK',
        'Max': f'{np.max(map_data):.6e} µK'
    }
    
    print(f"NSIDE: {nside}, NPIX: {len(map_data):,}")
    
    # Compute power spectrum
    lmax = 3000
    ell, cl, dl = compute_power_spectrum(map_data, lmax=lmax)
    
    # Apply different smoothing methods
    print("\n" + "="*70)
    print("Applying Smoothing Methods")
    print("="*70)
    
    # Method 1: Binning
    ell_binned, dl_binned, dl_std = bin_power_spectrum(ell, dl, bin_width=10)
    
    # Method 2: Gaussian smoothing (light)
    dl_gauss_light = smooth_power_spectrum_gaussian(ell, dl, sigma=5)
    
    # Method 3: Gaussian smoothing (moderate)
    dl_gauss_mod = smooth_power_spectrum_gaussian(ell, dl, sigma=15)
    
    # Method 4: Spline smoothing
    dl_spline = smooth_power_spectrum_spline(ell, dl, smoothing=None)
    
    # Plot comparison
    print("\n" + "="*70)
    print("Creating Comparison Plot")
    print("="*70)
    
    plot_comparison(
        ell, dl,
        [dl_gauss_light, dl_gauss_mod, dl_spline],
        ['Gaussian (σ=5)', 'Gaussian (σ=15)', 'Spline'],
        dl_binned, ell_binned
    )
    
    # Export to text files
    print("\n" + "="*70)
    print("Exporting Data")
    print("="*70)
    
    # Export raw power spectrum
    export_power_spectrum(
        'power_spectrum_raw.txt',
        ell, dl,
        header_info=map_stats
    )
    
    # Export raw + smoothed
    export_power_spectrum(
        'power_spectrum_smooth_gaussian.txt',
        ell, dl, dl_gauss_mod,
        header_info={**map_stats, 'Smoothing': 'Gaussian (σ=15)'}
    )
    
    export_power_spectrum(
        'power_spectrum_smooth_spline.txt',
        ell, dl, dl_spline,
        header_info={**map_stats, 'Smoothing': 'Cubic spline'}
    )
    
    # Export binned power spectrum
    export_power_spectrum(
        'power_spectrum_binned.txt',
        ell_binned.astype(int), dl_binned,
        header_info={**map_stats, 'Binning': 'Δℓ = 10'}
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nGenerated files:")
    print("  1. power_spectrum_raw.txt              - Raw C_ℓ and D_ℓ")
    print("  2. power_spectrum_smooth_gaussian.txt  - Gaussian smoothed (σ=15)")
    print("  3. power_spectrum_smooth_spline.txt    - Spline smoothed")
    print("  4. power_spectrum_binned.txt           - Binned (Δℓ=10)")
    print("  5. power_spectrum_comparison.png       - Comparison plot")
    
    print("\nPower spectrum features:")
    print(f"  Peak at ℓ ~ {ell[np.argmax(dl[:100])]:.0f} (Doppler)")
    print(f"  D_ℓ(peak) ~ {np.max(dl[:100]):.2f} µK²")
    print(f"  D_ℓ(ℓ=3000) ~ {dl[3000]:.2f} µK²")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
