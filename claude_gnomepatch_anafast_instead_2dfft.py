#!/usr/bin/env python3
"""
Compute power spectrum from a patch using HEALPix masking
Much simpler and more reliable than FFT method!
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

def create_circular_mask(nside, ra_center, dec_center, radius_deg):
    """
    Create a circular mask for HEALPix map
    
    Parameters:
    -----------
    nside : int
        HEALPix NSIDE parameter
    ra_center, dec_center : float
        Center coordinates in degrees
    radius_deg : float
        Radius of circular patch in degrees
    
    Returns:
    --------
    mask : array
        Boolean mask (True = inside patch, False = outside)
    """
    npix = hp.nside2npix(nside)
    
    # Convert to theta, phi
    theta_center = np.radians(90 - dec_center)
    phi_center = np.radians(ra_center)
    
    # Get vector pointing to center
    vec = hp.ang2vec(theta_center, phi_center)
    
    # Query pixels within radius
    radius_rad = np.radians(radius_deg)
    pixels_in_patch = hp.query_disc(nside, vec, radius_rad)
    
    # Create mask
    mask = np.zeros(npix, dtype=bool)
    mask[pixels_in_patch] = True
    
    print(f"\nCreated circular mask:")
    print(f"  Center: RA={ra_center}°, Dec={dec_center}°")
    print(f"  Radius: {radius_deg}°")
    print(f"  Pixels in patch: {len(pixels_in_patch):,} ({100*len(pixels_in_patch)/npix:.2f}% of sky)")
    print(f"  Patch area: {len(pixels_in_patch) * hp.nside2pixarea(nside, degrees=True):.1f} deg²")
    
    return mask

def compute_patch_power_spectrum(map_data, mask, lmax=10000):
    """
    Compute power spectrum from masked map
    
    Parameters:
    -----------
    map_data : array
        Full HEALPix map
    mask : array
        Boolean mask (True = include, False = exclude)
    lmax : int
        Maximum multipole
    
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
        print(f"\nWarning: Requested lmax={lmax} exceeds maximum for NSIDE={nside}")
        print(f"         Setting lmax = {max_possible_lmax}")
        lmax = max_possible_lmax
    
    print(f"\nComputing power spectrum from masked map...")
    print(f"  lmax = {lmax}")
    
    # Create masked map (set pixels outside mask to UNSEEN)
    masked_map = np.copy(map_data)
    masked_map[~mask] = hp.UNSEEN
    
    # Statistics of patch
    patch_data = map_data[mask]
    print(f"\nPatch statistics:")
    print(f"  Mean: {np.mean(patch_data):.3f} µK")
    print(f"  RMS:  {np.std(patch_data):.3f} µK")
    print(f"  Min:  {np.min(patch_data):.3f} µK")
    print(f"  Max:  {np.max(patch_data):.3f} µK")
    
    # Compute power spectrum from masked map
    # Note: This includes the effect of the mask (window function)
    cl = hp.anafast(masked_map, lmax=lmax)
    ell = np.arange(len(cl))
    
    # Convert to D_ℓ
    dl = np.zeros_like(cl)
    dl[1:] = ell[1:] * (ell[1:] + 1) * cl[1:] / (2 * np.pi)
    
    print(f"\nComputed {len(ell)} multipoles")
    
    return ell, cl, dl

def export_power_spectrum(filename, ell, dl, ell_min=100, ell_max=10000, 
                         description=""):
    """
    Export power spectrum to text file in specified ℓ range
    
    Parameters:
    -----------
    filename : str
        Output filename
    ell : array
        Multipole values
    dl : array
        D_ℓ values
    ell_min, ell_max : int
        Range of multipoles to export
    description : str
        Description for header
    """
    # Select range
    mask = (ell >= ell_min) & (ell <= ell_max)
    ell_selected = ell[mask]
    dl_selected = dl[mask]
    
    print(f"\nExporting to {filename}...")
    print(f"  ℓ range: [{ell_min}, {ell_max}]")
    print(f"  Number of points: {len(ell_selected)}")
    
    # Prepare header
    header = f"""Angular Power Spectrum: D_ℓ vs ℓ
{description}
D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in µK²
ℓ range: [{ell_min}, {ell_max}]

Columns: ℓ    D_ℓ"""
    
    data = np.column_stack([ell_selected, dl_selected])
    np.savetxt(filename, data, fmt=['%6d', '%15.6e'], header=header)
    print(f"Saved!")

def plot_patch_vs_fullsky(ell_full, dl_full, ell_patch, dl_patch,
                          patch_info, ell_min=100, ell_max=10000):
    """
    Plot comparison of full sky and patch power spectra
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Select range for plotting
    mask_full = (ell_full >= ell_min) & (ell_full <= ell_max)
    mask_patch = (ell_patch >= ell_min) & (ell_patch <= ell_max)
    
    # Linear scale
    ax1.plot(ell_full[mask_full], dl_full[mask_full], 'b-', 
            linewidth=2, label='Full Sky', alpha=0.7)
    ax1.plot(ell_patch[mask_patch], dl_patch[mask_patch], 'r-', 
            linewidth=2, label=f'Patch ({patch_info})', alpha=0.7)
    ax1.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax1.set_ylabel(r'$D_\ell$ [µK$^2$]', fontsize=12)
    ax1.set_title(f'Power Spectrum: ℓ = {ell_min} to {ell_max} (Linear)', fontsize=14)
    ax1.set_xlim(ell_min, ell_max)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Log scale
    ax2.loglog(ell_full[mask_full], dl_full[mask_full], 'b-', 
              linewidth=2, label='Full Sky', alpha=0.7)
    ax2.loglog(ell_patch[mask_patch], dl_patch[mask_patch], 'r-', 
              linewidth=2, label=f'Patch ({patch_info})', alpha=0.7)
    ax2.set_xlabel(r'Multipole $\ell$', fontsize=12)
    ax2.set_ylabel(r'$D_\ell$ [µK$^2$]', fontsize=12)
    ax2.set_title(f'Power Spectrum: ℓ = {ell_min} to {ell_max} (Log-Log)', fontsize=14)
    ax2.set_xlim(ell_min, ell_max)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('power_spectrum_patch_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: power_spectrum_patch_comparison.png")
    plt.close()

def visualize_mask(map_data, mask, ra_center, dec_center):
    """
    Visualize the mask on the map
    """
    # Create masked map for visualization
    masked_map = np.copy(map_data)
    masked_map[~mask] = hp.UNSEEN
    
    fig = plt.figure(figsize=(16, 8))
    
    # Full sky with mask overlay
    hp.mollview(map_data, 
                title='Full Sky Map',
                unit='µK',
                cmap='RdBu_r',
                min=-50, max=50,
                sub=(1, 2, 1),
                hold=True)
    
    # Get pixels in mask and plot their positions
    nside = hp.npix2nside(len(map_data))
    pixels_in_mask = np.where(mask)[0]
    theta, phi = hp.pix2ang(nside, pixels_in_mask)
    
    # Plot circle showing mask boundary (approximate)
    hp.projplot(theta, phi, 'k.', markersize=0.1, lonlat=False, coord='C')
    
    # Masked map
    hp.mollview(masked_map, 
                title=f'Patch Only (RA={ra_center}°, Dec={dec_center}°)',
                unit='µK',
                cmap='RdBu_r',
                min=-50, max=50,
                sub=(1, 2, 2),
                hold=True)
    
    plt.savefig('mask_visualization.png', dpi=150, bbox_inches='tight')
    print("\nSaved: mask_visualization.png")
    plt.close()

def main():
    print("="*70)
    print("Patch Power Spectrum using HEALPix Masking")
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
    
    lmax = 10000
    max_possible_lmax = 3 * nside - 1
    if lmax > max_possible_lmax:
        lmax = max_possible_lmax
    
    print(f"\nComputing full-sky power spectrum up to ℓ = {lmax}...")
    cl_full = hp.anafast(map_data, lmax=lmax)
    ell_full = np.arange(len(cl_full))
    
    # Convert to D_ℓ
    dl_full = np.zeros_like(cl_full)
    dl_full[1:] = ell_full[1:] * (ell_full[1:] + 1) * cl_full[1:] / (2 * np.pi)
    
    print(f"Computed {len(ell_full)} multipoles")
    
    # Export full sky (ℓ = 100 to 10000)
    export_power_spectrum(
        'power_spectrum_fullsky_100_10k.txt',
        ell_full,
        dl_full,
        ell_min=100,
        ell_max=10000,
        description=f'Full sky HEALPix map (NSIDE={nside})'
    )
    
    # ========================================
    # PATCH POWER SPECTRUM
    # ========================================
    print("\n" + "="*70)
    print("PATCH POWER SPECTRUM")
    print("="*70)
    
    # Define patch (circular region)
    ra_center = 180.0    # degrees
    dec_center = 30.0    # degrees
    radius_deg = 16.0    # 32° diameter patch
    
    # Create mask
    mask = create_circular_mask(nside, ra_center, dec_center, radius_deg)
    
    # Visualize mask
    visualize_mask(map_data, mask, ra_center, dec_center)
    
    # Compute power spectrum from patch
    ell_patch, cl_patch, dl_patch = compute_patch_power_spectrum(
        map_data, mask, lmax=lmax
    )
    
    # Export patch (ℓ = 100 to 10000)
    patch_info = f"R={radius_deg}°, RA={ra_center}°, Dec={dec_center}°"
    export_power_spectrum(
        'power_spectrum_patch_100_10k.txt',
        ell_patch,
        dl_patch,
        ell_min=100,
        ell_max=10000,
        description=f'Circular patch: {patch_info}'
    )
    
    # ========================================
    # COMPARISON PLOT
    # ========================================
    print("\n" + "="*70)
    print("CREATING COMPARISON")
    print("="*70)
    
    plot_patch_vs_fullsky(ell_full, dl_full, ell_patch, dl_patch,
                         f"{2*radius_deg}° diameter",
                         ell_min=100, ell_max=10000)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nGenerated files:")
    print("  1. power_spectrum_fullsky_100_10k.txt   - Full sky (ℓ=100-10000)")
    print("  2. power_spectrum_patch_100_10k.txt     - Patch (ℓ=100-10000)")
    print("  3. power_spectrum_patch_comparison.png  - Comparison plot")
    print("  4. mask_visualization.png               - Shows patch location")
    
    print(f"\nPower spectrum at selected ℓ:")
    ell_test = [100, 500, 1000, 3000, 10000]
    for ell_val in ell_test:
        if ell_val < len(dl_full):
            idx = ell_val
            print(f"  ℓ={ell_val:5d}: Full sky D_ℓ = {dl_full[idx]:.3f} µK², " + 
                  f"Patch D_ℓ = {dl_patch[idx]:.3f} µK²")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    main()
