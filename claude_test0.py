#!/usr/bin/env python3
"""
HEALPix kSZ Map Explorer and Visualizer
Explores and plots the Alvarez 2016 kSZ maps
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
import os

# File path - adjust if needed
FITS_FILE = "~/Desktop/Project2/Plots/Alvarez2016_Maps/"  # Add filename

def explore_fits_file(filename):
    """Explore the structure and contents of the FITS file"""
    print("="*70)
    print(f"Exploring FITS file: {filename}")
    print("="*70)
    
    # Open and inspect the FITS file
    with fits.open(filename) as hdul:
        print(f"\nNumber of HDUs: {len(hdul)}")
        print("\nHDU Information:")
        hdul.info()
        
        # Explore each HDU
        for i, hdu in enumerate(hdul):
            print(f"\n{'='*70}")
            print(f"HDU {i}: {hdu.name}")
            print(f"{'='*70}")
            
            # Print header info
            print("\nKey Header Keywords:")
            for keyword in ['NSIDE', 'ORDERING', 'COORDSYS', 'PIXTYPE', 
                          'OBJECT', 'EXTNAME', 'TTYPE1', 'TUNIT1']:
                if keyword in hdu.header:
                    print(f"  {keyword:15s}: {hdu.header[keyword]}")
            
            # If it's a binary table, show column info
            if isinstance(hdu, fits.BinTableHDU):
                print(f"\nColumns: {hdu.columns.names}")
                print(f"Number of rows: {len(hdu.data)}")
                
                # Show data statistics for each column
                for col in hdu.columns.names:
                    data = hdu.data[col]
                    if np.issubdtype(data.dtype, np.number):
                        print(f"\n  Column: {col}")
                        print(f"    Shape: {data.shape}")
                        print(f"    Min:   {np.min(data):.6e}")
                        print(f"    Max:   {np.max(data):.6e}")
                        print(f"    Mean:  {np.mean(data):.6e}")
                        print(f"    Std:   {np.std(data):.6e}")
    
    return

def plot_healpix_map(filename, column=0, output_prefix="ksz_map"):
    """
    Plot HEALPix map from FITS file
    
    Parameters:
    -----------
    filename : str
        Path to FITS file
    column : int or str
        Column index or name to plot
    output_prefix : str
        Prefix for output files
    """
    print(f"\nReading map data...")
    
    # Read the map
    map_data = hp.read_map(filename, field=column, verbose=True)
    
    # Get NSIDE
    nside = hp.npix2nside(len(map_data))
    print(f"\nNSIDE: {nside}")
    print(f"NPIX:  {len(map_data)}")
    print(f"Resolution: ~{hp.nside2resol(nside, arcmin=True):.2f} arcmin")
    
    # Statistics
    print(f"\nMap Statistics:")
    print(f"  Min:  {np.min(map_data):.6e} µK")
    print(f"  Max:  {np.max(map_data):.6e} µK")
    print(f"  Mean: {np.mean(map_data):.6e} µK")
    print(f"  RMS:  {np.std(map_data):.6e} µK")
    
    # Create figure with multiple projections
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Mollweide projection (like Figure 6)
    plt.subplot(2, 2, 1)
    hp.mollview(map_data, 
                title='kSZ Temperature Map (Mollweide)',
                unit='µK',
                cmap='RdBu_r',  # Red-Blue colormap (reversed)
                min=-50, max=50,  # Adjust based on actual range
                hold=True,
                sub=(2, 2, 1))
    
    # 2. Orthographic projection (North pole)
    hp.orthview(map_data,
                title='kSZ Map (North Pole)',
                unit='µK',
                cmap='RdBu_r',
                min=-50, max=50,
                half_sky=True,
                hold=True,
                sub=(2, 2, 2))
    
    # 3. Cartesian projection
    hp.cartview(map_data,
                title='kSZ Map (Cartesian)',
                unit='µK',
                cmap='RdBu_r',
                min=-50, max=50,
                hold=True,
                sub=(2, 2, 3))
    
    # 4. Power spectrum
    plt.subplot(2, 2, 4)
    cl = hp.anafast(map_data, lmax=3000)
    ell = np.arange(len(cl))
    plt.plot(ell, ell * (ell + 1) * cl / (2 * np.pi), 'b-', linewidth=1)
    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell / 2\pi$ [µK$^2$]')
    plt.title('Angular Power Spectrum')
    plt.xlim(0, 3000)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overview.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved overview figure: {output_prefix}_overview.png")
    
    # Create a separate high-res Mollweide view (like Figure 6)
    plt.figure(figsize=(16, 8))
    hp.mollview(map_data,
                title='kSZ Temperature Fluctuations (Full Sky)',
                unit='µK',
                cmap='RdBu_r',
                min=-50, max=50,  # Adjust these limits based on your data
                format='%.1f')
    plt.savefig(f'{output_prefix}_mollweide.png', dpi=200, bbox_inches='tight')
    print(f"Saved Mollweide figure: {output_prefix}_mollweide.png")
    
    # Histogram of values
    plt.figure(figsize=(10, 6))
    plt.hist(map_data, bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('Temperature (µK)')
    plt.ylabel('Number of pixels')
    plt.title('Distribution of kSZ Temperature Fluctuations')
    plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_prefix}_histogram.png', dpi=150, bbox_inches='tight')
    print(f"Saved histogram: {output_prefix}_histogram.png")
    
    return map_data, cl

def plot_small_patch(map_data, ra_center=0, dec_center=0, size=32):
    """
    Plot a small patch similar to Figure 7 in the paper
    
    Parameters:
    -----------
    map_data : array
        HEALPix map data
    ra_center : float
        RA center in degrees
    dec_center : float
        Dec center in degrees  
    size : float
        Size of patch in degrees
    """
    plt.figure(figsize=(10, 10))
    hp.gnomview(map_data,
                rot=(ra_center, dec_center, 0),
                xsize=800,
                reso=2.5,  # arcmin resolution
                title=f'{size}° × {size}° Patch (RA={ra_center}°, Dec={dec_center}°)',
                unit='µK',
                cmap='RdBu_r',
                min=-50, max=50)
    plt.savefig(f'ksz_patch_{ra_center}_{dec_center}.png', dpi=150, bbox_inches='tight')
    print(f"Saved patch figure: ksz_patch_{ra_center}_{dec_center}.png")

def main():
    # List files in directory
    base_dir = os.path.expanduser("~/Desktop/Project2/Plots/Alvarez2016_Maps/")
    print(f"\nLooking for FITS files in: {base_dir}")
    
    if os.path.exists(base_dir):
        fits_files = [f for f in os.listdir(base_dir) if f.endswith('.fits')]
        print(f"\nFound {len(fits_files)} FITS file(s):")
        for i, f in enumerate(fits_files):
            size_mb = os.path.getsize(os.path.join(base_dir, f)) / (1024**2)
            print(f"  {i}: {f} ({size_mb:.1f} MB)")
        
        if fits_files:
            # Use the first FITS file found
            fits_file = os.path.join(base_dir, fits_files[0])
            
            # First explore the file structure
            explore_fits_file(fits_file)
            
            # Then plot it
            print("\n" + "="*70)
            print("PLOTTING MAP")
            print("="*70)
            
            map_data, cl = plot_healpix_map(fits_file)
            
            # Optionally plot a small patch (like Figure 7)
            # Uncomment to create:
            # plot_small_patch(map_data, ra_center=180, dec_center=30, size=32)
            
            print("\n" + "="*70)
            print("DONE! Check the output PNG files.")
            print("="*70)
        else:
            print("\nNo FITS files found. Please check the directory path.")
    else:
        print(f"\nDirectory not found: {base_dir}")
        print("Please update the FITS_FILE path in the script.")

if __name__ == "__main__":
    main()
