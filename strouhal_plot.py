#!/usr/bin/env python3
"""
Strouhal Number Analysis Tool

A standalone module for converting FFT frequency data to Strouhal numbers
and generating plots. Can be used independently or in conjunction with
fft_analysis.py.

The Strouhal number (St) is a dimensionless quantity in fluid mechanics:
    St = f * d / v

where:
    f = frequency (Hz)
    d = characteristic length, e.g. cylinder diameter (m)
    v = flow velocity (m/s)

Common Strouhal numbers:
    - Vortex shedding from cylinder: St ≈ 0.2 (for Re 250 to 2×10^5)
    - Human walking: St ≈ 0.2
    - Fish swimming: St ≈ 0.25-0.35

Usage:
    # Interactive mode - prompts for velocity and diameter
    python strouhal_plot.py fft_data.csv

    # Command line arguments
    python strouhal_plot.py fft_data.csv --velocity 10 --diameter 0.05

    # Process audio file directly
    python strouhal_plot.py --audio recording.m4a --velocity 10 --diameter 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_fft_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load FFT data from a CSV file.

    Expected format: two columns (frequency, magnitude) with optional header.

    Args:
        filepath: Path to CSV file

    Returns:
        Tuple of (frequencies, magnitudes)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Try loading with header first
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data.ndim == 1 or data.shape[1] < 2:
            # Might not have header, try again
            data = np.genfromtxt(filepath, delimiter=',')
    except Exception:
        # Try without assuming comma delimiter
        data = np.loadtxt(filepath)

    if data.ndim == 1:
        raise ValueError("CSV must have at least two columns (frequency, magnitude)")

    frequencies = data[:, 0]
    magnitudes = data[:, 1]

    return frequencies, magnitudes


def frequency_to_strouhal(
    frequencies: np.ndarray,
    velocity: float,
    diameter: float
) -> np.ndarray:
    """
    Convert frequencies to Strouhal numbers.

    St = f * d / v

    Args:
        frequencies: Array of frequencies in Hz
        velocity: Flow velocity in m/s
        diameter: Characteristic length in m

    Returns:
        Array of Strouhal numbers
    """
    if velocity <= 0:
        raise ValueError("Velocity must be positive")
    if diameter <= 0:
        raise ValueError("Diameter/characteristic length must be positive")

    return (frequencies * diameter) / velocity


def strouhal_to_frequency(
    strouhal: np.ndarray,
    velocity: float,
    diameter: float
) -> np.ndarray:
    """
    Convert Strouhal numbers back to frequencies.

    f = St * v / d

    Args:
        strouhal: Array of Strouhal numbers
        velocity: Flow velocity in m/s
        diameter: Characteristic length in m

    Returns:
        Array of frequencies in Hz
    """
    if velocity <= 0:
        raise ValueError("Velocity must be positive")
    if diameter <= 0:
        raise ValueError("Diameter/characteristic length must be positive")

    return (strouhal * velocity) / diameter


def get_parameters_interactive() -> tuple[float, float]:
    """
    Interactively prompt user for velocity and diameter.

    Returns:
        Tuple of (velocity, diameter)
    """
    print("\n" + "=" * 50)
    print("Strouhal Number Parameters")
    print("=" * 50)

    # Velocity input
    while True:
        try:
            v_input = input("Enter flow velocity [m/s]: ").strip()
            velocity = float(v_input)
            if velocity <= 0:
                print("  Velocity must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  Invalid number. Please enter a numeric value.")

    # Diameter input
    while True:
        try:
            d_input = input("Enter characteristic length/diameter [m]: ").strip()
            diameter = float(d_input)
            if diameter <= 0:
                print("  Diameter must be positive. Try again.")
                continue
            break
        except ValueError:
            print("  Invalid number. Please enter a numeric value.")

    print(f"\nUsing: v = {velocity} m/s, d = {diameter} m")
    print("=" * 50 + "\n")

    return velocity, diameter


def plot_strouhal_spectrum(
    strouhal: np.ndarray,
    magnitude: np.ndarray,
    velocity: float,
    diameter: float,
    title: str | None = None,
    output_path: str | None = None,
    log_scale: bool = False,
    show_reference_lines: bool = True
) -> None:
    """
    Plot magnitude spectrum against Strouhal number.

    Args:
        strouhal: Array of Strouhal numbers
        magnitude: Magnitude values (typically in dB)
        velocity: Flow velocity (for annotation)
        diameter: Characteristic length (for annotation)
        title: Plot title (auto-generated if None)
        output_path: Save to file if provided
        log_scale: Use log scale for x-axis
        show_reference_lines: Show common Strouhal reference values
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter out zero/negative Strouhal values for cleaner plot
    valid_mask = strouhal > 0
    st_valid = strouhal[valid_mask]
    mag_valid = magnitude[valid_mask]

    ax.plot(st_valid, mag_valid, linewidth=0.5, color='steelblue')

    # Reference lines for common Strouhal numbers
    if show_reference_lines:
        reference_lines = [
            (0.2, 'Cylinder vortex shedding (St ≈ 0.2)'),
            (0.3, 'Fish swimming (St ≈ 0.25-0.35)'),
        ]
        for st_ref, label in reference_lines:
            if st_valid.min() <= st_ref <= st_valid.max():
                ax.axvline(st_ref, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.annotate(
                    label,
                    xy=(st_ref, ax.get_ylim()[1]),
                    xytext=(5, -5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7,
                    rotation=90,
                    va='top'
                )

    ax.set_xlabel('Strouhal Number (St = fd/v)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)

    if title is None:
        title = f'Strouhal Spectrum\n(v = {velocity} m/s, d = {diameter} m)'
    ax.set_title(title, fontsize=14)

    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_xscale('log')

    # Add parameter annotation box
    param_text = f'v = {velocity} m/s\nd = {diameter} m'
    ax.annotate(
        param_text,
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        fontsize=9,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def analyze_from_audio(
    audio_path: str,
    velocity: float,
    diameter: float,
    window: str = 'hann',
    output_path: str | None = None,
    log_scale: bool = False
) -> None:
    """
    Load audio file, compute FFT, and plot Strouhal spectrum.

    This function integrates with fft_analysis.py for a complete pipeline.

    Args:
        audio_path: Path to audio file
        velocity: Flow velocity in m/s
        diameter: Characteristic length in m
        window: FFT window function
        output_path: Save plot to file if provided
        log_scale: Use log scale for x-axis
    """
    # Import from companion module
    from fft_analysis import load_audio, compute_fft

    print(f"Loading audio: {audio_path}")
    samples, sample_rate = load_audio(audio_path)

    print(f"Computing FFT with {window} window...")
    frequencies, magnitude_db = compute_fft(samples, sample_rate, window)

    print(f"Converting to Strouhal numbers (v={velocity} m/s, d={diameter} m)...")
    strouhal = frequency_to_strouhal(frequencies, velocity, diameter)

    # Find peak
    peak_idx = np.argmax(magnitude_db[1:]) + 1
    peak_freq = frequencies[peak_idx]
    peak_st = strouhal[peak_idx]
    print(f"  Peak frequency: {peak_freq:.2f} Hz")
    print(f"  Peak Strouhal: {peak_st:.4f}")

    title = f'Strouhal Spectrum - {Path(audio_path).name}'
    plot_strouhal_spectrum(
        strouhal, magnitude_db,
        velocity, diameter,
        title=title,
        output_path=output_path,
        log_scale=log_scale
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convert FFT frequency data to Strouhal numbers and plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From CSV file (interactive parameter input)
  %(prog)s fft_data.csv

  # From CSV with parameters
  %(prog)s fft_data.csv -v 10 -d 0.05

  # Directly from audio file
  %(prog)s --audio recording.m4a -v 15 -d 0.02

  # Save output
  %(prog)s data.csv -v 10 -d 0.05 -o strouhal_plot.png
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'csv_file',
        nargs='?',
        help='CSV file with frequency,magnitude columns'
    )
    input_group.add_argument(
        '--audio', '-a',
        help='Process audio file directly (requires fft_analysis.py)'
    )

    # Strouhal parameters
    parser.add_argument(
        '--velocity', '-v',
        type=float,
        help='Flow velocity in m/s (prompted if not provided)'
    )
    parser.add_argument(
        '--diameter', '-d',
        type=float,
        help='Characteristic length/diameter in m (prompted if not provided)'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Save plot to file instead of displaying'
    )
    parser.add_argument(
        '--log-scale',
        action='store_true',
        help='Use logarithmic scale for Strouhal axis'
    )
    parser.add_argument(
        '--no-reference',
        action='store_true',
        help='Hide reference Strouhal lines'
    )
    parser.add_argument(
        '--window', '-w',
        choices=['hann', 'hamming', 'blackman', 'bartlett', 'rectangular'],
        default='hann',
        help='Window function for FFT when using --audio (default: hann)'
    )

    args = parser.parse_args()

    # Get velocity and diameter (interactively if not provided)
    if args.velocity is None or args.diameter is None:
        try:
            velocity, diameter = get_parameters_interactive()
            if args.velocity is not None:
                velocity = args.velocity
            if args.diameter is not None:
                diameter = args.diameter
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            sys.exit(0)
    else:
        velocity = args.velocity
        diameter = args.diameter

    # Process based on input type
    if args.audio:
        try:
            analyze_from_audio(
                args.audio,
                velocity,
                diameter,
                window=args.window,
                output_path=args.output,
                log_scale=args.log_scale
            )
        except ImportError:
            print("Error: Could not import fft_analysis.py", file=sys.stderr)
            print("Make sure fft_analysis.py is in the same directory.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Load from CSV
        try:
            frequencies, magnitudes = load_fft_csv(args.csv_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        strouhal = frequency_to_strouhal(frequencies, velocity, diameter)

        # Find and report peak
        peak_idx = np.argmax(magnitudes[1:]) + 1 if len(magnitudes) > 1 else 0
        print(f"Peak frequency: {frequencies[peak_idx]:.2f} Hz")
        print(f"Peak Strouhal: {strouhal[peak_idx]:.4f}")

        plot_strouhal_spectrum(
            strouhal, magnitudes,
            velocity, diameter,
            title=f'Strouhal Spectrum - {Path(args.csv_file).name}',
            output_path=args.output,
            log_scale=args.log_scale,
            show_reference_lines=not args.no_reference
        )


if __name__ == '__main__':
    main()
