#!/usr/bin/env python3
"""
FFT Analysis Tool for Audio Files

A resilient FFT analyzer that handles a wide variety of audio formats,
including iPhone Voice Memos (m4a/AAC), mp3, wav, flac, ogg, and more.

Usage:
    python fft_analysis.py <audio_file> [options]

Examples:
    python fft_analysis.py recording.m4a
    python fft_analysis.py recording.wav --window hann --no-plot
    python fft_analysis.py memo.m4a --strouhal --velocity 10.5 --diameter 0.05
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import get_window


# =============================================================================
# Audio Loading
# =============================================================================

def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load audio from various formats using pydub (which uses ffmpeg under the hood).

    Supports: wav, mp3, m4a, aac, flac, ogg, wma, aiff, and more.

    Args:
        filepath: Path to the audio file

    Returns:
        Tuple of (samples as numpy array normalized to [-1, 1], sample rate)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported or corrupted
    """
    from pydub import AudioSegment

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Get file extension (without the dot, lowercase)
    extension = path.suffix.lower().lstrip('.')

    # Map common extensions to pydub format names
    # pydub/ffmpeg handles most formats automatically
    format_map = {
        'm4a': 'm4a',
        'aac': 'aac',
        'mp3': 'mp3',
        'wav': 'wav',
        'flac': 'flac',
        'ogg': 'ogg',
        'wma': 'wma',
        'aiff': 'aiff',
        'aif': 'aiff',
        'caf': 'caf',  # Core Audio Format (Apple)
    }

    fmt = format_map.get(extension, extension)

    try:
        # Load the audio file
        audio = AudioSegment.from_file(filepath, format=fmt)

        # Convert stereo to mono by averaging channels
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Extract raw samples
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)

        # Normalize to [-1, 1] based on bit depth
        # pydub uses sample_width in bytes (1=8bit, 2=16bit, 4=32bit)
        max_val = 2 ** (audio.sample_width * 8 - 1)
        samples = samples / max_val

        sample_rate = audio.frame_rate

        return samples, sample_rate

    except Exception as e:
        raise ValueError(f"Failed to load audio file '{filepath}': {e}")


# =============================================================================
# FFT Computation
# =============================================================================

def compute_fft(
    samples: np.ndarray,
    sample_rate: int,
    window_type: str = 'hann'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of the audio signal.

    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        window_type: Window function to apply ('hann', 'hamming', 'blackman',
                     'bartlett', 'rectangular'). Default is 'hann'.

    Returns:
        Tuple of (frequencies in Hz, magnitude spectrum in dB)
    """
    n_samples = len(samples)

    # Apply window function to reduce spectral leakage
    if window_type == 'rectangular' or window_type == 'none':
        window = np.ones(n_samples)
    else:
        window = get_window(window_type, n_samples)

    windowed_samples = samples * window

    # Compute FFT
    fft_result = fft(windowed_samples)

    # Get frequency bins (only positive frequencies)
    frequencies = fftfreq(n_samples, d=1/sample_rate)
    positive_mask = frequencies >= 0

    frequencies = frequencies[positive_mask]
    fft_result = fft_result[positive_mask]

    # Compute magnitude spectrum
    # Multiply by 2 to account for discarded negative frequencies (except DC)
    magnitude = np.abs(fft_result) * 2 / n_samples
    magnitude[0] /= 2  # DC component shouldn't be doubled

    # Convert to decibels (with floor to avoid log(0))
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

    return frequencies, magnitude_db


# =============================================================================
# Strouhal Number Conversion
# =============================================================================

def frequency_to_strouhal(
    frequencies: np.ndarray,
    velocity: float,
    diameter: float
) -> np.ndarray:
    """
    Convert frequencies to Strouhal numbers.

    The Strouhal number is a dimensionless quantity used in fluid mechanics:
        St = f * d / v

    where:
        f = frequency (Hz)
        d = characteristic length/diameter (m)
        v = flow velocity (m/s)

    Args:
        frequencies: Array of frequencies in Hz
        velocity: Flow velocity in m/s
        diameter: Characteristic length/diameter in m

    Returns:
        Array of Strouhal numbers
    """
    if velocity <= 0:
        raise ValueError("Velocity must be positive")
    if diameter <= 0:
        raise ValueError("Diameter must be positive")

    return (frequencies * diameter) / velocity


# =============================================================================
# Plotting
# =============================================================================

def plot_spectrum(
    x_data: np.ndarray,
    magnitude_db: np.ndarray,
    x_label: str = 'Frequency (Hz)',
    title: str = 'FFT Spectrum',
    output_path: str | None = None,
    x_log_scale: bool = False
) -> None:
    """
    Plot the frequency/Strouhal spectrum.

    Args:
        x_data: X-axis data (frequencies or Strouhal numbers)
        magnitude_db: Magnitude spectrum in dB
        x_label: Label for x-axis
        title: Plot title
        output_path: If provided, save figure to this path instead of showing
        x_log_scale: If True, use logarithmic scale for x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Skip DC component (index 0) for cleaner plotting
    ax.plot(x_data[1:], magnitude_db[1:], linewidth=0.5)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if x_log_scale and x_data[1] > 0:  # Avoid log(0)
        ax.set_xscale('log')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FFT analysis tool for audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s recording.m4a
  %(prog)s voice_memo.m4a --window hamming
  %(prog)s data.wav --strouhal --velocity 15.0 --diameter 0.02
  %(prog)s audio.mp3 --output spectrum.png --log-scale
        """
    )

    parser.add_argument(
        'audio_file',
        help='Path to the audio file (supports wav, mp3, m4a, flac, ogg, etc.)'
    )

    parser.add_argument(
        '--window', '-w',
        choices=['hann', 'hamming', 'blackman', 'bartlett', 'rectangular'],
        default='hann',
        help='Window function for FFT (default: hann)'
    )

    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting, only print summary statistics'
    )

    parser.add_argument(
        '--output', '-o',
        help='Save plot to file instead of displaying'
    )

    parser.add_argument(
        '--log-scale',
        action='store_true',
        help='Use logarithmic scale for frequency/Strouhal axis'
    )

    # Strouhal number options
    parser.add_argument(
        '--strouhal', '-s',
        action='store_true',
        help='Plot against Strouhal number instead of frequency'
    )

    parser.add_argument(
        '--velocity', '-v',
        type=float,
        help='Flow velocity in m/s (required for Strouhal plot)'
    )

    parser.add_argument(
        '--diameter', '-d',
        type=float,
        help='Characteristic length/diameter in m (required for Strouhal plot)'
    )

    args = parser.parse_args()

    # Validate Strouhal arguments
    if args.strouhal:
        if args.velocity is None or args.diameter is None:
            # Prompt for values interactively
            print("Strouhal number calculation requires velocity and diameter.")
            try:
                if args.velocity is None:
                    args.velocity = float(input("Enter flow velocity (m/s): "))
                if args.diameter is None:
                    args.diameter = float(input("Enter characteristic diameter (m): "))
            except (ValueError, EOFError):
                print("Error: Invalid input for velocity or diameter", file=sys.stderr)
                sys.exit(1)

    # Load audio
    print(f"Loading: {args.audio_file}")
    try:
        samples, sample_rate = load_audio(args.audio_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    duration = len(samples) / sample_rate
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(samples):,}")

    # Compute FFT
    print(f"Computing FFT with {args.window} window...")
    frequencies, magnitude_db = compute_fft(samples, sample_rate, args.window)

    # Find dominant frequency (excluding DC)
    peak_idx = np.argmax(magnitude_db[1:]) + 1
    peak_freq = frequencies[peak_idx]
    peak_mag = magnitude_db[peak_idx]
    print(f"  Dominant frequency: {peak_freq:.2f} Hz ({peak_mag:.1f} dB)")

    # Convert to Strouhal if requested
    if args.strouhal:
        x_data = frequency_to_strouhal(frequencies, args.velocity, args.diameter)
        x_label = 'Strouhal Number (St)'
        peak_st = frequency_to_strouhal(np.array([peak_freq]), args.velocity, args.diameter)[0]
        print(f"  Dominant Strouhal: {peak_st:.4f}")
        title = f'FFT Spectrum - {Path(args.audio_file).name}\n(v={args.velocity} m/s, d={args.diameter} m)'
    else:
        x_data = frequencies
        x_label = 'Frequency (Hz)'
        title = f'FFT Spectrum - {Path(args.audio_file).name}'

    # Plot
    if not args.no_plot:
        plot_spectrum(
            x_data,
            magnitude_db,
            x_label=x_label,
            title=title,
            output_path=args.output,
            x_log_scale=args.log_scale
        )


if __name__ == '__main__':
    main()
