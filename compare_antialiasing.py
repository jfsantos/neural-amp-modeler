"""
Compare aliasing between two NAM WaveNet models (with and without anti-aliasing).

Generates several diagnostic plots:
  1. Single-tone spectral analysis at multiple frequencies
  2. Swept-sine spectrogram comparison
  3. Alias energy vs. frequency curve

Usage:
    python compare_antialiasing.py model_no_aa.ckpt model_aa.ckpt --model-config config.json
    python compare_antialiasing.py model_no_aa.nam model_aa.nam
    python compare_antialiasing.py model_no_aa.nam model_aa.nam --sample-rate 48000
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nam.models._from_nam import init_from_nam
from nam.train.lightning_module import LightningModule


def load_model(model_path: str, model_config_path: str = None) -> torch.nn.Module:
    model_path = Path(model_path)
    if model_path.suffix == ".nam":
        with open(model_path) as f:
            return init_from_nam(json.load(f))
    elif model_path.suffix == ".ckpt":
        if model_config_path is None:
            ckpt = LightningModule.load_from_checkpoint(model_path)
        else:
            with open(model_config_path) as f:
                config = json.load(f)
            ckpt = LightningModule.load_from_checkpoint(
                model_path, **LightningModule.parse_config(config)
            )
        return ckpt.net
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def run_model(model, x_np):
    """Run inference on a numpy array, return numpy array."""
    x = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    return y.numpy()


def generate_sine(freq, sr, duration, amplitude=0.5):
    t = np.arange(int(sr * duration)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_swept_sine(f_start, f_end, sr, duration, amplitude=0.5):
    t = np.arange(int(sr * duration)) / sr
    phase = 2 * np.pi * f_start * t + (
        2 * np.pi * (f_end - f_start) / (2 * duration) * t**2
    )
    return amplitude * np.sin(phase)


def compute_spectrum_db(signal, sr, nfft=None):
    if nfft is None:
        nfft = len(signal)
    window = np.hanning(len(signal))
    spectrum = np.fft.rfft(signal * window, n=nfft)
    mag = np.abs(spectrum) / (len(signal) / 2)
    mag_db = 20 * np.log10(mag + 1e-12)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    return freqs, mag_db


def measure_alias_energy(signal, fundamental_freq, sr, num_harmonics=20):
    """Measure energy in non-harmonic (aliased) frequency bins.

    Returns (alias_energy_db, harmonic_energy_db, total_energy_db).
    """
    nfft = len(signal)
    window = np.hanning(nfft)
    spectrum = np.abs(np.fft.rfft(signal * window, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    bin_width = sr / nfft

    # Mark harmonic bins (fundamental + harmonics, +/- 3 bins for spectral leakage)
    harmonic_mask = np.zeros(len(freqs), dtype=bool)
    for h in range(1, num_harmonics + 1):
        hf = fundamental_freq * h
        if hf > sr / 2:
            break
        bin_idx = int(round(hf / bin_width))
        lo = max(0, bin_idx - 3)
        hi = min(len(freqs), bin_idx + 4)
        harmonic_mask[lo:hi] = True

    # Also mask DC and near-DC
    harmonic_mask[:5] = True

    harmonic_power = np.sum(spectrum[harmonic_mask] ** 2)
    alias_power = np.sum(spectrum[~harmonic_mask] ** 2)
    total_power = np.sum(spectrum**2)

    to_db = lambda p: 10 * np.log10(p + 1e-30)
    return to_db(alias_power), to_db(harmonic_power), to_db(total_power)


def plot_single_tone_spectra(
    model_baseline, model_aa, sr, test_freqs, duration=0.5, output_path=None
):
    """Plot output spectra for single-tone inputs at several frequencies."""
    n_freqs = len(test_freqs)
    fig, axes = plt.subplots(n_freqs, 1, figsize=(12, 3.5 * n_freqs), sharex=True)
    if n_freqs == 1:
        axes = [axes]

    for ax, freq in zip(axes, test_freqs):
        x = generate_sine(freq, sr, duration)
        y_base = run_model(model_baseline, x)
        y_aa = run_model(model_aa, x)

        freqs_b, spec_b = compute_spectrum_db(y_base, sr)
        freqs_a, spec_a = compute_spectrum_db(y_aa, sr)

        ax.plot(freqs_b, spec_b, alpha=0.7, label="Baseline (no AA)", linewidth=0.6)
        ax.plot(freqs_a, spec_a, alpha=0.7, label="Anti-aliased", linewidth=0.6)

        # Mark expected harmonics
        for h in range(1, 21):
            hf = freq * h
            if hf > sr / 2:
                break
            ax.axvline(hf, color="gray", alpha=0.15, linewidth=0.5)

        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Input: {freq} Hz sine")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(bottom=-120)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Single-tone spectral comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_swept_sine_spectrograms(
    model_baseline, model_aa, sr, duration=2.0, output_path=None
):
    """Plot spectrograms of swept-sine output for both models."""
    f_start = 100
    f_end = sr * 0.45  # Just below Nyquist
    x = generate_swept_sine(f_start, f_end, sr, duration)
    y_base = run_model(model_baseline, x)
    y_aa = run_model(model_aa, x)

    nperseg = 1024
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, sharey=True)

    for ax, y, title in [
        (axes[0], x[: len(y_base)], "Input (swept sine)"),
        (axes[1], y_base, "Baseline (no AA)"),
        (axes[2], y_aa, "Anti-aliased"),
    ]:
        ax.specgram(
            y, NFFT=nperseg, Fs=sr, noverlap=nperseg // 2, cmap="magma", vmin=-120
        )
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Swept-sine spectrogram comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def plot_alias_energy_curve(
    model_baseline, model_aa, sr, duration=0.5, output_path=None
):
    """Plot alias energy as a function of input frequency."""
    # Test frequencies from ~100 Hz up to near Nyquist
    test_freqs = np.geomspace(100, sr * 0.4, num=40)

    alias_db_base = []
    alias_db_aa = []
    anr_base = []  # alias-to-harmonic noise ratio
    anr_aa = []

    for freq in test_freqs:
        x = generate_sine(freq, sr, duration)
        y_base = run_model(model_baseline, x)
        y_aa = run_model(model_aa, x)

        a_b, h_b, _ = measure_alias_energy(y_base, freq, sr)
        a_a, h_a, _ = measure_alias_energy(y_aa, freq, sr)

        alias_db_base.append(a_b)
        alias_db_aa.append(a_a)
        anr_base.append(a_b - h_b)  # alias relative to harmonics
        anr_aa.append(a_a - h_a)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(test_freqs, alias_db_base, "o-", label="Baseline (no AA)", markersize=3)
    ax1.plot(test_freqs, alias_db_aa, "o-", label="Anti-aliased", markersize=3)
    ax1.set_ylabel("Alias energy (dB)")
    ax1.set_title("Absolute alias energy vs. input frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(test_freqs, anr_base, "o-", label="Baseline (no AA)", markersize=3)
    ax2.plot(test_freqs, anr_aa, "o-", label="Anti-aliased", markersize=3)
    ax2.set_xlabel("Input frequency (Hz)")
    ax2.set_ylabel("Alias / Harmonic ratio (dB)")
    ax2.set_title("Alias-to-harmonic ratio vs. input frequency (lower is better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    fig.suptitle("Alias energy analysis", fontsize=14, y=1.01)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare aliasing between baseline and anti-aliased NAM models"
    )
    parser.add_argument("model_baseline", help="Path to baseline model (.nam or .ckpt)")
    parser.add_argument("model_aa", help="Path to anti-aliased model (.nam or .ckpt)")
    parser.add_argument(
        "--model-config", help="Model config JSON (for .ckpt files)", default=None
    )
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save plots (default: current dir)",
    )
    parser.add_argument(
        "--test-freqs",
        nargs="+",
        type=float,
        default=[200, 1000, 3000, 8000],
        help="Frequencies for single-tone test (default: 200 1000 3000 8000)",
    )
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")
    args = parser.parse_args()

    sr = args.sample_rate
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline model: {args.model_baseline}")
    model_base = load_model(args.model_baseline, args.model_config)
    model_base.cpu().eval()

    print(f"Loading AA model: {args.model_aa}")
    model_aa = load_model(args.model_aa, args.model_config)
    model_aa.cpu().eval()

    print(f"Sample rate: {sr} Hz")
    print(f"Test frequencies: {args.test_freqs}")
    print()

    print("1/3  Single-tone spectral analysis...")
    plot_single_tone_spectra(
        model_base,
        model_aa,
        sr,
        args.test_freqs,
        output_path=outdir / "aa_single_tone_spectra.png",
    )

    print("2/3  Swept-sine spectrograms...")
    plot_swept_sine_spectrograms(
        model_base,
        model_aa,
        sr,
        output_path=outdir / "aa_swept_sine_spectrograms.png",
    )

    print("3/3  Alias energy curve...")
    plot_alias_energy_curve(
        model_base,
        model_aa,
        sr,
        output_path=outdir / "aa_alias_energy_curve.png",
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
