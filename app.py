import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
import librosa
import librosa.display
import parselmouth
from parselmouth.praat import call
import tempfile
import soundfile as sf
import os

st.set_page_config(page_title="Voice Analysis", page_icon="🎤", layout="wide")

# ── Custom styling ──
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; }
    h1, h2, h3, p, span, label { color: white !important; }
    .stFileUploader label { color: white !important; }
    .metric-card {
        background: #16213e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value { font-size: 28px; font-weight: bold; color: white; }
    .metric-label { font-size: 14px; color: #888; margin-bottom: 5px; }
    .metric-note { font-size: 12px; color: #666; font-style: italic; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🎤 Voice Analysis")
st.markdown("Upload a singing or voice recording to analyze pitch, clarity, harmonics, and more.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg", "aac"])

if uploaded_file is not None:
    song_name = os.path.splitext(uploaded_file.name)[0].replace("_", " ").title()
    st.markdown(f"### Analyzing: *{song_name}*")

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Convert to wav if needed (parselmouth needs wav)
    with st.spinner("Loading audio..."):
        y, sr = librosa.load(tmp_path, sr=None)
        wav_path = tmp_path.replace(os.path.splitext(tmp_path)[1], ".wav")
        sf.write(wav_path, y, sr)
        snd = parselmouth.Sound(wav_path)
        duration = snd.get_total_duration()
        st.success(f"Loaded: {duration:.1f}s, {sr} Hz sample rate")

    progress = st.progress(0, text="Computing spectrogram...")

    # ── Spectrogram ──
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    progress.progress(10, text="Computing pitch...")

    # ── Pitch ──
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    pitch_values = pitch.selected_array["frequency"]
    pitch_times = pitch.xs()
    pitch_values[pitch_values == 0] = np.nan
    mean_pitch = np.nanmean(pitch_values)
    pitch_min = np.nanmin(pitch_values)
    pitch_max = np.nanmax(pitch_values)
    pitch_std = np.nanstd(pitch_values)
    progress.progress(20, text="Computing HNR...")

    # ── HNR ──
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_values = np.array([harmonicity.get_value(t) for t in harmonicity.xs()])
    hnr_clean = hnr_values.copy()
    hnr_clean[hnr_clean < 0] = np.nan
    hnr_mean = np.nanmean(hnr_clean)
    progress.progress(30, text="Computing jitter & shimmer...")

    # ── Jitter & Shimmer ──
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    progress.progress(40, text="Computing CPP (this takes a while)...")

    # ── CPP ──
    frame_len = 0.06
    hop = 0.03
    cpp_times = []
    cpp_values = []
    t = 0
    total_frames = int((duration - frame_len) / hop)
    frame_count = 0
    while t + frame_len < duration:
        frame_count += 1
        chunk = snd.extract_part(t, t + frame_len, parselmouth.WindowShape.HAMMING, 1.0, False)
        try:
            spectrum = chunk.to_spectrum()
            cepstrum = call(spectrum, "To PowerCepstrum")
            cpp = call(cepstrum, "Get peak prominence", 60, 333.3, "parabolic", 0.001, 0, "straight", "robust slow")
            cpp_values.append(cpp)
        except:
            cpp_values.append(np.nan)
        cpp_times.append(t + frame_len / 2)
        t += hop
        # Update progress bar within the CPP range (40-70%)
        if frame_count % 50 == 0:
            pct = 40 + int(30 * frame_count / total_frames)
            progress.progress(min(pct, 70), text=f"Computing CPP... {frame_count}/{total_frames}")

    cpp_times = np.array(cpp_times)
    cpp_values = np.array(cpp_values)

    # Filter quiet frames
    intensity = call(snd, "To Intensity", 75, 0.01)
    int_at_cpp = np.array([call(intensity, "Get value at time", t, "cubic") for t in cpp_times])
    quiet_mask = int_at_cpp < (np.nanmax(int_at_cpp) - 25)
    cpp_values[quiet_mask] = np.nan
    cpp_mean = np.nanmean(cpp_values)
    progress.progress(75, text="Computing LTAS...")

    # ── LTAS ──
    ltas = call(snd, "To Ltas", 100)
    n_bins = call(ltas, "Get number of bins")
    ltas_freqs = np.array([call(ltas, "Get frequency from bin number", i) for i in range(1, n_bins + 1)])
    ltas_values = np.array([call(ltas, "Get value in bin", i) for i in range(1, n_bins + 1)])
    mask_8k = ltas_freqs <= 8000

    # LTAS-derived metrics
    low_mask = (ltas_freqs >= 50) & (ltas_freqs < 1000)
    high_mask = (ltas_freqs >= 1000) & (ltas_freqs <= 5000)
    alpha_ratio = np.mean(ltas_values[low_mask]) - np.mean(ltas_values[high_mask])

    sf_mask = (ltas_freqs >= 2500) & (ltas_freqs <= 3500)
    sf_energy = np.max(ltas_values[sf_mask])
    overall_peak = np.max(ltas_values[(ltas_freqs >= 50) & (ltas_freqs <= 2500)])
    spr = overall_peak - sf_energy
    progress.progress(85, text="Computing H1-H2...")

    # ── H1-H2 ──
    spectrum = snd.to_spectrum()
    h1_freq = mean_pitch
    h2_freq = mean_pitch * 2
    h1_h2 = np.nan
    if not np.isnan(h1_freq):
        freqs_spec = np.linspace(0, sr / 2, spectrum.get_number_of_bins())
        spec_values = np.array([
            spectrum.get_real_value_in_bin(i) ** 2 + spectrum.get_imaginary_value_in_bin(i) ** 2
            for i in range(1, spectrum.get_number_of_bins() + 1)
        ])
        spec_db = 10 * np.log10(spec_values + 1e-20)

        def get_harmonic_amp(target_freq, tolerance=20):
            mask = (freqs_spec >= target_freq - tolerance) & (freqs_spec <= target_freq + tolerance)
            if mask.any():
                return np.max(spec_db[mask])
            return np.nan

        h1_amp = get_harmonic_amp(h1_freq)
        h2_amp = get_harmonic_amp(h2_freq)
        h1_h2 = h1_amp - h2_amp

    progress.progress(90, text="Generating plots...")

    # ── Helper functions ──
    def style_ax(ax):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.grid(True, alpha=0.15, color="white", linewidth=0.5)

    def quality_label(val, thresholds, labels):
        for thresh, lab in zip(thresholds, labels):
            if val < thresh:
                return lab
        return labels[-1]

    # ── Generate figure ──
    fig = plt.figure(figsize=(16, 18), facecolor="#1a1a2e")
    gs = GridSpec(4, 1, figure=fig, hspace=0.4)

    # 1. Spectrogram
    ax_spec = fig.add_subplot(gs[0])
    style_ax(ax_spec)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel",
                                    ax=ax_spec, cmap="magma", fmax=8000)
    ax_spec.set_title("Spectrogram", fontsize=14, fontweight="bold", pad=10)
    ax_spec.set_ylabel("Frequency (Hz)", fontsize=11)
    ax_spec.set_xlabel("")
    cax = ax_spec.inset_axes([0.92, 0.6, 0.015, 0.35])
    cb = fig.colorbar(img, cax=cax)
    cb.set_ticks([0, -20, -40, -60, -80])
    cb.set_ticklabels(["Peak", "Loud", "Med", "Quiet", "Silent"])
    cb.ax.tick_params(colors="white", labelsize=7)

    # 2. Pitch
    ax_pitch = fig.add_subplot(gs[1], sharex=ax_spec)
    style_ax(ax_pitch)
    ax_pitch.plot(pitch_times, pitch_values, color="#00d2ff", linewidth=1.2)
    ax_pitch.axhline(y=mean_pitch, color="#ff6b6b", linestyle="--", linewidth=1, alpha=0.7)
    ax_pitch.set_title("Pitch (F0)", fontsize=14, fontweight="bold", pad=10)
    ax_pitch.set_ylabel("Frequency (Hz)", fontsize=11)
    ax_pitch.set_ylim(50, 500)
    ax_pitch.set_xlabel("")
    ax_pitch.text(0.99, 0.95, f"Mean: {mean_pitch:.0f} Hz",
                  transform=ax_pitch.transAxes, ha="right", va="top",
                  fontsize=10, color="#ff6b6b",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="#16213e", edgecolor="#444"))

    # 3. CPP
    ax_cpp = fig.add_subplot(gs[2], sharex=ax_spec)
    style_ax(ax_cpp)
    ax_cpp.plot(cpp_times, cpp_values, color="#2ecc71", linewidth=1.2)
    ax_cpp.axhline(y=cpp_mean, color="#ff6b6b", linestyle="--", linewidth=1, alpha=0.7)
    ax_cpp.set_title("Cepstral Peak Prominence (Voice Clarity Over Time)", fontsize=14, fontweight="bold", pad=10)
    ax_cpp.set_ylabel("CPP (dB)", fontsize=11)
    ax_cpp.set_xlabel("")
    ax_cpp.axhspan(0, 4, alpha=0.08, color="#e74c3c")
    ax_cpp.axhspan(4, 8, alpha=0.08, color="#f39c12")
    ax_cpp.axhspan(8, 30, alpha=0.08, color="#2ecc71")
    ax_cpp.text(0.01, 0.05, "Breathy / rough", transform=ax_cpp.transAxes, fontsize=8, color="#e74c3c", alpha=0.7)
    ax_cpp.text(0.01, 0.35, "Average", transform=ax_cpp.transAxes, fontsize=8, color="#f39c12", alpha=0.7)
    ax_cpp.text(0.01, 0.75, "Clear / harmonic", transform=ax_cpp.transAxes, fontsize=8, color="#2ecc71", alpha=0.7)
    ax_cpp.text(0.99, 0.95, f"Mean: {cpp_mean:.1f} dB",
                transform=ax_cpp.transAxes, ha="right", va="top",
                fontsize=10, color="#ff6b6b",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#16213e", edgecolor="#444"))

    # 4. LTAS
    ax_ltas = fig.add_subplot(gs[3])
    style_ax(ax_ltas)
    ax_ltas.plot(ltas_freqs[mask_8k], ltas_values[mask_8k], color="#a855f7", linewidth=1.2)
    sf_region = (ltas_freqs >= 2500) & (ltas_freqs <= 3500) & mask_8k
    ax_ltas.fill_between(ltas_freqs[sf_region], ltas_values[sf_region],
                          np.min(ltas_values[mask_8k]),
                          alpha=0.25, color="#f39c12", label="Singer's formant region (2.5–3.5 kHz)")
    ax_ltas.set_title("Long-Term Average Spectrum (LTAS)", fontsize=14, fontweight="bold", pad=10)
    ax_ltas.set_ylabel("Energy (dB/Hz)", fontsize=11)
    ax_ltas.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_ltas.set_xlim(0, 8000)
    ax_ltas.legend(loc="upper right", fontsize=9, facecolor="#16213e", edgecolor="#444", labelcolor="white")

    fig.suptitle(f"Voice Analysis — {song_name}", fontsize=20, fontweight="bold", color="white", y=0.995)

    st.pyplot(fig)
    plt.close(fig)

    progress.progress(100, text="Done!")

    # ── Summary Dashboard ──
    st.markdown("---")
    st.markdown("## Voice Quality Summary")

    metrics = [
        ("Mean Pitch", f"{mean_pitch:.0f} Hz", f"Range: {pitch_min:.0f}–{pitch_max:.0f} Hz"),
        ("Mean HNR", f"{hnr_mean:.1f} dB",
         quality_label(hnr_mean, [10, 20], ["Rough", "Moderate", "Clear"])),
        ("Mean CPP", f"{cpp_mean:.1f} dB",
         quality_label(cpp_mean, [4, 8], ["Breathy/rough", "Average", "Clear/harmonic"])),
        ("H1–H2", f"{h1_h2:.1f} dB",
         quality_label(h1_h2, [-2, 2, 6], ["Pressed", "Balanced", "Slightly breathy", "Breathy"])),
    ]

    metrics2 = [
        ("Jitter", f"{jitter * 100:.2f}%",
         quality_label(jitter, [0.01, 0.02], ["Stable", "Elevated", "Unstable"])),
        ("Shimmer", f"{shimmer * 100:.2f}%",
         quality_label(shimmer, [0.03, 0.06], ["Stable", "Elevated", "Unstable"])),
        ("Alpha Ratio", f"{alpha_ratio:.1f} dB",
         "Low freq dominant" if alpha_ratio > 0 else "High freq dominant"),
        ("Singer's Formant", f"SPR: {spr:.1f} dB",
         quality_label(spr, [15, 25], ["Strong presence", "Moderate", "Weak"])),
    ]

    # Row 1
    cols = st.columns(4)
    for col, (label, value, note) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    cols2 = st.columns(4)
    for col, (label, value, note) in zip(cols2, metrics2):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Explanations ──
    st.markdown("---")
    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
        **Mean Pitch** — The fundamental frequency of your voice. Higher = higher singing voice.

        **Mean HNR (Harmonics-to-Noise Ratio)** — How much of your voice is clean tone vs noise. >20 dB = clear voice.

        **Mean CPP (Cepstral Peak Prominence)** — The gold standard for voice clarity. Higher = more harmonic, pleasant voice.

        **H1–H2** — Difference between your first two harmonics. Negative = pressed/tense. Near zero = balanced. Positive = breathy.

        **Jitter** — Pitch stability cycle-to-cycle. Under 1% = stable.

        **Shimmer** — Volume stability cycle-to-cycle. Under 3% = stable.

        **Alpha Ratio** — Energy balance between low (<1kHz) and high (>1kHz) frequencies. Positive = warm/bass-heavy voice.

        **Singer's Formant (SPR)** — Measures projection ability. Lower SPR = stronger presence in the 2.5-3.5 kHz range that lets a voice cut through.
        """)

    # Cleanup temp files
    os.unlink(tmp_path)
    if os.path.exists(wav_path):
        os.unlink(wav_path)
