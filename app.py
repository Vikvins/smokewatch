"""SmokeWatch web app: analyze a video and look at the detected events.

Run locally:
    python -m streamlit run app.py

Everything happens on this machine: uploaded videos are saved to
data/uploads/ and results to outputs/, both git-ignored.
"""

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from smokewatch.annotations import load_clips
from smokewatch.classifier import SmokingClassifier
from smokewatch.events import OUTPUT_COLUMNS
from smokewatch.overlay import render_overlay
from smokewatch.pipeline import analyze_video, load_roi_mask

CONFIG_PATH = Path("configs/default.yaml")
CLIPS_PATH = Path("data/annotations/clips.csv")
GT_PATH = Path("data/annotations/events.csv")
RAW_DIR = Path("data/raw")
UPLOAD_DIR = Path("data/uploads")
OUTPUTS = Path("outputs")
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "wmv"]
DEVICES = ["mps", "cpu", "cuda"]


@st.cache_resource(show_spinner="Loading the CLIP model...")
def get_classifier(classifier_config: str) -> SmokingClassifier:
    # the config is passed as YAML text so Streamlit can hash it
    return SmokingClassifier(yaml.safe_load(classifier_config))


def save_upload(upload) -> tuple[Path, str]:
    """Save an uploaded file; its clip_id comes from the content, so caches stay valid."""
    data = upload.getvalue()
    clip_id = "upload_" + hashlib.sha1(data).hexdigest()[:10]
    path = UPLOAD_DIR / f"{clip_id}{Path(upload.name).suffix.lower()}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path, clip_id


def choose_video() -> tuple[Path, pd.Series] | None:
    """Sidebar: a clip from clips.csv or an uploaded file."""
    sources = ["Upload a video"]
    if CLIPS_PATH.exists():
        sources.insert(0, "Clip from clips.csv")
    source = st.sidebar.radio("Video", sources)

    if source == "Clip from clips.csv":
        clips = load_clips(CLIPS_PATH)
        clip_id = st.sidebar.selectbox("Clip", clips["clip_id"])
        clip = clips.set_index("clip_id").loc[clip_id]
        return RAW_DIR / clip["path"], pd.Series({"clip_id": clip_id, **clip})

    upload = st.sidebar.file_uploader("Video file", type=VIDEO_TYPES)
    if upload is None:
        return None
    path, clip_id = save_upload(upload)
    return path, pd.Series({"clip_id": clip_id, "camera_id": "upload"})


def choose_settings(config: dict) -> dict:
    """Sidebar: the few settings worth changing by hand."""
    st.sidebar.subheader("Settings")
    config["classifier"]["enabled"] = st.sidebar.checkbox(
        "CLIP check (drops drinking, phone, ...)", value=config["classifier"]["enabled"]
    )
    config["classifier"]["min_score"] = st.sidebar.slider(
        "CLIP min score", 0.0, 0.5, float(config["classifier"]["min_score"]), 0.01
    )
    config["events"]["threshold"] = st.sidebar.slider(
        "Event confidence threshold", 0.0, 1.0, float(config["events"]["threshold"]), 0.05
    )
    config["pose"]["device"] = config["classifier"]["device"] = st.sidebar.selectbox(
        "Device", DEVICES, index=DEVICES.index(config["pose"]["device"])
    )
    return config


def analyze(video: Path, clip: pd.Series, config: dict) -> dict:
    classifier = None
    if config["classifier"]["enabled"]:
        classifier = get_classifier(yaml.safe_dump(config["classifier"]))

    with st.spinner("Finding people, poses and gestures (cached after the first run)..."):
        roi_mask = load_roi_mask(Path("configs/roi"), str(clip["camera_id"]))
        rows, cycles, poses = analyze_video(video, clip, config, OUTPUTS, roi_mask, classifier)
    events = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    gt = None
    if GT_PATH.exists():
        gt = pd.read_csv(GT_PATH, dtype={"clip_id": str})
        gt = gt[(gt["clip_id"] == clip["clip_id"]) & (gt["label"] == "POSITIVE")]

    bar = st.progress(0.0, text="Drawing the overlay video...")
    overlay = render_overlay(
        video,
        poses,
        cycles,
        events,
        OUTPUTS / "overlays" / "app" / f"{clip['clip_id']}.mp4",
        gt=gt,
        max_width=1280,
        progress=lambda share: bar.progress(share, text="Drawing the overlay video..."),
    )
    bar.empty()
    return {
        "clip_id": clip["clip_id"],
        "events": events,
        "cycles": cycles,
        "gt": gt,
        "overlay": overlay,
    }


def show(result: dict) -> None:
    events, cycles, gt = result["events"], result["cycles"], result["gt"]
    left, middle, right = st.columns(3)
    left.metric("Smoking-like events", len(events))
    middle.metric("Hand-to-mouth cycles", len(cycles))
    right.metric("Ground-truth events", "-" if gt is None else len(gt))

    start = 0
    if len(events):
        labels = [
            f"{row.event_id}: {row.start_sec:.1f}-{row.end_sec:.1f} s, person {row.person_track_id}"
            for row in events.itertuples()
        ]
        chosen = st.selectbox("Jump to event", ["start of the video"] + labels)
        if chosen != "start of the video":
            start = int(events.iloc[labels.index(chosen)]["start_sec"])
    st.video(str(result["overlay"]), start_time=start)
    st.caption(
        "Grey: tracked person. Yellow: hand at the mouth. Red: person in a detected event. "
        "Green bar at the bottom: ground truth (if labeled)."
    )

    st.subheader("Events")
    if len(events):
        st.dataframe(events, hide_index=True)
        st.download_button(
            "Download events.csv",
            events.to_csv(index=False),
            file_name=f"{result['clip_id']}_events.csv",
        )
    else:
        st.write("No smoking-like events found.")

    with st.expander("All hand-to-mouth cycles"):
        st.dataframe(cycles, hide_index=True)
    if gt is not None and len(gt):
        with st.expander("Ground truth"):
            st.dataframe(
                gt[["event_id", "start_sec", "end_sec", "cue", "comment"]], hide_index=True
            )


def main() -> None:
    st.set_page_config(page_title="SmokeWatch", layout="wide")
    st.title("SmokeWatch")
    st.write(
        "Event-level detection of smoking-like behavior: pose tracking, hand-to-mouth "
        "gesture cycles and a zero-shot CLIP check. Runs fully on this machine."
    )

    chosen = choose_video()
    config = choose_settings(yaml.safe_load(CONFIG_PATH.read_text()))
    if chosen is None:
        st.info("Choose a clip or upload a video in the sidebar.")
        return
    video, clip = chosen

    if st.sidebar.button("Analyze", type="primary"):
        st.session_state["result"] = analyze(video, clip, config)
    result = st.session_state.get("result")
    if result is not None and result["clip_id"] == clip["clip_id"]:
        show(result)
    else:
        st.info("Press **Analyze** in the sidebar.")


main()
