import re
import subprocess
import sys
import os
import tempfile
import glob

import pandas as pd
import streamlit as st


st.set_page_config(page_title="CompressGuard", layout="wide")
st.title("CompressGuard: compression-robust deepfake detection")
tab1, tab2 = st.tabs(["Results", "Try a video"])


def num(value):
    match = re.search(r"^\s*([\d.]+)", str(value))
    if not match:
        raise ValueError(f"Could not parse numeric value from: {value}")
    return float(match.group(1))


with tab1:
    df = pd.read_csv("outputs/v2/eval/summary.csv")
    df["AUC"] = df["auc"].map(num)

    keep = [
        "2a_c23",
        "2b_c23",
        "mixed",
        "mixed_cons",
        "old_compressguard",
        "old_baseline",
    ]

    auc = (
        df[df["name"].isin(keep)]
        .drop_duplicates(["name", "test_variant"], keep="last")
        .pivot(index="name", columns="test_variant", values="AUC")
        .reindex(keep)
    )

    st.subheader("Test AUC, video level (60 test videos)")
    st.bar_chart(auc)
    st.dataframe(
        df[
            [
                "name",
                "test_variant",
                "thr_variant",
                "acc",
                "auc",
                "rec_fake",
                "f1_fake",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Paired bootstrap (AUC difference, 95% CI)")
    st.dataframe(
        pd.read_csv("outputs/v2/eval/paired_bootstrap.csv").round(3),
        use_container_width=True,
    )
    st.caption(
        "FF++ Deepfakes only, 60 test videos, single seed. "
        "Differences between 2b and mixed_cons are not statistically confirmed."
    )


with tab2:
    st.subheader("Analyze a video")
    st.write(
        "Upload a video containing a visible face to estimate whether "
        "it may be manipulated."
    )

    thr = st.selectbox(
        "Evaluation threshold",
        [0.3446, 0.5161, 0.4527],
        format_func=lambda threshold: {
            0.3446: "c23 — 0.3446",
            0.5161: "c40 — 0.5161",
            0.4527: "cascade — 0.4527",
        }[threshold],
    )

    up = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

    if up is not None:
        st.video(up)

    if up is not None and st.button("Analyze video", type="primary"):
        path = None

        try:
            suffix = os.path.splitext(up.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as temp_file:
                temp_file.write(up.getvalue())
                path = temp_file.name

            hf_token = st.secrets.get("HF_TOKEN")
            if not hf_token:
                st.error(
                    "HF_TOKEN is missing. Add it under Streamlit Cloud "
                    "App settings → Secrets."
                )
                st.stop()

            child_env = os.environ.copy()
            child_env["HF_TOKEN"] = hf_token

            with st.spinner("Analyzing video frames..."):
                result = subprocess.run(
                    [
                        sys.executable,
                        "predict_video.py",
                        path,
                        "--threshold",
                        str(thr),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=child_env,
                )

            output = result.stdout + "\n" + result.stderr

            if result.returncode != 0:
                st.error("Video analysis failed.")
                st.code(output[-3000:])
            else:
                mean_match = re.search(
                    r"Mean fake probability:\s*([\d.]+)",
                    result.stdout,
                )
                verdict_match = re.search(
                    r"Verdict:\s*(LIKELY FAKE|LIKELY REAL)",
                    result.stdout,
                )
                faces_match = re.search(
                    r"faces found:\s*(\d+)",
                    result.stdout,
                    re.IGNORECASE,
                )

                if not mean_match or not verdict_match:
                    st.error("Could not read the prediction from the output.")
                    st.code(result.stdout)
                else:
                    probability = float(mean_match.group(1))
                    verdict = verdict_match.group(1)
                    faces = (
                        int(faces_match.group(1))
                        if faces_match
                        else None
                    )

                    st.divider()
                    st.subheader("Analysis result")

                    if verdict == "LIKELY FAKE":
                        st.error("Verdict: Likely Fake")
                    else:
                        st.success("Verdict: Likely Real")

                    st.metric(
                        "Mean fake probability",
                        f"{probability * 100:.1f}%",
                    )
                    st.progress(max(0.0, min(1.0, probability)))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Faces detected",
                            faces if faces is not None else "N/A",
                        )
                    with col2:
                        st.metric("Decision threshold", f"{thr:.4f}")

                    st.warning(
                        "This is a model prediction, not proof that a "
                        "video is real or fake. Performance may vary on "
                        "videos outside the FaceForensics++ dataset."
                    )

                    with st.expander("View frame-level predictions"):
                        frame_match = re.search(
                            r"Per-frame fake probability:\s*(\[[^\]]*\])",
                            result.stdout,
                        )
                        if frame_match:
                            values = [
                                float(value)
                                for value in re.findall(
                                    r"\d+(?:\.\d+)?",
                                    frame_match.group(1),
                                )
                            ]
                            st.dataframe(
                                pd.DataFrame(
                                    {
                                        "Frame": range(1, len(values) + 1),
                                        "Fake probability": values,
                                    }
                                ),
                                use_container_width=True,
                            )

                    crop_dir_match = re.search(
                        r"crops saved in (.+)",
                        result.stdout,
                    )
                    if crop_dir_match:
                        crop_dir = crop_dir_match.group(1).strip()
                        crops = sorted(
                            glob.glob(os.path.join(crop_dir, "*.jpg"))
                        )
                        if crops:
                            with st.expander("View detected face crops"):
                                st.image(
                                    crops,
                                    width=110,
                                    caption=[
                                        os.path.basename(crop)
                                        for crop in crops
                                    ],
                                )

        except subprocess.TimeoutExpired:
            st.error("Analysis timed out. Try a shorter video.")
        except Exception as error:
            st.error(f"An error occurred: {error}")
        finally:
            if path and os.path.exists(path):
                os.remove(path)
