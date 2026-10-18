import re, subprocess, sys, os, tempfile, glob
import pandas as pd, streamlit as st

st.set_page_config(page_title="CompressGuard", layout="wide")
st.title("CompressGuard: compression-robust deepfake detection")
tab1, tab2 = st.tabs(["Results", "Try a video"])

def num(s):
    return float(re.match(r"\s*([\d.]+)", str(s)).group(1))

with tab1:
    df = pd.read_csv("outputs/v2/eval/summary.csv")
    df["AUC"] = df["auc"].map(num)
    keep = ["2a_c23", "2b_c23", "mixed", "mixed_cons", "old_compressguard", "old_baseline"]
    # AUC does not depend on the threshold, so keep the last row per model/variant
    auc = (df[df["name"].isin(keep)].drop_duplicates(["name", "test_variant"], keep="last")
           .pivot(index="name", columns="test_variant", values="AUC").reindex(keep))
    st.subheader("Test AUC, video level (60 test videos)")
    st.bar_chart(auc)
    st.dataframe(df[["name", "test_variant", "thr_variant", "acc", "auc", "rec_fake", "f1_fake", "TN", "FP", "FN", "TP"]],
                 use_container_width=True)
    st.subheader("Paired bootstrap (AUC difference, 95% CI)")
    st.dataframe(pd.read_csv("outputs/v2/eval/paired_bootstrap.csv").round(3), use_container_width=True)
    st.caption("FF++ Deepfakes only, 60 test videos, single seed. Differences between 2b and mixed_cons are not statistically confirmed.")

with tab2:
    st.write("Upload a short video with a visible face. The model was trained only on FF++ Deepfakes, so treat the result as a probability, not proof.")
    thr = st.selectbox("Threshold (mixed_cons val thresholds)", [0.3446, 0.5161, 0.4527],
                       format_func=lambda t: {0.3446: "c23: 0.3446", 0.5161: "c40: 0.5161", 0.4527: "cascade: 0.4527"}[t])
    up = st.file_uploader("Video", type=["mp4", "mov", "avi"])
    if up and st.button("Analyze"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(up.read()); path = f.name
        with st.spinner("Detecting faces and running the model..."):
            r = subprocess.run([sys.executable, "predict_video.py", path, "--threshold", str(thr)],
                               capture_output=True, text=True)
        st.code(r.stdout + r.stderr[-500:] if r.returncode else r.stdout)
        crops = sorted(glob.glob(os.path.join("outputs", "v2", "predict_tmp",
                       os.path.splitext(os.path.basename(path))[0], "*.jpg")))
        if crops:
            st.image(crops, width=110, caption=[os.path.basename(c) for c in crops])
        os.remove(path)