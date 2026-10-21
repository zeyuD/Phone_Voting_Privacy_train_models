import os
import sys
import glob
import pandas as pd

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config

config = load_machine_config()

work_directory = config["data_dir"] + "Phone_Privacy/"
metrics_directory = os.path.join(work_directory, "data", "sr_metrics_hpc")

metrics = ["psnr_global_db", "psnr_mean_db", "ssim_mean"]
# Videos whose SSIM is more than this many std below their [resolution, setup, user] group mean are flagged
outlier_std = 2.0
# Set to True to save the merged scores and summary as csv files in metrics_directory
save_csv = True

# Read "sr_metrics_hpc/shard_*/video_scores.csv" and merge into one table
score_files = sorted(glob.glob(os.path.join(metrics_directory, "*", "video_scores.csv")))
if not score_files:
    raise FileNotFoundError(f"No video_scores.csv found under {metrics_directory}")
print(f"Found {len(score_files)} video_scores.csv files in {metrics_directory}")

frames = []
for score_file in score_files:
    # The files start with a UTF-8 BOM, "utf-8-sig" strips it from the first column name
    df = pd.read_csv(score_file, encoding="utf-8-sig", dtype={"sample_index": str})
    df["shard"] = os.path.basename(os.path.dirname(score_file))
    frames.append(df)
scores = pd.concat(frames, ignore_index=True)

# "180p/pad_op2/JingweiPad/A/003_ESRGAN_1080p.mp4" -> setup "pad_op2", user "JingweiPad" (keep original case)
path_parts = scores["sr_relative_path"].str.split("/")
scores["setup"] = path_parts.str[1]
scores["user"] = path_parts.str[2]

duplicated = scores.duplicated("sr_relative_path", keep=False)
if duplicated.any():
    print(f"\nWarning: {duplicated.sum()} rows share an sr_relative_path across shards:")
    print(scores.loc[duplicated, ["shard", "sr_relative_path"]].to_string(index=False))

print(f"\nTotal videos: {len(scores)}")
print(scores["status"].value_counts().to_string())

# Videos that could not be evaluated
failed = scores[scores["status"] != "ok"]
if not failed.empty:
    print(f"\n{len(failed)} videos failed evaluation:")
    print(failed[["shard", "sr_relative_path", "gt_relative_path", "status", "error"]].to_string(index=False))

ok = scores[scores["status"] == "ok"].copy()

# Sanity checks on the evaluated videos
size_mismatch = ok[(ok["gt_width"] != ok["sr_width"]) | (ok["gt_height"] != ok["sr_height"])]
if not size_mismatch.empty:
    print(f"\nWarning: {len(size_mismatch)} videos have different GT and SR frame sizes:")
    print(size_mismatch[["sr_relative_path", "gt_width", "gt_height", "sr_width", "sr_height"]].to_string(index=False))

frame_mismatch = ok[ok["gt_frames_decoded"] != ok["sr_frames_decoded"]]
if not frame_mismatch.empty:
    print(f"\nWarning: {len(frame_mismatch)} videos have different GT and SR decoded frame counts "
          "(only frames_compared frames were scored):")
    print(frame_mismatch[["sr_relative_path", "gt_frames_decoded", "sr_frames_decoded", "frames_compared"]].to_string(index=False))

# Quality summary
print("\n===== Quality by input resolution =====")
print(ok.groupby("input_resolution")[metrics].agg(["mean", "std"]).round(4).to_string())

print("\n===== Quality by input resolution and setup =====")
summary = ok.groupby(["input_resolution", "setup"])[metrics].agg(["count", "mean", "std", "min", "max"]).round(4)
print(summary.to_string())

print("\n===== Mean quality by input resolution, setup and user =====")
print(ok.groupby(["input_resolution", "setup", "user"])[metrics].mean().round(4).to_string())

print("\n===== Mean SSIM by input resolution, setup and vote choice =====")
print(ok.pivot_table(index=["input_resolution", "setup"], columns="choice", values="ssim_mean", aggfunc="mean").round(4).to_string())

# Flag low-quality videos relative to the same user's other videos, so a user with overall lower quality is not flagged entirely
group_ssim = ok.groupby(["input_resolution", "setup", "user"])["ssim_mean"]
ok["ssim_z"] = (ok["ssim_mean"] - group_ssim.transform("mean")) / group_ssim.transform("std")
outliers = ok[ok["ssim_z"] < -outlier_std].sort_values("ssim_z")
print(f"\n===== {len(outliers)} videos with SSIM more than {outlier_std} std below their group mean =====")
if not outliers.empty:
    print(outliers[["sr_relative_path", "frames_compared", "psnr_mean_db", "ssim_mean", "ssim_z"]].round(4).to_string(index=False))

if save_csv:
    merged_path = os.path.join(metrics_directory, "video_scores_all.csv")
    summary_path = os.path.join(metrics_directory, "video_scores_summary.csv")
    scores.to_csv(merged_path, index=False)
    summary.to_csv(summary_path)
    print(f"\nSaved merged scores to {merged_path}")
    print(f"Saved summary to {summary_path}")
