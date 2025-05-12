# main.py
import os
import logging
import sys
from app.features_univariate import UnivariateFeatures
import pandas as pd
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger()

if __name__ == "__main__":
    print("Starting main.py")
    input_dir = os.getenv("INPUT_DIR", "/data/input")
    output_dir = os.getenv("OUTPUT_DIR", "/data/output")
    os.makedirs(output_dir, exist_ok=True)

    subject_id = "sub-RID0031"  # You may want to load this dynamically later

    logger.info(f"Running univariate feature extraction for {subject_id}")
    features = UnivariateFeatures(subject_id)

    logger.info("Saving Catch22 features...")
    features.catch22_features().to_csv(os.path.join(output_dir, f"{subject_id}_catch22.csv"))

    logger.info("Saving FOOOF features...")
    features.fooof_features().to_csv(os.path.join(output_dir, f"{subject_id}_fooof.csv"))

    logger.info("Saving Bandpower features...")
    features.bandpower_features().to_csv(os.path.join(output_dir, f"{subject_id}_bandpower.csv"))

    logger.info("Saving Entropy features...")
    entropy = features.entropy_features()
    with open(os.path.join(output_dir, f"{subject_id}_entropy.json"), 'w') as f:
        json.dump(entropy, f)

    logger.info("Finished feature extraction")