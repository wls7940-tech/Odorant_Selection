import os
import numpy as np
import pandas as pd

# =========================
# LOAD GT
# =========================

# The GT vector file should be placed in the same directory
# as this Python script.
GT_FILE_PATH = "GT_vectors.xlsx"

if not os.path.isfile(GT_FILE_PATH):
    raise FileNotFoundError(
        f"Ground-truth vector file not found: {GT_FILE_PATH}\n"
        "Please place 'GT_vectors.xlsx' in the same directory as this script."
    )

df = pd.read_excel(GT_FILE_PATH)
df.columns = df.columns.str.strip()


# =========================
# UTIL
# =========================
def split_terms(text):
    return [
        x.strip().lower()
        for x in str(text).split(";")
        if x.strip()
    ]


def split_numbers(text):
    return [
        float(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]


def normalize(v):
    v = np.array(v, dtype=float)

    s = np.sum(v)

    if s == 0:
        return v

    return v / s


# =========================
# COSINE SIMILARITY
# =========================
def cosine(a, b):

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    cos_sim = np.dot(a, b) / denom

    # Prevent floating-point errors
    cos_sim = np.clip(
        cos_sim,
        -1.0,
        1.0
    )

    return cos_sim


# =========================
# ANGULAR DISTANCE
# =========================
def angular_distance(a, b):

    cos_sim = cosine(a, b)

    # np.arccos returns the angle in radians
    angle = np.arccos(cos_sim)

    return angle


# =========================
# BUILD VOCAB (GT + PRED)
# =========================
def build_vocab(all_pred_words):

    vocab = set()

    # GT descriptors
    for _, row in df.iterrows():

        vocab.update(
            split_terms(
                row["descriptor"]
            )
        )

    # Predicted descriptors
    for pred in all_pred_words:

        for item in pred:

            label = (
                item
                .split("(")[0]
                .strip()
                .lower()
            )

            vocab.add(label)

    vocab = sorted(
        list(vocab)
    )

    vocab_index = {
        w: i
        for i, w in enumerate(vocab)
    }

    return vocab, vocab_index


# =========================
# GT VECTOR
# =========================
def build_gt_vec(
    descriptor_text,
    weight_text,
    vocab_index,
    vocab
):

    vec = np.zeros(
        len(vocab)
    )

    descriptors = split_terms(
        descriptor_text
    )

    weights = normalize(
        split_numbers(
            weight_text
        )
    )

    for descriptor, weight in zip(
        descriptors,
        weights
    ):

        if descriptor in vocab_index:

            vec[
                vocab_index[descriptor]
            ] = weight

    return vec


# =========================
# PREDICTION VECTOR (TOP 5)
# =========================
def build_pred_vec(
    pred_words,
    vocab_index,
    vocab
):

    vec = np.zeros(
        len(vocab)
    )

    labels = []
    probs = []

    for item in pred_words:

        label = (
            item
            .split("(")[0]
            .strip()
            .lower()
        )

        prob = float(
            item
            .split("(")[1]
            .replace(")", "")
            .strip()
        )

        labels.append(label)
        probs.append(prob)

    probs = np.array(
        probs,
        dtype=float
    )

    # Normalize top-5 prediction probabilities
    if probs.sum() != 0:
        probs = probs / probs.sum()

    for label, prob in zip(
        labels,
        probs
    ):

        if label in vocab_index:

            vec[
                vocab_index[label]
            ] = prob

    return vec


# =========================
# ANGULAR DISTANCE MATCHING
# =========================
def run_angular_distance(
    all_pred_words
):

    vocab, vocab_index = build_vocab(
        all_pred_words
    )

    results = []

    for i in range(
        len(all_pred_words)
    ):

        pred_vec = build_pred_vec(
            all_pred_words[i],
            vocab_index,
            vocab
        )

        # Smaller angular distance = more similar
        best_angle = float("inf")
        best_odorant = None

        # Compare prediction with every GT vector
        for j in range(len(df)):

            gt_vec = build_gt_vec(
                df.iloc[j]["descriptor"],
                df.iloc[j]["weight"],
                vocab_index,
                vocab
            )

            angle = angular_distance(
                pred_vec,
                gt_vec
            )

            if angle < best_angle:

                best_angle = angle
                best_odorant = df.iloc[j]["odorant"]

        print("\nSample:", i)

        print(
            "BEST MATCH:",
            best_odorant
        )

        print(
            "ANGULAR DISTANCE:",
            round(best_angle, 4),
            "radians"
        )

        results.append(
            best_angle
        )

    print(
        "\n========================"
    )

    print(
        "AVG ANGULAR DISTANCE:",
        round(
            np.mean(results),
            4
        ),
        "radians"
    )

    print(
        "========================"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # Please specify the directory containing
    # the input mass-spectrum CSV files.
    INPUT_CSV_DIR = r"path/to/your/input_csv_directory"

    if not os.path.isdir(INPUT_CSV_DIR):
        raise FileNotFoundError(
            f"Input CSV directory not found: {INPUT_CSV_DIR}\n"
            "Please set INPUT_CSV_DIR to the correct CSV directory."
        )

    while True:

        # =========================
        # INPUT
        # =========================
        user_input = input(
            "\nEnter CSV filename (q to quit): "
        )

        if user_input.lower() == "q":
            break

        if user_input.strip() == "":

            filename = "1-butanol.csv"

        else:

            filename = user_input.strip()

            if not filename.lower().endswith(".csv"):
                filename += ".csv"

        file_path = os.path.join(
            INPUT_CSV_DIR,
            filename
        )

        if not os.path.isfile(file_path):
            print(
                f"\nFile not found: {file_path}"
            )
            continue

        print(
            f"\nProcessing: {file_path}"
        )

        # =========================
        # INFERENCE
        # =========================
        test_loader = to_dataloader(
            signal_processing(
                file_path
            )
        )

        (
            emb,
            ensemble_probs,
            all_fold_probs,
            all_pred_words,
            all_pred_probs
        ) = predict_with_ensemble(
            test_loader,
            label_names,
            device,
            base_path=base_path,
            num_folds=NUM_FOLDS,
            threshold=0.3
        )

        # =========================
        # ANGULAR DISTANCE EVALUATION
        # =========================
        run_angular_distance(
            all_pred_words
        )
