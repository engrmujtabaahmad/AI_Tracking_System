import os
import itertools
import numpy as np


# ============================================================
# PERSON IDENTITY GALLERY EVALUATION - LEAVE ONE OUT
# ============================================================

print("=" * 70)
print("PERSON IDENTITY GALLERY EVALUATION - LEAVE ONE OUT")
print("=" * 70)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PERSON_IDENTITY_DIR = os.path.join(
    BASE_DIR,
    "person_identity"
)

EMBEDDINGS_DIR = os.path.join(
    PERSON_IDENTITY_DIR,
    "embeddings"
)


# ============================================================
# SETTINGS
# ============================================================

TOP_K = 3

# Minimum number of gallery images required for a valid
# identity comparison after removing the query image.
MIN_GALLERY_IMAGES = 1


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator <= 0:

        return 0.0

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(
    embedding
):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm <= 0:

        return embedding

    return (
        embedding / norm
    )


# ============================================================
# LOAD GALLERY
# ============================================================

def load_gallery():

    print("")
    print("=" * 70)
    print("LOADING GALLERY")
    print("=" * 70)

    if not os.path.exists(
        EMBEDDINGS_DIR
    ):

        print("")
        print(
            "ERROR: Embeddings directory does not exist:"
        )

        print(
            EMBEDDINGS_DIR
        )

        return {}

    gallery = {}

    for filename in sorted(
        os.listdir(
            EMBEDDINGS_DIR
        )
    ):

        if not filename.lower().endswith(
            ".npy"
        ):

            continue

        path = os.path.join(
            EMBEDDINGS_DIR,
            filename
        )

        try:

            embedding = np.load(
                path
            )

            embedding = normalize_embedding(
                embedding
            )

            person_id = filename.split(
                "__"
            )[0]

            if person_id not in gallery:

                gallery[person_id] = []

            gallery[person_id].append(
                {
                    "filename":
                        filename,

                    "embedding":
                        embedding
                }
            )

        except Exception as e:

            print(
                f"ERROR loading {filename}: {e}"
            )

    print("")

    for person_id in sorted(
        gallery
    ):

        print(
            f"{person_id}: "
            f"{len(gallery[person_id])} embeddings"
        )

    print("")

    print(
        f"Total identities: "
        f"{len(gallery)}"
    )

    print(
        f"Total embeddings: "
        f"{sum(len(v) for v in gallery.values())}"
    )

    return gallery


# ============================================================
# COMPARE QUERY AGAINST ONE IDENTITY
# ============================================================

def compare_identity(
    query_embedding,
    person_gallery
):

    scores = []

    for item in person_gallery:

        score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        scores.append(
            {
                "filename":
                    item["filename"],

                "score":
                    score
            }
        )

    if not scores:

        return None

    scores.sort(
        key=lambda x:
        x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Best similarity
    # --------------------------------------------------------

    best_similarity = (
        scores[0]["score"]
    )

    # --------------------------------------------------------
    # Top-K similarities
    # --------------------------------------------------------

    top_scores = [
        item["score"]
        for item in scores[
            :min(
                TOP_K,
                len(scores)
            )
        ]
    ]

    top_average = float(
        np.mean(
            top_scores
        )
    )

    top_median = float(
        np.median(
            top_scores
        )
    )

    # --------------------------------------------------------
    # ALL IMAGE average
    # --------------------------------------------------------

    all_scores = [
        item["score"]
        for item in scores
    ]

    all_average = float(
        np.mean(
            all_scores
        )
    )

    # --------------------------------------------------------
    # Balanced identity score
    #
    # Best image should matter,
    # but multiple supporting images
    # should also matter.
    # --------------------------------------------------------

    identity_score = (
        0.45 * best_similarity
        +
        0.35 * top_average
        +
        0.20 * all_average
    )

    return {

        "best":
            best_similarity,

        "top_average":
            top_average,

        "top_median":
            top_median,

        "all_average":
            all_average,

        "identity_score":
            identity_score,

        "scores":
            scores
    }


# ============================================================
# LEAVE-ONE-OUT EVALUATION
# ============================================================

def evaluate_leave_one_out(
    gallery
):

    print("")
    print("=" * 70)
    print("LEAVE-ONE-OUT IMAGE-LEVEL EVALUATION")
    print("=" * 70)

    print("")
    print(
        "Each query image is REMOVED from its own identity"
    )

    print(
        "before identity prediction."
    )

    print("")
    print(
        "This prevents exact self-matching."
    )

    identities = sorted(
        gallery.keys()
    )

    total = 0
    correct = 0

    image_results = []

    # --------------------------------------------------------
    # Per-person statistics
    # --------------------------------------------------------

    per_person = {}

    for person_id in identities:

        per_person[person_id] = {
            "total": 0,
            "correct": 0
        }

    # --------------------------------------------------------
    # Every image becomes a query
    # --------------------------------------------------------

    for true_person in identities:

        for query_item in gallery[
            true_person
        ]:

            query_filename = (
                query_item["filename"]
            )

            query_embedding = normalize_embedding(
                query_item["embedding"]
            )

            comparisons = []

            # ------------------------------------------------
            # Compare against every identity
            # ------------------------------------------------

            for candidate_person in identities:

                candidate_gallery = []

                for gallery_item in gallery[
                    candidate_person
                ]:

                    # IMPORTANT:
                    #
                    # Remove the exact query image.
                    #

                    if (
                        candidate_person
                        ==
                        true_person
                        and
                        gallery_item["filename"]
                        ==
                        query_filename
                    ):

                        continue

                    candidate_gallery.append(
                        gallery_item
                    )

                if len(
                    candidate_gallery
                ) < MIN_GALLERY_IMAGES:

                    continue

                result = compare_identity(
                    query_embedding,
                    candidate_gallery
                )

                if result is None:

                    continue

                result["person_id"] = (
                    candidate_person
                )

                comparisons.append(
                    result
                )

            if not comparisons:

                continue

            # ------------------------------------------------
            # Rank identities
            # ------------------------------------------------

            comparisons.sort(
                key=lambda x:
                x["identity_score"],
                reverse=True
            )

            best = comparisons[0]

            second = (
                comparisons[1]
                if len(comparisons) > 1
                else None
            )

            if second is not None:

                margin = (
                    best["identity_score"]
                    -
                    second["identity_score"]
                )

            else:

                margin = 1.0

            predicted_person = (
                best["person_id"]
            )

            is_correct = (
                predicted_person
                ==
                true_person
            )

            total += 1

            per_person[
                true_person
            ]["total"] += 1

            if is_correct:

                correct += 1

                per_person[
                    true_person
                ]["correct"] += 1

            image_results.append(
                {
                    "true_person":
                        true_person,

                    "query":
                        query_filename,

                    "predicted":
                        predicted_person,

                    "identity_score":
                        best[
                            "identity_score"
                        ],

                    "best_similarity":
                        best[
                            "best"
                        ],

                    "top_average":
                        best[
                            "top_average"
                        ],

                    "all_average":
                        best[
                            "all_average"
                        ],

                    "margin":
                        margin,

                    "correct":
                        is_correct,

                    "comparisons":
                        comparisons
                }
            )

            status = (
                "CORRECT"
                if is_correct
                else "WRONG"
            )

            print("")

            print(
                f"{status:<8} "
                f"True={true_person:<12} "
                f"Query={query_filename:<25} "
                f"Predicted={predicted_person:<12} "
                f"Score={best['identity_score']:.4f} "
                f"Best={best['best']:.4f} "
                f"Margin={margin:.4f}"
            )

    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    print("")
    print("=" * 70)
    print("OVERALL ACCURACY")
    print("=" * 70)

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print("")

    print(
        f"Correct : "
        f"{correct}/{total}"
    )

    print(
        f"Wrong   : "
        f"{total - correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ========================================================
    # PER PERSON ACCURACY
    # ========================================================

    print("")
    print("=" * 70)
    print("PER-PERSON ACCURACY")
    print("=" * 70)

    for person_id in identities:

        stats = per_person[
            person_id
        ]

        person_accuracy = (
            stats["correct"]
            /
            stats["total"]
            if stats["total"] > 0
            else 0.0
        )

        print(
            f"{person_id:<15}"
            f"Correct="
            f"{stats['correct']}/"
            f"{stats['total']}   "
            f"Accuracy="
            f"{person_accuracy * 100:.2f}%"
        )

    return image_results


# ============================================================
# CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    image_results,
    gallery
):

    print("")
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    identities = sorted(
        gallery.keys()
    )

    matrix = {}

    for true_person in identities:

        matrix[
            true_person
        ] = {}

        for predicted_person in identities:

            matrix[
                true_person
            ][
                predicted_person
            ] = 0

    for result in image_results:

        true_person = result[
            "true_person"
        ]

        predicted_person = result[
            "predicted"
        ]

        if (
            true_person in matrix
            and
            predicted_person
            in matrix[true_person]
        ):

            matrix[
                true_person
            ][
                predicted_person
            ] += 1

    print("")

    print(
        f"{'TRUE / PRED':<18}",
        end=""
    )

    for person_id in identities:

        print(
            f"{person_id:<15}",
            end=""
        )

    print("")

    print(
        "-" * (
            18
            +
            15 * len(identities)
        )
    )

    for true_person in identities:

        print(
            f"{true_person:<18}",
            end=""
        )

        for predicted_person in identities:

            print(
                f"{matrix[true_person][predicted_person]:<15}",
                end=""
            )

        print("")


# ============================================================
# DIFFICULT IMAGES
# ============================================================

def analyze_difficult_images(
    image_results
):

    print("")
    print("=" * 70)
    print("DIFFICULT / AMBIGUOUS IMAGES")
    print("=" * 70)

    difficult = []

    for result in image_results:

        if (
            not result["correct"]
            or
            result["margin"] < 0.05
        ):

            difficult.append(
                result
            )

    difficult.sort(
        key=lambda x:
        (
            x["correct"],
            x["margin"]
        )
    )

    if not difficult:

        print("")
        print(
            "No difficult images detected."
        )

        return

    for result in difficult:

        print("")

        print(
            f"True identity : "
            f"{result['true_person']}"
        )

        print(
            f"Image         : "
            f"{result['query']}"
        )

        print(
            f"Predicted     : "
            f"{result['predicted']}"
        )

        print(
            f"Identity score: "
            f"{result['identity_score']:.4f}"
        )

        print(
            f"Best similarity: "
            f"{result['best_similarity']:.4f}"
        )

        print(
            f"Top-{TOP_K} average: "
            f"{result['top_average']:.4f}"
        )

        print(
            f"Margin        : "
            f"{result['margin']:.4f}"
        )

        print(
            f"Correct       : "
            f"{result['correct']}"
        )


# ============================================================
# SAME-PERSON SIMILARITY
# ============================================================

def calculate_same_person_similarity(
    gallery
):

    same_scores = []

    identities = sorted(
        gallery.keys()
    )

    for person_id in identities:

        embeddings = gallery[
            person_id
        ]

        for a, b in itertools.combinations(
            embeddings,
            2
        ):

            score = cosine_similarity(
                a["embedding"],
                b["embedding"]
            )

            same_scores.append(
                score
            )

    return same_scores


# ============================================================
# DIFFERENT-PERSON SIMILARITY
# ============================================================

def calculate_different_person_similarity(
    gallery
):

    different_scores = []

    identities = sorted(
        gallery.keys()
    )

    for person_a, person_b in itertools.combinations(
        identities,
        2
    ):

        for image_a in gallery[
            person_a
        ]:

            for image_b in gallery[
                person_b
            ]:

                score = cosine_similarity(
                    image_a["embedding"],
                    image_b["embedding"]
                )

                different_scores.append(
                    score
                )

    return different_scores


# ============================================================
# GLOBAL SIMILARITY ANALYSIS
# ============================================================

def analyze_global_similarity(
    gallery
):

    print("")
    print("=" * 70)
    print("GLOBAL SIMILARITY ANALYSIS")
    print("=" * 70)

    same_scores = (
        calculate_same_person_similarity(
            gallery
        )
    )

    different_scores = (
        calculate_different_person_similarity(
            gallery
        )
    )

    if not same_scores:

        print(
            "Not enough same-person data."
        )

        return

    if not different_scores:

        print(
            "Not enough different-person data."
        )

        return

    same_average = float(
        np.mean(
            same_scores
        )
    )

    same_min = float(
        np.min(
            same_scores
        )
    )

    same_max = float(
        np.max(
            same_scores
        )
    )

    different_average = float(
        np.mean(
            different_scores
        )
    )

    different_min = float(
        np.min(
            different_scores
        )
    )

    different_max = float(
        np.max(
            different_scores
        )
    )

    separation_gap = (
        same_average
        -
        different_average
    )

    print("")

    print(
        f"Same-person average      : "
        f"{same_average:.4f}"
    )

    print(
        f"Same-person minimum      : "
        f"{same_min:.4f}"
    )

    print(
        f"Same-person maximum      : "
        f"{same_max:.4f}"
    )

    print("")

    print(
        f"Different-person average : "
        f"{different_average:.4f}"
    )

    print(
        f"Different-person minimum : "
        f"{different_min:.4f}"
    )

    print(
        f"Different-person maximum : "
        f"{different_max:.4f}"
    )

    print("")

    print(
        f"Separation gap           : "
        f"{separation_gap:.4f}"
    )

    print("")

    if separation_gap >= 0.15:

        quality = "GOOD"

    elif separation_gap >= 0.08:

        quality = "MODERATE"

    elif separation_gap >= 0.03:

        quality = "WEAK"

    else:

        quality = "VERY WEAK"

    print(
        f"QUALITY: {quality}"
    )


# ============================================================
# TOP CROSS-PERSON CONFUSIONS
# ============================================================

def analyze_cross_person_confusion(
    gallery
):

    print("")
    print("=" * 70)
    print("TOP CROSS-PERSON CONFUSIONS")
    print("=" * 70)

    identities = sorted(
        gallery.keys()
    )

    confusion_pairs = []

    for person_a, person_b in itertools.combinations(
        identities,
        2
    ):

        for image_a in gallery[
            person_a
        ]:

            for image_b in gallery[
                person_b
            ]:

                score = cosine_similarity(
                    image_a["embedding"],
                    image_b["embedding"]
                )

                confusion_pairs.append(
                    {
                        "person_a":
                            person_a,

                        "image_a":
                            image_a["filename"],

                        "person_b":
                            person_b,

                        "image_b":
                            image_b["filename"],

                        "score":
                            score
                    }
                )

    confusion_pairs.sort(
        key=lambda x:
        x["score"],
        reverse=True
    )

    print("")

    for item in confusion_pairs[:15]:

        print(
            f"{item['person_a']} / "
            f"{item['image_a']}"
            f"  <->  "
            f"{item['person_b']} / "
            f"{item['image_b']}"
            f"  =  "
            f"{item['score']:.4f}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    image_results
):

    print("")
    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    total = len(
        image_results
    )

    correct = sum(
        1
        for result in image_results
        if result["correct"]
    )

    wrong = (
        total
        -
        correct
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print("")

    print(
        f"Total queries : {total}"
    )

    print(
        f"Correct       : {correct}"
    )

    print(
        f"Wrong         : {wrong}"
    )

    print(
        f"Accuracy      : "
        f"{accuracy * 100:.2f}%"
    )

    print("")

    if accuracy >= 0.80:

        print(
            "RESULT: GOOD - ReID is performing well."
        )

    elif accuracy >= 0.60:

        print(
            "RESULT: MODERATE - ReID needs improvement."
        )

    elif accuracy >= 0.40:

        print(
            "RESULT: WEAK - ReID needs significant improvement."
        )

    else:

        print(
            "RESULT: VERY WEAK - Do not integrate into "
            "the final tracking system yet."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load gallery
    # --------------------------------------------------------

    gallery = load_gallery()

    if not gallery:

        return 1

    # --------------------------------------------------------
    # Leave-one-out evaluation
    # --------------------------------------------------------

    image_results = evaluate_leave_one_out(
        gallery
    )

    if not image_results:

        print("")
        print(
            "ERROR: No evaluation results generated."
        )

        return 1

    # --------------------------------------------------------
    # Per-person confusion matrix
    # --------------------------------------------------------

    print_confusion_matrix(
        image_results,
        gallery
    )

    # --------------------------------------------------------
    # Difficult images
    # --------------------------------------------------------

    analyze_difficult_images(
        image_results
    )

    # --------------------------------------------------------
    # Global similarity
    # --------------------------------------------------------

    analyze_global_similarity(
        gallery
    )

    # --------------------------------------------------------
    # Cross-person confusion
    # --------------------------------------------------------

    analyze_cross_person_confusion(
        gallery
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_final_summary(
        image_results
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print("")

    print(
        "IMPORTANT:"
    )

    print(
        "This evaluation does NOT modify your gallery."
    )

    print(
        "Every query image is excluded from its own "
        "identity gallery during prediction."
    )

    print(
        "This is a leave-one-out gallery evaluation."
    )

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )