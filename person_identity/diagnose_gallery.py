import os
import numpy as np
from itertools import combinations


# ============================================================
# REID GALLERY DIAGNOSTIC
# ============================================================

print("=" * 70)
print("REID GALLERY DIAGNOSTIC")
print("=" * 70)


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

EMBEDDINGS_DIR = os.path.join(
    BASE_DIR,
    "person_identity",
    "embeddings"
)


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not os.path.exists(EMBEDDINGS_DIR):
    print("\nERROR: Embeddings directory not found:")
    print(EMBEDDINGS_DIR)
    raise SystemExit(1)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

gallery = {}

for filename in sorted(os.listdir(EMBEDDINGS_DIR)):

    if not filename.endswith(".npy"):
        continue

    # Expected:
    # person_001__01.npy

    parts = filename.split("__")

    if len(parts) != 2:
        print(f"Skipping unexpected file: {filename}")
        continue

    person_id = parts[0]

    path = os.path.join(
        EMBEDDINGS_DIR,
        filename
    )

    embedding = np.load(path)

    embedding = embedding.astype(np.float32)

    # Normalize
    norm = np.linalg.norm(embedding)

    if norm == 0:
        print(f"WARNING: Zero embedding: {filename}")
        continue

    embedding = embedding / norm

    if person_id not in gallery:
        gallery[person_id] = []

    gallery[person_id].append(
        (filename, embedding)
    )


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("GALLERY INFORMATION")
print("=" * 70)

print(f"\nEmbedding directory:")
print(EMBEDDINGS_DIR)

print(f"\nTotal identities: {len(gallery)}")

total_embeddings = sum(
    len(items)
    for items in gallery.values()
)

print(f"Total embeddings: {total_embeddings}")

for person_id in sorted(gallery):
    print(
        f"{person_id}: "
        f"{len(gallery[person_id])} embeddings"
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
    )


# ============================================================
# INTRA-PERSON SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("INTRA-PERSON SIMILARITY")
print("=" * 70)

print(
    "\nThis measures how similar different images "
    "of the SAME person are."
)


intra_results = {}


for person_id in sorted(gallery):

    items = gallery[person_id]

    similarities = []

    for (name1, emb1), (name2, emb2) in combinations(
        items,
        2
    ):

        score = cosine_similarity(
            emb1,
            emb2
        )

        similarities.append(
            (
                score,
                name1,
                name2
            )
        )

    intra_results[person_id] = similarities

    if not similarities:
        continue

    scores = [
        x[0]
        for x in similarities
    ]

    print("\n" + "-" * 70)
    print(f"{person_id}")
    print("-" * 70)

    print(
        f"Pairs       : {len(scores)}"
    )

    print(
        f"Average     : {np.mean(scores):.4f}"
    )

    print(
        f"Median      : {np.median(scores):.4f}"
    )

    print(
        f"Minimum     : {np.min(scores):.4f}"
    )

    print(
        f"Maximum     : {np.max(scores):.4f}"
    )

    print("\nLowest similarity pairs:")

    for score, name1, name2 in sorted(
        similarities
    )[:3]:

        print(
            f"  {name1} <-> {name2}"
            f" = {score:.4f}"
        )

    print("\nHighest similarity pairs:")

    for score, name1, name2 in sorted(
        similarities,
        reverse=True
    )[:3]:

        print(
            f"  {name1} <-> {name2}"
            f" = {score:.4f}"
        )


# ============================================================
# INTER-PERSON SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("INTER-PERSON SIMILARITY")
print("=" * 70)

print(
    "\nThis measures similarity between DIFFERENT people."
)


people = sorted(gallery.keys())


inter_results = {}


for person_a, person_b in combinations(
    people,
    2
):

    similarities = []

    for name_a, emb_a in gallery[person_a]:

        for name_b, emb_b in gallery[person_b]:

            score = cosine_similarity(
                emb_a,
                emb_b
            )

            similarities.append(
                (
                    score,
                    name_a,
                    name_b
                )
            )

    inter_results[
        (person_a, person_b)
    ] = similarities

    scores = [
        x[0]
        for x in similarities
    ]

    print("\n" + "-" * 70)

    print(
        f"{person_a}  <->  {person_b}"
    )

    print("-" * 70)

    print(
        f"Pairs       : {len(scores)}"
    )

    print(
        f"Average     : {np.mean(scores):.4f}"
    )

    print(
        f"Median      : {np.median(scores):.4f}"
    )

    print(
        f"Minimum     : {np.min(scores):.4f}"
    )

    print(
        f"Maximum     : {np.max(scores):.4f}"
    )

    print("\nMost similar cross-person pairs:")

    for score, name_a, name_b in sorted(
        similarities,
        reverse=True
    )[:5]:

        print(
            f"  {name_a} <-> {name_b}"
            f" = {score:.4f}"
        )


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("IDENTITY SEPARATION ANALYSIS")
print("=" * 70)


all_intra_scores = []

for similarities in intra_results.values():

    for score, _, _ in similarities:

        all_intra_scores.append(
            score
        )


all_inter_scores = []

for similarities in inter_results.values():

    for score, _, _ in similarities:

        all_inter_scores.append(
            score
        )


if all_intra_scores:

    print(
        f"\nSame-person average similarity:"
        f" {np.mean(all_intra_scores):.4f}"
    )

    print(
        f"Same-person minimum:"
        f" {np.min(all_intra_scores):.4f}"
    )

    print(
        f"Same-person maximum:"
        f" {np.max(all_intra_scores):.4f}"
    )


if all_inter_scores:

    print(
        f"\nDifferent-person average similarity:"
        f" {np.mean(all_inter_scores):.4f}"
    )

    print(
        f"Different-person minimum:"
        f" {np.min(all_inter_scores):.4f}"
    )

    print(
        f"Different-person maximum:"
        f" {np.max(all_inter_scores):.4f}"
    )


# ============================================================
# OVERALL GAP
# ============================================================

if all_intra_scores and all_inter_scores:

    same_avg = np.mean(
        all_intra_scores
    )

    different_avg = np.mean(
        all_inter_scores
    )

    gap = same_avg - different_avg

    print(
        f"\nOverall separation gap:"
        f" {gap:.4f}"
    )

    print("\nInterpretation:")

    if gap >= 0.20:

        print(
            "GOOD: Strong separation between "
            "same-person and different-person images."
        )

    elif gap >= 0.10:

        print(
            "MODERATE: Some identity separation "
            "exists, but there is room for improvement."
        )

    else:

        print(
            "WEAK: Same-person and different-person "
            "similarities overlap significantly."
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)

print(
    "\nNo gallery files were modified."
)

print(
    "No embeddings were modified."
)

print(
    "No database changes were made."
)

print(
    "\nUse the output above to evaluate "
    "the quality of the current ReID gallery."
)

print("=" * 70)