import re
from collections import Counter


class MultiFrameOCR:
    """
    Collects OCR readings from multiple video frames and selects
    the most reliable plate reading.
    """

    def __init__(
        self,
        min_confidence=0.25,
        min_observations=2,
        similarity_threshold=0.65
    ):
        self.min_confidence = min_confidence
        self.min_observations = min_observations
        self.similarity_threshold = similarity_threshold

        self.observations = []

    # ---------------------------------------------------------
    # Normalize OCR text
    # ---------------------------------------------------------

    def normalize(self, text):
        if not text:
            return ""

        text = text.upper()

        # Keep only letters and numbers
        text = re.sub(r"[^A-Z0-9]", "", text)

        return text

    # ---------------------------------------------------------
    # Add OCR observation
    # ---------------------------------------------------------

    def add_observation(
        self,
        text,
        ocr_confidence,
        frame_number,
        yolo_confidence=0.0
    ):

        text = self.normalize(text)

        if not text:
            return False

        if ocr_confidence < self.min_confidence:
            return False

        self.observations.append({
            "text": text,
            "ocr_confidence": float(ocr_confidence),
            "yolo_confidence": float(yolo_confidence),
            "frame": frame_number
        })

        return True

    # ---------------------------------------------------------
    # Get all observations
    # ---------------------------------------------------------

    def get_observations(self):
        return self.observations

    # ---------------------------------------------------------
    # Select best plate
    # ---------------------------------------------------------

    def get_best_plate(self):

        if not self.observations:
            return None

        # -----------------------------------------------------
        # Group exact OCR readings
        # -----------------------------------------------------

        groups = {}

        for obs in self.observations:

            plate = obs["text"]

            if plate not in groups:
                groups[plate] = []

            groups[plate].append(obs)

        # -----------------------------------------------------
        # Calculate score for each reading
        # -----------------------------------------------------

        candidates = []

        for plate, observations in groups.items():

            count = len(observations)

            avg_ocr = sum(
                x["ocr_confidence"]
                for x in observations
            ) / count

            avg_yolo = sum(
                x["yolo_confidence"]
                for x in observations
            ) / count

            # Repeated observations are more valuable.
            score = (
                (count * 0.50)
                + (avg_ocr * 0.35)
                + (avg_yolo * 0.15)
            )

            candidates.append({
                "plate": plate,
                "count": count,
                "avg_ocr": avg_ocr,
                "avg_yolo": avg_yolo,
                "score": score,
                "frames": [
                    x["frame"]
                    for x in observations
                ]
            })

        # Highest score first
        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        best = candidates[0]

        # -----------------------------------------------------
        # Confirmation rule
        # -----------------------------------------------------

        # We confirm when:
        #
        # 1. Same reading appears at least twice
        # OR
        # 2. OCR confidence is very high

        confirmed = (
            best["count"] >= self.min_observations
            or best["avg_ocr"] >= 0.75
        )

        best["confirmed"] = confirmed

        return best

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    def print_summary(self):

        print()
        print("=" * 60)
        print("MULTI-FRAME OCR SUMMARY")
        print("=" * 60)

        if not self.observations:

            print("No valid OCR observations.")

            return

        for plate, observations in Counter(
            x["text"]
            for x in self.observations
        ).most_common():

            matching = [
                x
                for x in self.observations
                if x["text"] == plate
            ]

            avg_conf = sum(
                x["ocr_confidence"]
                for x in matching
            ) / len(matching)

            print(
                f"{plate:<15} "
                f"Count: {len(matching):<3} "
                f"Avg OCR: {avg_conf:.2f}"
            )

        best = self.get_best_plate()

        print()
        print("-" * 60)

        if best and best["confirmed"]:

            print("CONFIRMED PLATE")
            print("-" * 60)
            print(f"Plate          : {best['plate']}")
            print(f"Observations   : {best['count']}")
            print(f"Average OCR    : {best['avg_ocr']:.2f}")
            print(f"Average YOLO   : {best['avg_yolo']:.2f}")
            print(f"Frames         : {best['frames']}")

        else:

            print("NO PLATE CONFIRMED")

        print("=" * 60)


# ============================================================
# TEST THE MODULE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MULTI-FRAME OCR TEST")
    print("=" * 60)

    ocr = MultiFrameOCR(
        min_confidence=0.25,
        min_observations=2
    )

    # Example readings similar to what your video produced.

    test_readings = [
        ("F215", 0.51, 108, 0.64),
        ("F215", 0.48, 109, 0.70),
        ("FIC2015", 0.12, 110, 0.75),
        ("F215", 0.55, 111, 0.72),
        ("F215", 0.46, 112, 0.69),
        ("NC2022", 0.22, 118, 0.68),
        ("NI2022", 0.38, 119, 0.72),
    ]

    for text, ocr_conf, frame, yolo_conf in test_readings:

        added = ocr.add_observation(
            text=text,
            ocr_confidence=ocr_conf,
            frame_number=frame,
            yolo_confidence=yolo_conf
        )

        print(
            f"Frame {frame:<4} "
            f"OCR: {text:<10} "
            f"Confidence: {ocr_conf:.2f} "
            f"Added: {added}"
        )

    ocr.print_summary()