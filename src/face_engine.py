import pickle
import cv2
import numpy as np

from insightface.app import FaceAnalysis


class FaceRecognitionEngine:

    def __init__(
        self,
        embedding_file="models/face_embeddings.pkl",
        threshold=0.40
    ):

        self.embedding_file = embedding_file
        self.threshold = threshold

        # Load registered student
        with open(
            self.embedding_file,
            "rb"
        ) as file:

            data = pickle.load(file)

        self.student_id = data["student_id"]
        self.student_name = data["name"]

        self.reference_embedding = (
            data["reference_embedding"]
        )

        # Load InsightFace
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )


    # ======================================
    # Cosine Similarity
    # ======================================

    def cosine_similarity(self, a, b):

        return np.dot(a, b) / (
            np.linalg.norm(a) *
            np.linalg.norm(b)
        )


    # ======================================
    # Recognize Faces
    # ======================================

    def recognize(self, frame):

        faces = self.app.get(frame)

        results = []


        for face in faces:

            # Bounding box
            x1, y1, x2, y2 = (
                face.bbox.astype(int)
            )

            # Embedding
            embedding = face.embedding

            # Normalize
            embedding = (
                embedding /
                np.linalg.norm(embedding)
            )

            # Similarity
            similarity = (
                self.cosine_similarity(
                    embedding,
                    self.reference_embedding
                )
            )


            # Identity
            if similarity >= self.threshold:

                name = self.student_name

                student_id = self.student_id

                matched = True

            else:

                name = "Unknown"

                student_id = None

                matched = False


            # Store result
            results.append(
                {
                    "name": name,
                    "student_id": student_id,
                    "similarity": float(
                        similarity
                    ),
                    "matched": matched,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }
            )


        return results


    # ======================================
    # Draw Recognition Result
    # ======================================

    def draw_results(
        self,
        frame,
        results
    ):

        for result in results:

            x1, y1, x2, y2 = (
                result["bbox"]
            )

            name = result["name"]

            similarity = result[
                "similarity"
            ]

            matched = result[
                "matched"
            ]


            if matched:

                color = (
                    0,
                    255,
                    0
                )

                status = "MATCH"

            else:

                color = (
                    0,
                    0,
                    255
                )

                status = "UNKNOWN"


            # Face box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            # Name
            cv2.putText(
                frame,
                f"Name: {name}",
                (x1, y1 - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


            # Similarity
            cv2.putText(
                frame,
                f"Similarity: {similarity:.2f}",
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )


            # Status
            cv2.putText(
                frame,
                status,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )


        return frame