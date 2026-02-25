"""LocalInsightFaceProvider — wraps insightface for face detection, age estimation, and embedding generation."""

import logging

import cv2
import numpy as np

from .base import AIProvider, AgeEstimationResult, FaceDetectionResult
from services.age_rules import classify_age_group
from utils.embedding import l2_normalize

logger = logging.getLogger(__name__)


class LocalInsightFaceProvider(AIProvider):
    """AI provider backed by InsightFace (RetinaFace + ArcFace) with MTCNN fallback."""

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0, det_size: tuple[int, int] = (640, 640)):
        self._model_name = model_name
        self._ctx_id = ctx_id
        self._det_size = det_size
        self._app = None
        self._mtcnn = None
        self._loaded = False
        # Cache the last InsightFace face object from detect_face so that
        # estimate_age and generate_embedding don't need to re-detect.
        self._last_faces: dict[int, object] = {}  # keyed by id(face_image)

    def load_models(self) -> None:
        """Load InsightFace FaceAnalysis and MTCNN fallback models."""
        import insightface

        self._app = insightface.app.FaceAnalysis(name=self._model_name)
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)

        from facenet_pytorch import MTCNN

        self._mtcnn = MTCNN(keep_all=True, device="cpu")
        self._loaded = True
        logger.info("InsightFace and MTCNN models loaded successfully.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_insightface(self, image: np.ndarray) -> list:
        """Run InsightFace analysis on an image and return detected face objects."""
        if self._app is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._app.get(image)

    def _run_mtcnn_fallback(self, image: np.ndarray) -> list:
        """Use MTCNN as a fallback detector, returning pseudo-face objects with bounding boxes."""
        if self._mtcnn is None:
            return []

        # MTCNN expects RGB input
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image

        pil_image = Image.fromarray(rgb_image)
        boxes, probs = self._mtcnn.detect(pil_image)

        if boxes is None or len(boxes) == 0:
            return []

        results = []
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # Clamp to image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            results.append({"bbox": (x1, y1, x2, y2), "det_score": float(prob)})
        return results

    def _crop_face(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Crop a face region from the image using the bounding box."""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2].copy()

    # ------------------------------------------------------------------
    # AIProvider interface
    # ------------------------------------------------------------------

    async def detect_face(self, image: np.ndarray) -> FaceDetectionResult:
        """Detect exactly one face using RetinaFace, falling back to MTCNN if needed."""
        faces = self._run_insightface(image)

        # Fallback to MTCNN when RetinaFace finds nothing
        if len(faces) == 0:
            logger.info("RetinaFace detected no faces, falling back to MTCNN.")
            mtcnn_results = self._run_mtcnn_fallback(image)
            if len(mtcnn_results) == 0:
                raise ValueError("No face detected in the image")
            if len(mtcnn_results) > 1:
                raise ValueError(
                    "Multiple faces detected; please upload an image with exactly one face"
                )
            # MTCNN fallback — crop and return without InsightFace attributes
            det = mtcnn_results[0]
            bbox = det["bbox"]
            face_image = self._crop_face(image, bbox)
            return FaceDetectionResult(
                face_image=face_image,
                bbox=bbox,
                confidence=det["det_score"],
            )

        if len(faces) > 1:
            raise ValueError(
                "Multiple faces detected; please upload an image with exactly one face"
            )

        face = faces[0]
        bbox = tuple(int(c) for c in face.bbox)
        face_image = self._crop_face(image, bbox)
        result = FaceDetectionResult(
            face_image=face_image,
            bbox=bbox,
            confidence=float(face.det_score),
        )
        # Cache the InsightFace face object keyed by the cropped image's id
        self._last_faces[id(result.face_image)] = face
        return result

    async def estimate_age(self, face_image: np.ndarray) -> AgeEstimationResult:
        """Estimate age, using cached face data from detect_face when available."""
        cached = self._last_faces.get(id(face_image))
        if cached is not None and hasattr(cached, 'age'):
            age = int(cached.age)
            age_group = classify_age_group(age)
            return AgeEstimationResult(age=age, age_group=age_group)

        # Fallback: re-run detection on the image
        faces = self._run_insightface(face_image)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        age = int(faces[0].age)
        age_group = classify_age_group(age)
        return AgeEstimationResult(age=age, age_group=age_group)

    async def generate_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Generate a 512-dim L2-normalized embedding, using cached face data when available."""
        cached = self._last_faces.pop(id(face_image), None)
        if cached is not None and hasattr(cached, 'embedding'):
            return l2_normalize(cached.embedding)

        # Fallback: re-run detection on the image
        faces = self._run_insightface(face_image)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        embedding = faces[0].embedding
        return l2_normalize(embedding)

    async def is_loaded(self) -> bool:
        """Return whether models have been loaded."""
        return self._loaded
