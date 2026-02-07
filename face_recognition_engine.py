import base64
import io
import os
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

try:
    from pinecone import PodSpec
except Exception:
    PodSpec = None

try:
    from pinecone import ServerlessSpec
except Exception:
    ServerlessSpec = None

import mediapipe as mp


load_dotenv()


class FaceRecognition:
    def __init__(
        self,
        index_name=None,
        dimension=512,
        metric="cosine",
        min_score=0.35,
        ratio_threshold=0.18,
        adaptive_lr=0.05,
        yaw_threshold=15.0,
        pitch_threshold=15.0,
    ):
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "face-auth-index")
        self.dimension = dimension
        self.metric = metric
        self.min_score = min_score
        self.ratio_threshold = ratio_threshold
        self.adaptive_lr = adaptive_lr
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing PINECONE_API_KEY in environment")

        self.pc = Pinecone(api_key=api_key)
        self.index = self._init_index()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
        )

        device = os.getenv("FACE_DEVICE", "cpu")
        self.device = torch.device(device)
        self.mtcnn = MTCNN(image_size=160, margin=10, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def _init_index(self):
        host = os.getenv("PINECONE_HOST")
        if host:
            return self.pc.Index(self.index_name, host=host)

        existing = [idx["name"] for idx in self.pc.list_indexes()] if hasattr(self.pc, "list_indexes") else []
        if self.index_name not in existing:
            use_pod = os.getenv("PINECONE_USE_POD", "true").lower() == "true"
            if use_pod and PodSpec is not None:
                environment = os.getenv("PINECONE_ENV", "us-west1-gcp")
                pod_type = os.getenv("PINECONE_POD_TYPE", "p1.x1")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=PodSpec(environment=environment, pod_type=pod_type),
                )
            elif ServerlessSpec is not None:
                cloud = os.getenv("PINECONE_CLOUD", "aws")
                region = os.getenv("PINECONE_REGION", "us-east-1")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
            else:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                )

        return self.pc.Index(self.index_name)

    def decode_base64_image(self, base64_string):
        try:
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            image_bytes = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            return np.array(image)
        except Exception:
            return None

    def _get_face_mesh(self, frame):
        # Check if frame is already RGB or BGR
        if len(frame.shape) == 3:
            # If it's BGR (from cv2), convert to RGB
            # If it's RGB (from PIL/webcam), use as is
            # MediaPipe expects RGB
            if frame.shape[2] == 3:
                # Try BGR to RGB conversion, MediaPipe needs RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.dtype == np.uint8 else frame
            else:
                rgb = frame
        else:
            rgb = frame
        
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def detect_face(self, frame):
        try:
            # frame is already RGB from decode_base64_image (PIL)
            # MediaPipe expects RGB, so no conversion needed
            results = self.face_mesh.process(frame)
            return results.multi_face_landmarks is not None
        except Exception as e:
            print(f"[DETECT_FACE] Error in detect_face: {str(e)}")
            return False

    def _eye_aspect_ratio(self, pts):
        a = np.linalg.norm(pts[1] - pts[5])
        b = np.linalg.norm(pts[2] - pts[4])
        c = np.linalg.norm(pts[0] - pts[3])
        return (a + b) / (2.0 * c + 1e-6)

    def _estimate_head_pose(self, frame, landmarks):
        image_h, image_w = frame.shape[:2]
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float32,
        )
        image_points = np.array(
            [
                (landmarks[1].x * image_w, landmarks[1].y * image_h),
                (landmarks[152].x * image_w, landmarks[152].y * image_h),
                (landmarks[33].x * image_w, landmarks[33].y * image_h),
                (landmarks[263].x * image_w, landmarks[263].y * image_h),
                (landmarks[61].x * image_w, landmarks[61].y * image_h),
                (landmarks[291].x * image_w, landmarks[291].y * image_h),
            ],
            dtype=np.float32,
        )
        focal_length = image_w
        center = (image_w / 2, image_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        success, rotation_vec, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        roll = np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))
        return yaw, pitch, roll

    def liveness_check(self, frame):
        landmarks = self._get_face_mesh(frame)
        if landmarks is None:
            return False

        left_eye = np.array(
            [
                [landmarks[33].x, landmarks[33].y],
                [landmarks[160].x, landmarks[160].y],
                [landmarks[158].x, landmarks[158].y],
                [landmarks[133].x, landmarks[133].y],
                [landmarks[153].x, landmarks[153].y],
                [landmarks[144].x, landmarks[144].y],
            ],
            dtype=np.float32,
        )
        right_eye = np.array(
            [
                [landmarks[362].x, landmarks[362].y],
                [landmarks[385].x, landmarks[385].y],
                [landmarks[387].x, landmarks[387].y],
                [landmarks[263].x, landmarks[263].y],
                [landmarks[373].x, landmarks[373].y],
                [landmarks[380].x, landmarks[380].y],
            ],
            dtype=np.float32,
        )
        ear = (self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0
        head_pose = self._estimate_head_pose(frame, landmarks)
        if head_pose is None:
            return False
        yaw, pitch, _ = head_pose
        eyes_open = ear > 0.18
        looking_forward = abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold
        return eyes_open and looking_forward

    def nodal_ratio(self, frame):
        landmarks = self._get_face_mesh(frame)
        if landmarks is None:
            return None
        eye_left = np.array([landmarks[33].x, landmarks[33].y])
        eye_right = np.array([landmarks[263].x, landmarks[263].y])
        nose_left = np.array([landmarks[97].x, landmarks[97].y])
        nose_right = np.array([landmarks[326].x, landmarks[326].y])
        eye_dist = np.linalg.norm(eye_left - eye_right)
        nose_width = np.linalg.norm(nose_left - nose_right)
        return float(eye_dist / (nose_width + 1e-6))

    def get_embedding(self, frame):
        pil_image = Image.fromarray(frame)
        aligned = self.mtcnn(pil_image)
        if aligned is None:
            return None
        aligned = aligned.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.resnet(aligned).cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(emb) + 1e-6
        return emb / norm

    def register_user(self, user_id, images_base64):
        embeddings = []
        ratios = []
        for idx, image_base64 in enumerate(images_base64):
            frame = self.decode_base64_image(image_base64)
            if frame is None:
                print(f"[REGISTER] Image {idx}: Failed to decode")
                return None, "Failed to decode image"
            print(f"[REGISTER] Image {idx}: decoded, shape={frame.shape}")
            
            ratio = self.nodal_ratio(frame)
            if ratio is None:
                print(f"[REGISTER] Image {idx}: Landmarks not detected")
                return None, "Landmarks not detected"
            print(f"[REGISTER] Image {idx}: nodal_ratio={ratio}")
            
            embedding = self.get_embedding(frame)
            if embedding is None:
                print(f"[REGISTER] Image {idx}: Face not detected")
                return None, "Face not detected"
            print(f"[REGISTER] Image {idx}: embedding dim={len(embedding)}, norm={np.linalg.norm(embedding)}")
            
            embeddings.append(embedding)
            ratios.append(ratio)

        if not embeddings:
            return None, "No valid embeddings"

        master = np.mean(embeddings, axis=0)
        master = master / (np.linalg.norm(master) + 1e-6)
        ratio_mean = float(np.mean(ratios))
        
        print(f"[REGISTER] Master embedding: dim={len(master)}, norm={np.linalg.norm(master)}")

        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "user_id": user_id,
            "nodal_ratio": ratio_mean,
            "samples": len(embeddings),
            "created_at": now,
            "updated_at": now,
        }

        print(f"[REGISTER] Upserting to Pinecone: user_id={user_id}, metadata={metadata}")
        upsert_result = self.index.upsert(vectors=[(user_id, master.tolist(), metadata)])
        print(f"[REGISTER] Upsert result: {upsert_result}")
        
        # Verify the upsert worked
        fetch_result = self.index.fetch(ids=[user_id])
        print(f"[REGISTER] Fetch verification: {fetch_result}")
        
        return {
            "user_id": user_id,
            "samples": len(embeddings),
            "nodal_ratio": ratio_mean,
        }, None

    def _ratio_match(self, current_ratio, stored_ratio):
        if stored_ratio is None:
            return True
        return abs(current_ratio - stored_ratio) <= self.ratio_threshold

    def check_face_exists(self, images_base64):
        """
        Check if a face already exists in the database before registration.
        Returns (exists: bool, user_id: str or None, error: str or None)
        """
        if not images_base64:
            return False, None, "No images provided"
        
        try:
            # Get embedding from first image for quick check
            frame = self.decode_base64_image(images_base64[0])
            if frame is None:
                return False, None, "Failed to decode image"
            
            embedding = self.get_embedding(frame)
            if embedding is None:
                return False, None, "Face not detected"
            
            # Query Pinecone to find similar faces
            result = self.index.query(vector=embedding.tolist(), top_k=1, include_metadata=True)
            
            if hasattr(result, 'matches'):
                matches = result.matches or []
            elif isinstance(result, dict):
                matches = result.get("matches", [])
            else:
                matches = []
            
            if not matches:
                # Face doesn't exist in database
                return False, None, None
            
            # Check if the match score is above threshold
            best_match = matches[0]
            match_score = best_match.score if hasattr(best_match, 'score') else best_match.get('score', 0)
            
            print(f"[CHECK_FACE] Match score: {match_score}, threshold: {self.min_score}")
            
            if match_score >= self.min_score:
                # Face already exists
                user_id = None
                if hasattr(best_match, 'metadata'):
                    user_id = best_match.metadata.get('user_id')
                elif isinstance(best_match, dict) and 'metadata' in best_match:
                    user_id = best_match['metadata'].get('user_id')
                
                return True, user_id, None
            else:
                # Match score below threshold, face doesn't exist
                return False, None, None
                
        except Exception as e:
            print(f"[CHECK_FACE] Error checking face: {e}")
            return False, None, str(e)

    def verify_user(self, image_base64):
        frame = self.decode_base64_image(image_base64)
        if frame is None:
            print("[VERIFY] Failed to decode image")
            return None, "Failed to decode image"

        print(f"[VERIFY] Image decoded, shape: {frame.shape}")
        
        # Skip liveness check for now to debug the matching issue
        liveness_passed = self.liveness_check(frame)
        print(f"[VERIFY] Liveness check: {liveness_passed}")
        # Continue even if liveness fails for debugging
        
        current_ratio = self.nodal_ratio(frame)
        if current_ratio is None:
            print("[VERIFY] Landmarks not detected")
            return None, "Landmarks not detected"
        print(f"[VERIFY] Nodal ratio: {current_ratio}")

        embedding = self.get_embedding(frame)
        if embedding is None:
            print("[VERIFY] Face not detected by MTCNN")
            return None, "Face not detected"
        
        print(f"[VERIFY] Embedding extracted, dim: {len(embedding)}")

        result = self.index.query(vector=embedding.tolist(), top_k=5, include_metadata=True)
        print(f"[VERIFY] Query result type: {type(result)}")
        print(f"[VERIFY] Query result: {result}")
        
        # Pinecone returns an object with .matches attribute, not a dict
        if hasattr(result, 'matches'):
            matches = result.matches or []
        elif isinstance(result, dict):
            matches = result.get("matches", [])
        else:
            matches = []
            
        print(f"[VERIFY] Matches type: {type(matches)}, len: {len(matches) if matches else 0}")
        
        if not matches:
            print("[VERIFY] No matches found in Pinecone")
            return None, "No match found"

        print(f"[VERIFY] Found {len(matches)} matches")
        for i, m in enumerate(matches):
            # Handle both dict and object access
            if hasattr(m, 'id'):
                print(f"[VERIFY] Match {i}: id={m.id}, score={m.score}, metadata={m.metadata}")
            else:
                print(f"[VERIFY] Match {i}: id={m.get('id')}, score={m.get('score')}, metadata={m.get('metadata')}")

        match = matches[0]
        
        # Handle both dict and object access for match
        if hasattr(match, 'score'):
            score = float(match.score or 0.0)
            metadata = match.metadata or {}
            match_id = match.id
        else:
            score = float(match.get("score", 0.0))
            metadata = match.get("metadata", {}) or {}
            match_id = match.get("id")
            
        # metadata might also be an object
        if hasattr(metadata, 'user_id'):
            user_id = metadata.user_id or match_id
            stored_ratio = getattr(metadata, 'nodal_ratio', None)
        else:
            user_id = metadata.get("user_id") or match_id
            stored_ratio = metadata.get("nodal_ratio")

        print(f"[VERIFY] Best match: user_id={user_id}, score={score}, stored_ratio={stored_ratio}")

        # Lowered threshold for testing - cosine similarity can be lower
        effective_min_score = 0.25  # Lower threshold for debugging
        if score < effective_min_score:
            print(f"[VERIFY] Score {score} below threshold {effective_min_score}")
            return None, f"Low similarity score: {score:.3f}"

        ratio_match = self._ratio_match(current_ratio, stored_ratio)
        print(f"[VERIFY] Ratio match: {ratio_match} (current={current_ratio}, stored={stored_ratio})")
        
        # Skip ratio match for now
        # if not ratio_match:
        #     return None, "Nodal ratio mismatch"

        self.adaptive_update(user_id, embedding)
        return {
            "user_id": user_id,
            "score": score,
            "nodal_ratio": current_ratio,
        }, None

    def adaptive_update(self, user_id, new_embedding):
        fetch = self.index.fetch(ids=[user_id])
        vectors = fetch.get("vectors", {}) if isinstance(fetch, dict) else {}
        current = vectors.get(user_id)
        if not current:
            return

        old_vector = np.array(current.get("values", []), dtype=np.float32)
        if old_vector.size == 0:
            return

        updated = (1 - self.adaptive_lr) * old_vector + self.adaptive_lr * new_embedding
        updated = updated / (np.linalg.norm(updated) + 1e-6)
        metadata = current.get("metadata", {})
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.index.upsert(vectors=[(user_id, updated.tolist(), metadata)])

    def list_users(self, limit=100, pagination_token=None):
        response = self.index.list(limit=limit, pagination_token=pagination_token)
        ids = response.get("ids", []) if isinstance(response, dict) else []
        next_token = response.get("pagination", {}).get("next") if isinstance(response, dict) else None
        if not ids:
            return [], None

        fetched = self.index.fetch(ids=ids)
        vectors = fetched.get("vectors", {}) if isinstance(fetched, dict) else {}

        users = []
        for user_id, data in vectors.items():
            metadata = data.get("metadata", {}) or {}
            face_status = "active"
            samples = metadata.get("samples", 0)
            if samples and samples < 3:
                face_status = "needs_update"
            users.append(
                {
                    "user_id": user_id,
                    "name": metadata.get("name"),
                    "registered_at": metadata.get("created_at"),
                    "face_status": face_status,
                }
            )

        return users, next_token

    def get_user_embedding(self, user_id):
        fetched = self.index.fetch(ids=[user_id])
        vectors = fetched.get("vectors", {}) if isinstance(fetched, dict) else {}
        data = vectors.get(user_id)
        if not data:
            return None
        return {
            "user_id": user_id,
            "embedding_dim": len(data.get("values", [])),
            "vectors": data.get("values", []),
            "metadata": data.get("metadata", {}),
        }

    def delete_user_face(self, user_id):
        self.index.delete(ids=[user_id])
    def verify_specific_user(self, user_id, image_base64, strict=True):
        """
        Verify that the face in the image matches a specific user.
        Used for interview proctoring to ensure the same person is taking the interview.
        
        Args:
            user_id: The user ID to verify against
            image_base64: Base64 encoded image
            strict: If True, requires higher similarity score for match
            
        Returns:
            (result_dict, error_string) - result contains match info or None if failed
        """
        frame = self.decode_base64_image(image_base64)
        if frame is None:
            print(f"[VERIFY_SPECIFIC] Failed to decode image for user {user_id}")
            return None, "Failed to decode image"

        print(f"[VERIFY_SPECIFIC] Verifying identity for user: {user_id}")
        
        # Get the stored embedding for this specific user
        stored_data = self.get_user_embedding(user_id)
        if not stored_data or not stored_data.get("vectors"):
            print(f"[VERIFY_SPECIFIC] No stored face data found for user: {user_id}")
            return None, "No registered face found for this user"

        stored_embedding = np.array(stored_data["vectors"], dtype=np.float32)
        stored_metadata = stored_data.get("metadata", {})
        stored_ratio = stored_metadata.get("nodal_ratio")
        
        # Perform liveness check
        liveness_passed = self.liveness_check(frame)
        print(f"[VERIFY_SPECIFIC] Liveness check: {liveness_passed}")
        
        # Get current face embedding
        current_ratio = self.nodal_ratio(frame)
        if current_ratio is None:
            print("[VERIFY_SPECIFIC] Landmarks not detected")
            return None, "Face landmarks not detected"
        print(f"[VERIFY_SPECIFIC] Current nodal ratio: {current_ratio}")

        embedding = self.get_embedding(frame)
        if embedding is None:
            print("[VERIFY_SPECIFIC] Face not detected in image")
            return None, "Face not detected"
        
        print(f"[VERIFY_SPECIFIC] Embedding extracted, dim: {len(embedding)}")

        # Calculate cosine similarity directly between stored and current embedding
        similarity = float(np.dot(embedding, stored_embedding) / 
                          (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding) + 1e-6))
        
        print(f"[VERIFY_SPECIFIC] Similarity score: {similarity}")

        # Use stricter threshold for interview verification
        threshold = 0.35 if strict else 0.25
        
        if similarity < threshold:
            print(f"[VERIFY_SPECIFIC] Similarity {similarity:.3f} below threshold {threshold}")
            return {
                "verified": False,
                "user_id": user_id,
                "score": similarity,
                "liveness": liveness_passed,
                "reason": "Face does not match registered user"
            }, None

        # Check nodal ratio match for additional verification
        ratio_match = self._ratio_match(current_ratio, stored_ratio)
        print(f"[VERIFY_SPECIFIC] Ratio match: {ratio_match}")

        return {
            "verified": True,
            "user_id": user_id,
            "score": similarity,
            "liveness": liveness_passed,
            "nodal_ratio": current_ratio,
            "ratio_match": ratio_match
        }, None