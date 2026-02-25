# Requirements Document

## Introduction

The Age-Invariant Face Recognition System is a web application that compares two human face images and determines whether they belong to the same person, even when the photos were taken up to 40 years apart. The system detects faces, estimates ages, enforces age-based comparison rules, generates face embeddings, and computes similarity scores. It exposes a REST API backend (Python FastAPI) with an AI provider abstraction layer, and a React frontend for image upload and result display.

## Glossary

- **System**: The Age-Invariant Face Recognition web application (backend + frontend)
- **Backend**: The Python FastAPI server that processes images and returns comparison results
- **Frontend**: The React + Vite single-page application for uploading images and displaying results
- **Face_Detector**: The component responsible for detecting and cropping faces from uploaded images, using InsightFace RetinaFace as primary and MTCNN as fallback
- **Age_Estimator**: The component that estimates the age and age group of a detected face
- **Embedding_Generator**: The component that produces 512-dimensional ArcFace embeddings from cropped face images
- **Similarity_Calculator**: The component that computes cosine similarity between two face embeddings
- **Age_Rule_Engine**: The component that evaluates whether two age groups are eligible for comparison
- **AI_Provider**: An abstraction interface for face analysis operations, allowing swappable implementations (local InsightFace, OpenAI)
- **Age_Group**: A categorical classification of estimated age: infant (0–4), child (5–12), teen (13–19), adult (20–49), senior (50+)
- **Embedding**: A 512-dimensional L2-normalized vector representing a face's identity features
- **Similarity_Score**: A floating-point value between -1 and 1 produced by cosine similarity of two embeddings
- **Comparison_Result**: The final JSON response containing ages, age groups, similarity score, confidence, result label, and message

## Requirements

### Requirement 1: Image Upload and Validation

**User Story:** As a user, I want to upload two face images for comparison, so that the system can determine whether they depict the same person.

#### Acceptance Criteria

1. WHEN a user submits two images via the `/compare-faces` endpoint, THE Backend SHALL accept multipart/form-data containing fields `image1` and `image2`
2. WHEN an uploaded file has a format other than jpg, jpeg, png, or webp, THE Backend SHALL reject the request with a descriptive error message indicating the unsupported format
3. WHEN an uploaded file exceeds 10 MB in size, THE Backend SHALL reject the request with a descriptive error message indicating the size limit
4. WHEN a valid image is uploaded, THE Backend SHALL decode the image into an in-memory representation without persisting the file to disk
5. WHEN image processing is complete, THE Backend SHALL release all in-memory image data so that no user image data is retained

### Requirement 2: Face Detection

**User Story:** As a user, I want the system to detect exactly one face in each uploaded image, so that the comparison is performed on valid single-face inputs.

#### Acceptance Criteria

1. WHEN a valid image is provided, THE Face_Detector SHALL attempt detection using InsightFace RetinaFace as the primary detector
2. IF the primary detector fails to detect a face, THEN THE Face_Detector SHALL fall back to MTCNN for detection
3. WHEN no face is detected in an image, THE Face_Detector SHALL return an error message stating "No face detected in the image"
4. WHEN multiple faces are detected in an image, THE Face_Detector SHALL return an error message stating "Multiple faces detected; please upload an image with exactly one face"
5. WHEN exactly one face is detected, THE Face_Detector SHALL crop and return the face region for downstream processing

### Requirement 3: Age Estimation

**User Story:** As a user, I want the system to estimate the age of each face, so that age-based comparison rules can be applied and age information is included in the result.

#### Acceptance Criteria

1. WHEN a cropped face image is provided, THE Age_Estimator SHALL return a numeric estimated age
2. WHEN a numeric age is determined, THE Age_Estimator SHALL classify the age into exactly one age group: infant (0–4), child (5–12), teen (13–19), adult (20–49), or senior (50+)
3. WHEN the estimated age is a boundary value (0, 5, 13, 20, or 50), THE Age_Estimator SHALL assign the age group whose range starts at that boundary value

### Requirement 4: Age Comparison Rules

**User Story:** As a user, I want the system to reject unreliable comparisons between infant images and older-age images, so that I receive only meaningful results.

#### Acceptance Criteria

1. WHEN both faces are classified as infant, THE Age_Rule_Engine SHALL allow the comparison to proceed
2. WHEN one face is classified as infant and the other is classified as teen, adult, or senior, THE Age_Rule_Engine SHALL reject the comparison with the message "Cannot reliably compare infant/childhood images with adult images"
3. WHEN one face is classified as infant and the other is classified as child, THE Age_Rule_Engine SHALL allow the comparison to proceed
4. WHEN neither face is classified as infant, THE Age_Rule_Engine SHALL allow the comparison to proceed
5. WHEN the Age_Rule_Engine rejects a comparison, THE Backend SHALL return the rejection message along with the estimated ages and age groups of both faces

### Requirement 5: Face Embedding Generation

**User Story:** As a user, I want the system to generate identity embeddings from each face, so that a numerical similarity comparison can be performed.

#### Acceptance Criteria

1. WHEN a cropped face image is provided, THE Embedding_Generator SHALL produce a 512-dimensional embedding vector using ArcFace via InsightFace
2. THE Embedding_Generator SHALL normalize each embedding vector using L2 normalization so that the vector has unit length
3. WHEN L2 normalization is applied to an embedding, THE Embedding_Generator SHALL produce a vector whose L2 norm equals 1.0 within a tolerance of 1e-6

### Requirement 6: Similarity Calculation

**User Story:** As a user, I want the system to compute a similarity score between two face embeddings, so that I can understand how likely the two faces belong to the same person.

#### Acceptance Criteria

1. WHEN two L2-normalized embedding vectors are provided, THE Similarity_Calculator SHALL compute cosine similarity and return a score between -1.0 and 1.0
2. WHEN the similarity score is greater than or equal to 0.35, THE Similarity_Calculator SHALL classify the result as "same_person"
3. WHEN the similarity score is less than 0.35, THE Similarity_Calculator SHALL classify the result as "different_person"

### Requirement 7: Result Response

**User Story:** As a user, I want to receive a structured JSON response with all comparison details, so that I can understand the system's determination and the supporting data.

#### Acceptance Criteria

1. WHEN a successful comparison is completed, THE Backend SHALL return a JSON response containing the fields: age1, age2, age_group1, age_group2, similarity_score, confidence, result, and message
2. WHEN an error occurs during processing, THE Backend SHALL return a JSON error response containing an error field with a descriptive message
3. WHEN the comparison is rejected by the Age_Rule_Engine, THE Backend SHALL return a JSON response containing age1, age2, age_group1, age_group2, and the rejection message

### Requirement 8: AI Provider Abstraction

**User Story:** As a developer, I want the face analysis operations to be abstracted behind a provider interface, so that I can swap between local InsightFace processing and cloud-based OpenAI processing via configuration.

#### Acceptance Criteria

1. THE AI_Provider SHALL define a common interface with methods for face detection, age estimation, and embedding generation
2. WHEN the environment variable `USE_OPENAI` is set to `false` or is absent, THE System SHALL use the LocalInsightFaceProvider implementation
3. WHEN the environment variable `USE_OPENAI` is set to `true`, THE System SHALL use the OpenAIProvider implementation
4. WHEN the OpenAIProvider is selected but `OPENAI_API_KEY` is not set, THE System SHALL fall back to the LocalInsightFaceProvider and log a warning

### Requirement 9: Health Check Endpoint

**User Story:** As a developer or operator, I want a health check endpoint, so that I can verify the service is running and models are loaded.

#### Acceptance Criteria

1. WHEN a GET request is made to `/health`, THE Backend SHALL return a JSON response with status "ok" and a boolean field indicating whether the AI models are loaded
2. WHEN AI models fail to load at startup, THE Backend SHALL return a health response with status "ok" and model_loaded set to false

### Requirement 10: Frontend User Interface

**User Story:** As a user, I want a web interface to upload two images and view comparison results, so that I can use the system without making direct API calls.

#### Acceptance Criteria

1. THE Frontend SHALL display two image upload areas, one for each face image
2. THE Frontend SHALL display a "Compare" button that is enabled only when both images have been selected
3. WHEN the user clicks "Compare", THE Frontend SHALL send both images to the `/compare-faces` endpoint and display a loading indicator
4. WHEN a successful response is received, THE Frontend SHALL display the detected age, age group, similarity score, confidence, and final result for each face
5. WHEN an error response is received, THE Frontend SHALL display the error message to the user
6. THE Frontend SHALL validate file type and size on the client side before submission and display a validation message for invalid files

### Requirement 11: Containerized Deployment

**User Story:** As a developer, I want Docker configuration for the application, so that I can easily build and run the system in any environment.

#### Acceptance Criteria

1. THE System SHALL include a Dockerfile that builds the backend with all Python dependencies and model files
2. THE System SHALL include a docker-compose.yml that orchestrates the backend and frontend services
3. WHEN the Docker container starts, THE Backend SHALL download or load required AI model files automatically
4. WHERE GPU runtime is available, THE System SHALL support an optional GPU-enabled Docker configuration
