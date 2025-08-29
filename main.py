from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import random
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI(title="Emotion Detection Microservice")

# ----------------- CORS -----------------
# Put your actual frontend origins here. Add localhost & your GitHub Pages.
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://shashwat0202.github.io",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https://([a-z0-9-]+)\.github\.io$",  # allow any *.github.io
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Model & assets (lazy load) -----------------
MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", "emotion_model.h5")
_model = None  # lazy
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        _model = load_model(MODEL_PATH, compile=False)
    return _model

def preprocess_image(image: np.ndarray):
    if image is None:
        return None, "Invalid image data."
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None, "Could not convert image to grayscale."

    faces = _face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return None, "No face detected. Ensure your face is clearly visible."

    # Take first detected face
    x, y, w, h = faces[0]
    face = gray_image[y:y+h, x:x+w]

    try:
        resized_face = cv2.resize(face, (48, 48))
    except Exception:
        return None, "Failed to resize face."

    normalized_face = resized_face / 255.0
    # Keras expects (1, 48, 48, 1)
    face_arr = img_to_array(normalized_face)
    face_arr = np.expand_dims(face_arr, axis=-1)  # add channel dimension
    face_arr = np.expand_dims(face_arr, axis=0)   # add batch dimension
    return face_arr, None

def predict_emotion(image: np.ndarray):
    preprocessed_image, error = preprocess_image(image)
    if error:
        return None, error
    model = get_model()
    preds = model.predict(preprocessed_image)
    emotion_index = int(np.argmax(preds))
    return emotion_labels[emotion_index], None

recommendations = {
    "Happy": [
        "Keep smiling!", "Celebrate your happiness!", "Spread the joy around!",
        "Enjoy the moment!", "Cherish the little things.", "Share your happiness with a loved one.",
        "Capture the moment in a photo or journal.", "Plan something exciting to continue the positivity.",
        "Do something kind for someone else.", "Dance to your favorite song.", "Call someone and share your joy.",
        "Write a gratitude list to reflect on your blessings."
    ],
    "Sad": [
        "Take a walk outside.", "Talk to a trusted friend.", "Listen to your favorite music.",
        "Write your feelings down.", "Watch an uplifting movie or show.", "Focus on things you're grateful for.",
        "Engage in a hobby you enjoy.", "Give yourself permission to rest and heal.",
        "Practice deep breathing or meditation.", "Treat yourself to something small you love.",
        "Consider seeking support from a therapist.", "Take it one day at a time—small steps count."
    ],
    "Angry": [
        "Take a deep breath and count to 10.", "Step outside for fresh air.", "Practice mindfulness or meditation.",
        "Channel your energy into a creative activity.", "Engage in physical exercise to release tension.",
        "Write down your feelings to process them.", "Take a break and revisit the issue later.",
        "Listen to calming music or sounds.", "Focus on something that makes you laugh.",
        "Take a warm shower or bath to relax.", "Repeat a calming mantra or affirmation.",
        "Visualize a peaceful place or memory."
    ],
    "Neutral": [
        "Maintain your balance and focus.", "Enjoy your steady mood.", "Reflect on your day and plan ahead.",
        "Take time for a little self-care.", "Explore a new activity or hobby.", "Connect with someone and share your thoughts.",
        "Appreciate the calm and find joy in small things.", "Read a book or listen to a podcast.",
        "Organize your space for a sense of clarity.", "Set small goals to improve your routine.",
        "Spend time in nature to refresh your mind.", "Write down things you look forward to."
    ],
    "Surprise": [
        "Embrace the unexpected moment!", "Share the surprise with others.", "Keep an open mind for what's next!",
        "Take a moment to process your emotions.", "Write about the surprise to remember it.",
        "Turn the surprise into an opportunity.", "Enjoy the excitement and be present.",
        "Think about how this surprise fits into the bigger picture.", "Celebrate the novelty of the experience.",
        "Let yourself smile and laugh at the unexpected.", "Capture the moment if it's a good surprise.",
        "Prepare for surprises in the future by staying adaptable."
    ],
    "Fear": [
        "Ground yourself by breathing deeply.", "Talk to someone you trust.", "Focus on what you can control.",
        "Remind yourself that you're safe.", "Visualize a positive outcome.",
        "Engage in a soothing activity like drawing or knitting.", "Repeat calming affirmations to yourself.",
        "Take small steps to face and overcome your fear.", "Avoid overthinking by focusing on the present moment.",
        "Seek support from a professional if needed.", "Spend time in a comforting environment.",
        "Remind yourself that fear often fades with time."
    ],
    "Disgust": [
        "Refocus on things you enjoy.", "Take a moment to clear your mind.", "Engage in an activity you love.",
        "Take a break and refresh yourself.", "Remind yourself that this feeling will pass.",
        "Practice gratitude to shift your perspective.", "Surround yourself with uplifting environments or people.",
        "Distract yourself with a favorite book, show, or song.", "Take a deep breath and let the emotion subside.",
        "Talk to someone about your feelings.", "Shift your attention to something neutral or positive.",
        "Do something relaxing, like sipping tea or meditating."
    ]
}

@app.get("/health")
def health():
    # light endpoint to check uptime without loading the model
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_mood(image: UploadFile = File(...)):
    try:
        npimg = np.frombuffer(await image.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        mood, error = predict_emotion(img)
        if error:
            return JSONResponse(content={"mood": "Error", "recommendations": [error]}, status_code=200)

        recs = recommendations.get(mood, ["Stay positive and keep going!"])
        # avoid ValueError if list shorter than k
        k = 2 if len(recs) >= 2 else 1
        response = {"mood": mood, "recommendations": random.sample(recs, k=k)}
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        # If anything fails (including model load), return JSON with detail.
        raise HTTPException(status_code=500, detail=str(e))
