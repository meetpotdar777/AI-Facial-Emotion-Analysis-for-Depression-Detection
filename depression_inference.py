import random
import time
import numpy as np # Import numpy for numerical operations, specifically np.std()

# --- Configuration ---
# Define possible simulated emotions and engagement levels
SIMULATED_EMOTIONS = ['Happy', 'Sad', 'Neutral', 'Boring', 'Unresponsive']
SIMULATED_ENGAGEMENT_LEVELS = ['High', 'Medium', 'Low']

# Define thresholds for depression inference (these are purely illustrative)
# In a real system, these would be derived from clinical data and machine learning
THRESHOLD_SAD_RATIO_SEVERE = 0.6  # % of time spent sad/unresponsive
THRESHOLD_SAD_RATIO_MODERATE = 0.4
THRESHOLD_SAD_RATIO_MILD = 0.2

THRESHOLD_LOW_ENGAGEMENT_RATIO_SEVERE = 0.5 # % of time in low engagement
THRESHOLD_LOW_ENGAGEMENT_RATIO_MODERATE = 0.3

THRESHOLD_HAPPY_ABSENCE_SEVERE = 0.7 # % of time without happy emotions
THRESHOLD_HAPPY_ABSENCE_MODERATE = 0.5

THRESHOLD_MOOD_VARIABILITY_FLAT = 0.1 # Low variability might indicate flat affect

# --- Data Simulation Function ---
def simulate_facial_data_stream(duration_seconds=60, interval_seconds=5):
    """
    Simulates a stream of facial emotion and engagement data over time.
    In a real system, this data would come from a live facial analysis pipeline.
    """
    print(f"Simulating facial data stream for {duration_seconds} seconds...")
    data_points = []
    start_time = time.time()
    current_time_elapsed = 0

    while current_time_elapsed < duration_seconds:
        # Simulate an emotion and engagement level
        emotion = random.choice(SIMULATED_EMOTIONS)
        engagement = random.choice(SIMULATED_ENGAGEMENT_LEVELS)

        data_points.append({
            "time_elapsed": current_time_elapsed,
            "emotion": emotion,
            "engagement": engagement
        })

        # print(f"Time: {current_time_elapsed:2d}s | Emotion: {emotion:12s} | Engagement: {engagement}")
        time.sleep(interval_seconds) # Simulate time passing
        current_time_elapsed += interval_seconds

    print("Data stream simulation complete.")
    return data_points

# --- Inference Metric Calculation Functions ---

def calculate_sadness_duration_ratio(data_stream):
    """Calculates the ratio of time spent in 'Sad' or 'Unresponsive' states."""
    sad_unresponsive_count = sum(1 for dp in data_stream if dp['emotion'] in ['Sad', 'Unresponsive'])
    return sad_unresponsive_count / len(data_stream) if data_stream else 0

def calculate_low_engagement_ratio(data_stream):
    """Calculates the ratio of time spent in 'Low' engagement states."""
    low_engagement_count = sum(1 for dp in data_stream if dp['engagement'] == 'Low')
    return low_engagement_count / len(data_stream) if data_stream else 0

def calculate_happy_absence_ratio(data_stream):
    """Calculates the ratio of time when 'Happy' emotion is NOT present."""
    non_happy_count = sum(1 for dp in data_stream if dp['emotion'] != 'Happy')
    return non_happy_count / len(data_stream) if data_stream else 0

def calculate_mood_variability(data_stream):
    """
    Calculates a conceptual 'mood variability'.
    Lower values could indicate 'flat affect'. This is a very simplified measure.
    A more robust measure would use numerical emotion scores.
    """
    if not data_stream or len(data_stream) < 2:
        return 0.0 # Cannot calculate variability with less than 2 points

    # Map emotions to arbitrary numerical values for simple variability calculation
    emotion_to_num = {'Happy': 2, 'Neutral': 1, 'Boring': 0, 'Sad': -1, 'Unresponsive': -2}
    
    numerical_emotions = [emotion_to_num.get(dp['emotion'], 0) for dp in data_stream]
    
    # Calculate standard deviation as a proxy for variability
    return np.std(numerical_emotions)

# --- Advanced Depression Inference Logic ---

def infer_depression_level(data_stream):
    """
    Infers a conceptual depression level based on calculated metrics.
    This is a rule-based system for demonstration.
    """
    if not data_stream:
        return "No Data Available"

    sad_ratio = calculate_sadness_duration_ratio(data_stream)
    low_engagement_ratio = calculate_low_engagement_ratio(data_stream)
    happy_absence_ratio = calculate_happy_absence_ratio(data_stream)
    mood_variability = calculate_mood_variability(data_stream)

    print("\n--- Calculated Metrics ---")
    print(f"Sad/Unresponsive Ratio: {sad_ratio:.2f}")
    print(f"Low Engagement Ratio:   {low_engagement_ratio:.2f}")
    print(f"Happy Absence Ratio:    {happy_absence_ratio:.2f}")
    print(f"Mood Variability (StdDev): {mood_variability:.2f}")
    print("-" * 30)

    # Apply inference rules
    # More severe conditions
    if sad_ratio >= THRESHOLD_SAD_RATIO_SEVERE or \
       low_engagement_ratio >= THRESHOLD_LOW_ENGAGEMENT_RATIO_SEVERE or \
       happy_absence_ratio >= THRESHOLD_HAPPY_ABSENCE_SEVERE:
        return "Severe Depression Indication (Simulated)"
    
    # Moderate conditions
    if sad_ratio >= THRESHOLD_SAD_RATIO_MODERATE or \
       low_engagement_ratio >= THRESHOLD_LOW_ENGAGEMENT_RATIO_MODERATE or \
       happy_absence_ratio >= THRESHOLD_HAPPY_ABSENCE_MODERATE:
        return "Moderate Depression Indication (Simulated)"
    
    # Mild conditions (also consider low mood variability for flat affect)
    if sad_ratio >= THRESHOLD_SAD_RATIO_MILD or \
       happy_absence_ratio >= THRESHOLD_HAPPY_ABSENCE_MODERATE or \
       mood_variability < THRESHOLD_MOOD_VARIABILITY_FLAT: # Very low variability
        return "Mild Depression Indication (Simulated)"
    
    return "No Significant Depression Indication (Simulated)"

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Conceptual Depression Inference Logic System ---")
    print("This system *simulates* input data to demonstrate inference logic.")
    print("It is NOT a medical diagnostic tool.")
    print("Running a simulated 1-minute session...\n")

    # Simulate a stream of data (e.g., 1 minute of data, collected every 5 seconds)
    simulated_data = simulate_facial_data_stream(duration_seconds=60, interval_seconds=5)

    # Perform the depression inference
    depression_level = infer_depression_level(simulated_data)

    print(f"\n--- Inferred Depression Level ---")
    print(f"Result: {depression_level}")
    print("-" * 30)
    print("\nRemember: This is a conceptual demonstration. For mental health concerns, consult a professional.")
    # python depression_inference.py