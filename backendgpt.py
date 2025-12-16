import os
import time
import json
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # Raise error if key is missing
    raise ValueError("❌ Missing API Key! Set 'GEMINI_API_KEY' in your environment variables.")

# Initialize the new client
client = genai.Client(api_key=GEMINI_API_KEY)


MODEL_ID = "gemini-2.5-flash-lite"


def generate_subtopics(user_question, retries=1, delay=3):
    system_prompt = f"""
    You are a learning design assistant.
    
    Given a student's curiosity-based question, your job is NOT to answer it directly. Instead, analyze it and return the following in JSON format:
    
    - subject_area: Which academic subject(s) this question touches (e.g., Science, Math, History, etc.)
    - depth_level: Introductory / Intermediate / Advanced
    - question_type: Factual / Conceptual / Procedural / Opinion / Open-Ended
    - curiosity_tree: A list of 3–5 short, focused subtopics that help explore this question further
    
    Return your result as a raw JSON object **without any code block formatting**, like this:
    {{
      "subject_area": "...",
      "depth_level": "...",
      "question_type": "...",
      "curiosity_tree": [
        "...",
        "...",
        "...",
        "..."
      ]
    }}
    
    Here is the question to analyze:
    "{user_question}"
    """
    
    # New Config Pattern
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        temperature=0.7,
        max_output_tokens=500
    )
    
    for attempt in range(1, retries + 1):
        try:
            # New Generation Call
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=user_question,
                config=config
            )
            
            # Helper to handle potential parsing issues
            if response.text:
                return json.loads(response.text)
            else:
                print(f"⚠️ Empty response on attempt {attempt}")

        except errors.APIError as e:
            # Handle standard API errors (like 429 Rate Limit)
            print(f"❌ API Error on attempt {attempt}: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error on attempt {attempt}: {e}")
        
        # Exponential backoff
        if attempt < retries:
            wait_time = delay * (2 ** (attempt - 1))
            print(f"🔁 Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print("❌ Max retries reached. Could not generate subtopics.")
    return None


def generate_explanation_and_activity(subtopic, user_question, retries=1, delay=3):
    system_prompt = f"""
    You are an educational tutor and interaction designer that provides clear, engaging explanations for specific topics related to a student's curiosity.

    The student's broader curiosity question is:
    "{user_question}"

    One of the subtopics to help explain this question is:
    "{subtopic}"

    Your task:

    1. Write a explanation about the **subtopic only**. The explanation should:
    - Have a maximum of 500 words and a minimum of 100 words
    - Be written for an undergraduate-level audience
    - Use analogies or simple examples if helpful
    - Avoid heavy technical jargon
    - Do **not** attempt to answer the full original curiosity question

    2. Based on your explanation, choose any random template from the following list:

    Available templates:
    - drag_drop: Drag items into the relevant categories
    - match_pairs: Match terms to their correct definitions
    - fill_blanks: Fill in missing parts of a formula or sentence
    - toggle_true_false: Quickfire true/false quiz

    3. Return your result as a raw JSON object **without any code block formatting**, like this:

    {{
      "Topic": "...",
      "Explanation": "...",
      "Interactive Template": "..."
    }}

    Now, perform the task and return only the JSON object.
    """

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        temperature=0.7,
        max_output_tokens=1000
    )

    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=subtopic,
                config=config
            )
            return json.loads(response.text)
            
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
        
        if attempt < retries:
            wait_time = delay * (2 ** (attempt - 1))
            time.sleep(wait_time)
            
    print("❌ Max retries reached. Could not get explanation.")
    return None


def generate_interactive_activity(topic, explanation, template_type='drag_drop', retries=1, delay=3):
    system_prompt = f"""
    You are an educational interaction designer.
    
    Given a topic, an explanation, and a selected template type, your job is to generate an interactive activity in the following JSON format:
    For each activity, include around 5-7 questions.
    If the template is `drag_drop`, structure it like this:
    
    {{
      "id": "unique-id-for-the-activity",
      "type": "drag_drop",
      "title": "Title of the Game",
      "description": "One-line description of the drag-and-drop activity.",
      "draggableElements": [
        {{ "id": "id1", "label": "Draggable Term 1" }},
        {{ "id": "id2", "label": "Draggable Term 2" }}
      ],
      "droppableBlanks": [
        {{
          "id": "drop-1",
          "label": "Hint or definition where a term should go",
          "correctElementId": "id1"
        }}
      ]
    }}

    Any label should be max 5 words.
    
    If the selected template is `match_pairs`, respond with a JSON object in this format:

    {{
      "id": "photosynthesis-basics",
      "type": "match",
      "title": "Key Concepts in Photosynthesis",
      "description": "Match each photosynthesis-related term with its correct definition.",
      "pairs": [
        {{ "prompt": "Chlorophyll", "match": "Green pigment that captures light energy" }},
        {{ "prompt": "Stomata", "match": "Tiny pores on leaves where gas exchange occurs" }}
      ]
    }}
    
    Instructions:
    - Use `prompt` for terms, processes, or concepts.
    - Use `match` for definitions, functions, or descriptions (max 5 words).
    - Do NOT include:
      - Synonyms (e.g., avoid “plant sugar” → “glucose” if “glucose” is already a prompt)
      - Nicknames (e.g., no “sun sugar” → “glucose”)
      - Repeated ideas
    - Keep explanations short, factual, and directly linked to the topic.
    - Each pair should be unique and unambiguous.
    
    If the template is `fill_blanks`, structure it like this:
    
    {{
      "id": "unique-id",
      "type": "fill_in_blanks",
      "title": "Fill in the Blanks",
      "description": "Fill in the blanks using the correct terms.",
      "text": "... with ___ and ___",
      "blanks": {{
        "1": ["Melanin", "Keratin", "Chlorophyll"],
        "2": ["Melanocytes", "Blood cells", "Nerve cells"]
      }},
      "answers": {{
        "1": "Melanin",
        "2": "Melanocytes"
      }}
    }}
    
    If the template is `toggle_true_false`, structure it like this:
    
    {{
      "id": "unique-id",
      "type": "toggle_true_false",
      "title": "True or False",
      "description": "Decide if the following statements are true or false.",
      "statements": [
        {{ "id": "s1", "text": "Melanin protects the skin from UV radiation.", "correctAnswer": true }}
      ]
    }}
    
    Now generate a JSON object using the format for this template:
    Topic: {topic}
    Explanation: {explanation}
    Interactive Template: {template_type}
    
    Respond ONLY with the JSON object.
    """

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        temperature=0.7
    )

    for attempt in range(1, retries + 1):
        try:
            # Pass template_type as user content
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=template_type,
                config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
        
        if attempt < retries:
            wait_time = delay * (2 ** (attempt - 1))
            time.sleep(wait_time)
            
    print("❌ Max retries reached. Could not generate activity.")
    return None
