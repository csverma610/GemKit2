from google import genai
from google.genai import types
from PIL import Image
import json
import os
import re
from typing import Dict, Any


class DiagramAnalyzer:
    """
    Analyzes geometric diagrams from images and generates a JSON specification
    that can be used to recreate them.

    This class uses the Gemini API to interpret the contents of an image and
    produces a structured JSON output that describes the shapes, lines, and
    other elements in the diagram.
    """
    def __init__(self, api_key: str):
        """
        Initializes the DiagramAnalyzer.

        Args:
            api_key (str): The Google Gemini API key.
        """
        # Step 1: Create the client (NO genai.configure needed!)
        self.client = genai.Client(api_key=api_key)
        
        # Step 2: Define model configuration
        self.model_name = 'gemini-2.0-flash-exp'
        
        # Step 3: Configure generation parameters
        self.generation_config = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )
        
        # Step 4: Configure safety settings
        self.safety_settings = [
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE'
            ),
        ]
        
        print("✓ Google GenAI Client initialized successfully")
        print(f"  Model: {self.model_name}")
        print(f"  Temperature: {self.generation_config.temperature}")
        print(f"  Max tokens: {self.generation_config.max_output_tokens}")
        
    def analyze_diagram(self, image_path: str) -> Dict[str, Any]:
        """
        Analyzes a geometric diagram from an image file.

        This method sends the image to the Gemini API with a detailed prompt
        requesting a JSON specification of the diagram's contents.

        Args:
            image_path (str): The path to the input image file.

        Returns:
            Dict[str, Any]: A dictionary representing the diagram specification.
        """
        print(f"\nLoading image: {image_path}")
        
        # Load the image
        img = Image.open(image_path)
        print(f"✓ Image loaded: {img.size[0]}x{img.size[1]} pixels")
        
        # Create detailed prompt for Gemini
        prompt = """
        Analyze this geometric diagram carefully and provide a detailed JSON specification to recreate it.
        
        Your response must be ONLY a valid JSON object with this exact structure:
        {
            "canvas": {
                "width": <width in pixels>,
                "height": <height in pixels>,
                "background_color": "<color as RGB tuple or name>"
            },
            "shapes": [
                {
                    "type": "line|circle|ellipse|arc|sector|chord|half_circle|rectangle|triangle|polygon|point|curve|spline|bezier",
                    "coordinates": <appropriate coordinates for the shape>,
                    "color": "<RGB tuple or color name>",
                    "width": <line width>,
                    "fill": "<fill color if applicable or null>",
                    "label": "<label text if any or null>",
                    "label_position": [x, y],
                    "start_angle": <for arcs/sectors/chords, in degrees>,
                    "end_angle": <for arcs/sectors/chords, in degrees>,
                    "control_points": <for bezier curves, list of control points>
                }
            ]
        }
        
        Details for each shape type:
        - line: "coordinates": [[x1, y1], [x2, y2]]
        - circle: "coordinates": [center_x, center_y, radius]
        - ellipse: "coordinates": [center_x, center_y, width, height]
        - arc: "coordinates": [center_x, center_y, radius], "start_angle": degrees, "end_angle": degrees (outline only)
        - sector: "coordinates": [center_x, center_y, radius], "start_angle": degrees, "end_angle": degrees (pie slice - filled wedge from center)
        - chord: "coordinates": [center_x, center_y, radius], "start_angle": degrees, "end_angle": degrees (segment cut by straight line)
        - half_circle: "coordinates": [center_x, center_y, radius], "start_angle": degrees, "end_angle": degrees (special case of sector)
        - rectangle: "coordinates": [x1, y1, x2, y2]
        - triangle: "coordinates": [[x1, y1], [x2, y2], [x3, y3]]
        - polygon: "coordinates": [[x1, y1], [x2, y2], ..., [xn, yn]]
        - point: "coordinates": [x, y]
        - curve: "coordinates": [[x1, y1], [x2, y2], ...] (polyline through points)
        - spline: "coordinates": [[x1, y1], [x2, y2], ...] (smooth interpolated curve through points)
        - bezier: "coordinates": [[x1, y1], [x2, y2]], "control_points": [[cx1, cy1], [cx2, cy2]] (cubic Bezier curve)
        
        Important guidelines:
        1. Analyze the image dimensions and use them for canvas width/height
        2. Measure positions and sizes as accurately as possible
        3. Identify ALL shapes, lines, points, curves, and labels in the diagram
        4. Preserve relative positions and proportions precisely
        5. Use standard color names (red, blue, green, black, etc.) or RGB tuples like [255, 0, 0]
        6. For line width, estimate based on visual thickness (1-5 pixels typically)
        7. If a shape has a fill color, specify it; otherwise use null
        8. For each label, provide the text and its position coordinates
        9. Order shapes from background to foreground (draw order matters)
        10. Return ONLY the JSON object, no explanatory text before or after
        
        Be thorough and precise in your analysis!
        """
        
        print("Sending image to Gemini for analysis...")
        
        # Save image to bytes for upload
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or 'PNG')
        img_bytes = img_byte_arr.getvalue()
        
        # Use the client to generate content
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_text(prompt),
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=f'image/{img.format.lower()}' if img.format else 'image/png'
                )
            ],
            config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        print("\n" + "="*60)
        print("GEMINI RESPONSE:")
        print("="*60)
        print(response.text)
        print("="*60 + "\n")
        
        # Extract JSON from response
        json_str = self._extract_json(response.text)
        
        try:
            diagram_spec = json.loads(json_str)
            print("✓ JSON parsed successfully!")
            return diagram_spec
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON: {e}")
            print("\nAttempting to fix common JSON issues...")
            
            # Try to fix common issues
            json_str = self._fix_json(json_str)
            diagram_spec = json.loads(json_str)
            print("✓ JSON fixed and parsed successfully!")
            return diagram_spec
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from Gemini's response, handling markdown code blocks.
        
        Args:
            text: Raw text response from Gemini
            
        Returns:
            Clean JSON string
        """
        text = text.strip()
        
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find raw JSON (outermost braces)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def _fix_json(self, json_str: str) -> str:
        """
        Attempt to fix common JSON formatting issues.
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Fixed JSON string
        """
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        return json_str
    
    def save_specification(self, spec: Dict[str, Any], output_path: str):
        """
        Saves the diagram specification to a JSON file.

        Args:
            spec (Dict[str, Any]): The diagram specification dictionary.
            output_path (str): The path to save the JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"\n✓ Diagram specification saved to: {output_path}")
        
        # Print summary
        num_shapes = len(spec.get('shapes', []))
        canvas = spec.get('canvas', {})
        print(f"\nSummary:")
        print(f"  - Canvas: {canvas.get('width')}x{canvas.get('height')} pixels")
        print(f"  - Background: {canvas.get('background_color')}")
        print(f"  - Total shapes: {num_shapes}")
        
        # Count shape types
        shape_types = {}
        for shape in spec.get('shapes', []):
            shape_type = shape.get('type', 'unknown')
            shape_types[shape_type] = shape_types.get(shape_type, 0) + 1
        
        print(f"  - Shape breakdown:")
        for shape_type, count in sorted(shape_types.items()):
            print(f"    • {shape_type}: {count}")
    
    def analyze_and_save(self, image_path: str, output_json: str):
        """
        A convenience method that analyzes an image and saves the resulting
        diagram specification to a JSON file.

        Args:
            image_path (str): The path to the input image file.
            output_json (str): The path to save the output JSON file.
        """
        spec = self.analyze_diagram(image_path)
        self.save_specification(spec, output_json)
        return spec


def main():
    """
    Main function for Part 1: Image Analysis (Client-Based)
    """
    print("="*60)
    print("PART 1: GEOMETRIC DIAGRAM ANALYZER")
    print("(Using Client-Based Google GenAI)")
    print("="*60)
    print()
    
    # Get API key
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("\nPlease set your API key:")
        print("  Linux/Mac: export GEMINI_API_KEY='your-api-key-here'")
        print("  Windows:   set GEMINI_API_KEY=your-api-key-here")
        return
    
    # Input and output paths
    input_image = "input_diagram.png"
    output_json = "diagram_specification.json"
    
    # Check if input exists
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found")
        print("\nUsage:")
        print("  1. Place your geometric diagram image as 'input_diagram.png'")
        print("  2. Run: python part1_analyze_diagram_client.py")
        print("  3. Output will be saved as 'diagram_specification.json'")
        print("\nOr modify the 'input_image' variable in the script.")
        return
    
    # Create analyzer and process
    analyzer = DiagramAnalyzer(api_key)
    
    try:
        analyzer.analyze_and_save(input_image, output_json)
        print("\n" + "="*60)
        print("SUCCESS! You can now use Part 2 to draw the diagram.")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()spection: dict = Field(..., description="Inspection of penis, scrotum, perineum")
    palpation: dict = Field(..., description="Palpation: testicular size, consistency, tenderness, masses")
    hernia_exam: dict = Field(..., description="Inguinal hernia check, reducibility, tenderness")
    urethral_exam: dict = Field(..., description="Urethral meatus inspection, discharge, lesions")
    media: Optional[dict] = Field(None, description="Optional images or videos for telemedicine")

class LLMAssessment(BaseModel):
    primary_impression: str = Field(..., description="Primary impression from LLM or clinical reasoning")
    urgency: Literal["normal","monitor","urgent","emergency"] = Field(..., description="Triage urgency level")
    recommendations: List[str] from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

# ----------------------------
# 1. Pydantic Models
# ----------------------------
class PatientAnswer(BaseModel):
    question_id: str = Field(..., description="Unique patient question identifier")
    question_text: str = Field(..., description="Text of the question asked")
    answer_text: str = Field(..., description="Patient response")
    answer_code: Optional[str] = Field(None, description="LLM-coded response category")
    confidence: Optional[float] = Field(None, description="LLM confidence in coding")
    follow_up: Optional[List[str]] = Field(None, description="LLM-generated follow-up questions")

class NurseReport(BaseModel):
    = Field(..., description="Recommended clinical actions")
    confidence: Optional[float] = Field(None, description="Confidence of the assessment")

class MaleGenitaliaExam(BaseModel):
    patient_id: str = Field(..., description="Patient unique identifier")
    encounter_id: str = Field(..., description="Encounter unique identifier")
    timestamp: datetime = Field(..., description="Timestamp of exam")
    patient_answers: List[PatientAnswer] = Field(..., description="Patient history responses")
    nurse_report: NurseReport = Field(..., description="Structured nurse report")
    llm_assessment: LLMAssessment = Field(..., description="LLM evaluation")

# ----------------------------
# 2. Patient questions
# ----------------------------
patient_questions = [
    ("q1_pain", "Any pain in penis, scrotum, or testicles?"),
    ("q2_swelling", "Any swelling or lumps in testicles or scrotum?"),
    ("q3_discharge", "Any urethral discharge?"),
    ("q4_urination", "Any difficulty urinating, burning, or frequency changes?"),
    ("q5_erection", "Any erectile dysfunction or difficulty achieving erection?"),
    ("q6_trauma", "Any recent trauma or injury to genitalia?"),
    ("q7_infection_history", "Any history of STIs or urinary infections?"),
    ("q8_family_history", "Any family history of testicular or prostate disease?")
]


