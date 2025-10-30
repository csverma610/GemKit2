import os
import sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# --------------------------
# Pydantic Models
# --------------------------
class Node(BaseModel):
    """
    A Pydantic model representing a node in the object graph.
    """
    id: int
    label: str
    position: Optional[str] = Field(
        None,
        description="One of N, NE, E, SE, S, SW, W, NW, Center"
    )

class NodeRef(BaseModel):
    """
    A Pydantic model representing a reference to a node in the object graph.
    """
    id: int
    label: str

class Edge(BaseModel):
    """
    A Pydantic model representing an edge in the object graph.
    """
    source: NodeRef
    target: NodeRef
    relation: str = Field(
        ...,
        description="Relationship type: nearby, occludes, in_front_of"
    )

class Graph(BaseModel):
    """
    A Pydantic model representing the entire object graph.
    """
    nodes: List[Node]
    edges: List[Edge]

# --------------------------
# ObjectGraph Class
# --------------------------
class ObjectGraph:
    """
    Generates a spatial object graph from an image using the Gemini API.

    This class takes an image as input and uses the Gemini model to identify
    objects and their spatial relationships, representing them as a graph of
    nodes and edges.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the ObjectGraph instance.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-2.5-flash".
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing environment variable: GEMINI_API_KEY")
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def _load_image_bytes(self, image_path: str) -> bytes:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()

    def generate_graph(self, image_path: str) -> Graph:
        """
        Generates a spatial object graph from an image.

        Args:
            image_path (str): The path to the input image file.

        Returns:
            Graph: A Pydantic model representing the generated object graph.
        """
        image_bytes = self._load_image_bytes(image_path)

        # Official way to pass image to Gemini
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

        prompt_text = (
            "Analyze this image and generate a spatial object graph. "
            "Divide the image into nine subgrids: N, NE, E, SE, S, SW, W, NW, Center. "
            "Return JSON with 'nodes' (id, label, position) and "
            "'edges' (source: {id, label}, target: {id, label}, relation)."
        )

        # Use official response_schema style
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt_text, image_part],
            config={
                "response_mime_type": "application/json",
                "response_schema": Graph,
            },
        )

        # parsed is a Graph object (Pydantic instance)
        return response.parsed

# --------------------------
# CLI Entry Point
# --------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gemini_object_graph.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    graph = ObjectGraph()
    result = graph.generate_graph(image_path)

    print(result.model_dump_json(indent=2))

