import textwrap

from gemini_client import GeminiClient

def generate_kg_prompt(document_text: str) -> str:
    """
    Generates a structured prompt for creating a Knowledge Graph from a document.

    This function creates a detailed prompt that instructs a language model to
    extract entities, relationships, and attributes from a given text and format
    them as a list of triples and entity attributes.

    Args:
        document_text (str): The source text from which the knowledge graph
                             should be generated.

    Returns:
        str: A formatted string containing the full prompt.
    """

    # Define the core instructions for the Knowledge Graph generation
    instructions = textwrap.dedent("""
    Create a comprehensive **Knowledge Graph** from the following document, ensuring the graph includes:

    1.  **Nodes (Entities):** Identify and extract all primary and secondary entities (e.g., people, organizations, locations, concepts, dates, events, products) mentioned.
    2.  **Edges (Relationships):** Define the semantic relationships (e.g., "is a member of," "is located in," "was founded by," "works on," "produced") that connect the identified entities.
    3.  **Attributes:** Include relevant properties or attributes for key entities (e.g., "date founded" for an organization, "title" for a person, "population" for a location).

    **Output Format:** Present the graph as a list of **triples** in the format `(Source_Entity, Relationship, Target_Entity)` followed by a separate list of **Entity Attributes** in the format `(Entity, Attribute_Name, Attribute_Value)`.

    ---
    *Example Triple:* (Apple Inc., was founded by, Steve Jobs)
    *Example Attribute:* (Apple Inc., Headquarters, Cupertino, CA)
    ---
    """)

    # Combine the instructions and the document text
    full_prompt = (
        instructions +
        "\n**Document:**\n" +
        "==========================================================\n" +
        document_text +
        "\n==========================================================\n"
    )

    return full_prompt.strip()

def main():
    """
    An example of how to use the `generate_kg_prompt` function.
    """
    sample_document = (
        "Project Astra, an initiative by Google DeepMind, was announced in May 2024. "
        "It aims to create a universal AI assistant capable of understanding and "
        "interacting with the world in real-time. The lead researcher on the project "
        "is Dr. Jane Doe, who is based in London, UK. DeepMind is headquartered "
        "in London and was founded in 2010."
    )

    prompt = generate_kg_prompt(sample_document)

    llm = GeminiClient()
    text = llm.generate_text(prompt)
    print(text)

if __name__ == "__main__":
    main()


