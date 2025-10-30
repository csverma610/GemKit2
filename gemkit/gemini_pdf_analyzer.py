import sys
import os
import pathlib
import pypdf
import shutil
import json
from datetime import datetime
from google import genai
from google.genai import types

class PDFSplitter:
    """
    A utility class for handling PDF file operations, including splitting
    large PDFs into smaller chunks.
    """
    
    def __init__(self, max_pages: int = 100, max_size: int = 10 * 1024 * 1024):
        """
        Initializes the PDFSplitter.

        Args:
            max_pages (int, optional): The maximum number of pages per chunk.
            max_size (int, optional): The maximum size of a chunk in bytes.
        """
        self.max_pages = max_pages
        self.max_size = max_size
    
    def get_info(self, pdf_path: pathlib.Path) -> dict:
        """
        Retrieves information about a PDF file.

        Args:
            pdf_path (pathlib.Path): The path to the PDF file.

        Returns:
            dict: A dictionary containing the file size, number of pages, and
                  whether the PDF needs to be split.
        """
        file_size = os.path.getsize(pdf_path)
        num_pages = len(pypdf.PdfReader(pdf_path).pages)
        needs_split = num_pages > self.max_pages or file_size > self.max_size
        return {
            "size": file_size,
            "pages": num_pages,
            "needs_split": needs_split
        }
    
    def split(self, pdf_path: pathlib.Path, temp_dir: pathlib.Path) -> list[pathlib.Path]:
        """
        Splits a PDF file into smaller chunks.

        Args:
            pdf_path (pathlib.Path): The path to the PDF file to split.
            temp_dir (pathlib.Path): The directory to save the chunks in.

        Returns:
            list[pathlib.Path]: A list of paths to the created PDF chunks.
        """
        reader = pypdf.PdfReader(pdf_path)
        chunks = []
        
        for start in range(0, len(reader.pages), self.max_pages):
            writer = pypdf.PdfWriter()
            end = min(start + self.max_pages, len(reader.pages))
            
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            
            chunk_path = temp_dir / f"chunk_{start // self.max_pages + 1}.pdf"
            with open(chunk_path, "wb") as f:
                writer.write(f)
            chunks.append(chunk_path)
        
        return chunks


class GeminiPDFAnalyzer:
    """
    Analyzes PDF files using the Gemini API.

    This class can handle large PDF files by splitting them into smaller chunks
    and processing each chunk individually. The results for each chunk are saved
    to separate JSON files, and a merged output is also created.
    """
    
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initializes the GeminiPDFAnalyzer.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-2.5-flash".
        """
        self.model_name = model_name
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        self.splitter = PDFSplitter()

    def _process_chunk(self, chunk_path: pathlib.Path, prompt: str, 
                      chunk_num: int, total: int) -> dict:
        """Process a single PDF chunk with Gemini"""
        file_bytes = chunk_path.read_bytes()
        chunk_prompt = f"[Part {chunk_num}/{total}]\n\n{prompt}" if total > 1 else prompt
        
        contents = [
            types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'),
            chunk_prompt
        ]
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )
        
        return {
            "chunk": chunk_num,
            "total": total,
            "timestamp": datetime.now().isoformat(),
            "response": response.text,
            "status": "success"
        }

    def generate_text(self, pdf_path: str, prompt: str, output_dir: str = None) -> str:
        """
        Analyzes a PDF file and generates a text response.

        This method handles the entire process of splitting the PDF if necessary,
        processing each chunk with the Gemini API, saving the individual results,
        and creating a merged output.

        Args:
            pdf_path (str): The path to the PDF file.
            prompt (str): The prompt to use for the analysis.
            output_dir (str, optional): The directory to save the output files.
                                        If not provided, a directory will be created
                                        based on the PDF's filename.

        Returns:
            str: The combined text response from all chunks.
        """
        pdf_file = pathlib.Path(pdf_path)
        pdf_name = pdf_file.stem
        
        # Setup output directory
        out_dir = pathlib.Path(output_dir or f"{pdf_name}_output")
        out_dir.mkdir(exist_ok=True)
        
        # Get PDF info and check if splitting needed
        pdf_info = self.splitter.get_info(pdf_file)
        
        temp_dir = pathlib.Path("_temp_chunks")
        try:
            # Prepare chunks
            if pdf_info["needs_split"]:
                print(f"Splitting PDF: {pdf_info['pages']} pages, {pdf_info['size']:,} bytes")
                temp_dir.mkdir(exist_ok=True)
                chunks = self.splitter.split(pdf_file, temp_dir)
            else:
                chunks = [pdf_file]
            
            # Process each chunk
            results = []
            for i, chunk_path in enumerate(chunks, 1):
                print(f"Processing chunk {i}/{len(chunks)}...")
                
                try:
                    result = self._process_chunk(chunk_path, prompt, i, len(chunks))
                    # Save immediately
                    output_file = out_dir / f"{pdf_name}_chunk_{i:03d}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"✓ Saved: {output_file.name}")
                    results.append(result)
                    
                except Exception as e:
                    print(f"✗ Error on chunk {i}: {e}")
                    error_result = {
                        "chunk": i,
                        "total": len(chunks),
                        "timestamp": datetime.now().isoformat(),
                        "response": str(e),
                        "status": "error"
                    }
                    output_file = out_dir / f"{pdf_name}_chunk_{i:03d}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False)
            
            # Create merged output
            merged = {
                "pdf": pdf_name,
                "timestamp": datetime.now().isoformat(),
                "chunks_processed": len(results),
                "chunks_total": len(chunks),
                "combined": "\n\n".join(r["response"] for r in results if r["status"] == "success")
            }
            
            merged_file = out_dir / f"{pdf_name}_merged.json"
            with open(merged_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Merged output: {merged_file}")
            return merged["combined"]
            
        finally:
            # Cleanup temp files
            if pdf_info["needs_split"] and temp_dir.exists():
                shutil.rmtree(temp_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze PDF with Gemini API")
    parser.add_argument("-i", "--pdf_file", required=True, help="PDF file path")
    parser.add_argument("-p", "--prompt", required=True, help="Analysis prompt")
    parser.add_argument("-o", "--output_dir", help="Output directory (default: <pdf>_output)")
    args = parser.parse_args()

    try:
        analyzer = GeminiPDFAnalyzer()
        result = analyzer.generate_text(args.pdf_file, args.prompt, args.output_dir)
        print("\n" + "="*80)
        print("COMBINED OUTPUT:")
        print("="*80)
        print(result)
    except KeyboardInterrupt:
        print("\nInterrupted. Partial results saved.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
