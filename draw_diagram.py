from PIL import Image, ImageDraw, ImageFont
import json
import os
from typing import Dict, Any, List, Tuple


class DiagramDrawer:
    def __init__(self):
        """Initialize the diagram drawer."""
        self.font = self._load_font()
        
    def _load_font(self, size: int = 14):
        """
        Load font for text labels.
        
        Args:
            size: Font size
            
        Returns:
            PIL Font object
        """
        try:
            # Try to load TrueType font
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except:
            try:
                # Try alternative font paths
                return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", size)
            except:
                # Fall back to default font
                return ImageFont.load_default()
    
    def load_specification(self, json_path: str) -> Dict[str, Any]:
        """
        Load diagram specification from JSON file.
        
        Args:
            json_path: Path to JSON specification file
            
        Returns:
            Dictionary containing diagram specifications
        """
        print(f"Loading specification from: {json_path}")
        
        with open(json_path, 'r') as f:
            spec = json.load(f)
        
        print("✓ Specification loaded successfully!")
        
        # Print summary
        num_shapes = len(spec.get('shapes', []))
        canvas = spec.get('canvas', {})
        print(f"\nDiagram Info:")
        print(f"  - Canvas: {canvas.get('width')}x{canvas.get('height')} pixels")
        print(f"  - Background: {canvas.get('background_color')}")
        print(f"  - Total shapes: {num_shapes}")
        
        return spec
    
    def draw_diagram(self, spec: Dict[str, Any], output_path: str):
        """
        Draw the diagram based on the JSON specification.
        
        Args:
            spec: Diagram specification dictionary
            output_path: Path to save the output image
        """
        print(f"\nDrawing diagram...")
        
        # Create canvas
        canvas_spec = spec.get('canvas', {})
        width = canvas_spec.get('width', 800)
        height = canvas_spec.get('height', 600)
        bg_color = self._parse_color(canvas_spec.get('background_color', 'white'))
        
        # Create image
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw each shape
        shapes = spec.get('shapes', [])
        for i, shape in enumerate(shapes, 1):
            try:
                self._draw_shape(draw, shape)
                print(f"  ✓ Drew shape {i}/{len(shapes)}: {shape.get('type', 'unknown')}")
            except Exception as e:
                print(f"  ✗ Error drawing shape {i}: {e}")
        
        # Save the image
        img.save(output_path)
        print(f"\n✓ Diagram saved to: {output_path}")
    
    def _parse_color(self, color) -> Tuple[int, int, int] | str:
        """
        Parse color from various formats to RGB tuple or color name.
        
        Args:
            color: Color as string name, RGB list, or tuple
            
        Returns:
            Color in format accepted by Pillow
        """
        if color is None:
            return 'black'
        elif isinstance(color, str):
            return color
        elif isinstance(color, list) and len(color) == 3:
            return tuple(color)
        elif isinstance(color, tuple) and len(color) == 3:
            return color
        else:
            return 'black'
    
    def _draw_shape(self, draw: ImageDraw.Draw, shape: Dict[str, Any]):
        """
        Draw a single shape on the canvas.
        
        Args:
            draw: PIL ImageDraw object
            shape: Shape specification dictionary
        """
        shape_type = shape.get('type', '').lower()
        coords = shape.get('coordinates', [])
        color = self._parse_color(shape.get('color', 'black'))
        width = shape.get('width', 2)
        fill = self._parse_color(shape.get('fill')) if shape.get('fill') else None
        
        if shape_type == 'line':
            self._draw_line(draw, coords, color, width)
            
        elif shape_type == 'circle':
            self._draw_circle(draw, coords, color, width, fill)
            
        elif shape_type == 'ellipse':
            self._draw_ellipse(draw, coords, color, width, fill)
            
        elif shape_type == 'arc':
            self._draw_arc(draw, coords, color, width, shape)
            
        elif shape_type == 'sector':
            self._draw_sector(draw, coords, color, width, fill, shape)
            
        elif shape_type == 'chord':
            self._draw_chord(draw, coords, color, width, fill, shape)
            
        elif shape_type == 'half_circle':
            self._draw_half_circle(draw, coords, color, width, fill, shape)
            
        elif shape_type == 'rectangle':
            self._draw_rectangle(draw, coords, color, width, fill)
            
        elif shape_type == 'triangle':
            self._draw_triangle(draw, coords, color, width, fill)
            
        elif shape_type == 'polygon':
            self._draw_polygon(draw, coords, color, width, fill)
            
        elif shape_type == 'point':
            self._draw_point(draw, coords, color)
            
        elif shape_type == 'curve':
            self._draw_curve(draw, coords, color, width)
            
        elif shape_type == 'spline':
            self._draw_spline(draw, coords, color, width)
            
        elif shape_type == 'bezier':
            self._draw_bezier(draw, coords, color, width, shape)
        
        # Draw label if present
        if shape.get('label'):
            label_pos = shape.get('label_position')
            if label_pos and len(label_pos) == 2:
                draw.text(
                    (label_pos[0], label_pos[1]), 
                    shape['label'], 
                    fill=color, 
                    font=self.font
                )
    
    def _draw_line(self, draw: ImageDraw.Draw, coords: List, color, width: int):
        """Draw a line."""
        if len(coords) >= 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    
    def _draw_circle(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill):
        """Draw a circle."""
        if len(coords) >= 3:
            cx, cy, radius = coords[0], coords[1], coords[2]
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            
            if fill:
                draw.ellipse(bbox, outline=color, fill=fill, width=width)
            else:
                draw.ellipse(bbox, outline=color, width=width)
    
    def _draw_rectangle(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill):
        """Draw a rectangle."""
        if len(coords) >= 4:
            bbox = [coords[0], coords[1], coords[2], coords[3]]
            
            if fill:
                draw.rectangle(bbox, outline=color, fill=fill, width=width)
            else:
                draw.rectangle(bbox, outline=color, width=width)
    
    def _draw_triangle(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill):
        """Draw a triangle."""
        if len(coords) >= 3:
            points = [
                (coords[0][0], coords[0][1]),
                (coords[1][0], coords[1][1]),
                (coords[2][0], coords[2][1])
            ]
            
            if fill:
                draw.polygon(points, outline=color, fill=fill)
            else:
                # Draw outline manually for better control
                draw.line([points[0], points[1]], fill=color, width=width)
                draw.line([points[1], points[2]], fill=color, width=width)
                draw.line([points[2], points[0]], fill=color, width=width)
    
    def _draw_polygon(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill):
        """Draw a polygon."""
        if len(coords) >= 3:
            points = [(pt[0], pt[1]) for pt in coords]
            
            if fill:
                draw.polygon(points, outline=color, fill=fill)
            else:
                # Draw outline
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    draw.line([start, end], fill=color, width=width)
    
    def _draw_point(self, draw: ImageDraw.Draw, coords: List, color):
        """Draw a point as a small filled circle."""
        if len(coords) >= 2:
            x, y = coords[0], coords[1]
            radius = 4  # Point radius
            bbox = [x - radius, y - radius, x + radius, y + radius]
            draw.ellipse(bbox, fill=color)
    
    def _draw_ellipse(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill):
        """Draw an ellipse."""
        if len(coords) >= 4:
            cx, cy, ell_width, ell_height = coords[0], coords[1], coords[2], coords[3]
            bbox = [
                cx - ell_width/2, 
                cy - ell_height/2, 
                cx + ell_width/2, 
                cy + ell_height/2
            ]
            
            if fill:
                draw.ellipse(bbox, outline=color, fill=fill, width=width)
            else:
                draw.ellipse(bbox, outline=color, width=width)
    
    def _draw_arc(self, draw: ImageDraw.Draw, coords: List, color, width: int, shape: Dict):
        """Draw an arc (outline only, no fill)."""
        if len(coords) >= 3:
            cx, cy, radius = coords[0], coords[1], coords[2]
            start_angle = shape.get('start_angle', 0)
            end_angle = shape.get('end_angle', 180)
            
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.arc(bbox, start=start_angle, end=end_angle, fill=color, width=width)
    
    def _draw_sector(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill, shape: Dict):
        """Draw a sector (pie slice - wedge from center with optional fill)."""
        if len(coords) >= 3:
            cx, cy, radius = coords[0], coords[1], coords[2]
            start_angle = shape.get('start_angle', 0)
            end_angle = shape.get('end_angle', 90)
            
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            
            # pieslice draws a wedge from the center
            if fill:
                draw.pieslice(bbox, start=start_angle, end=end_angle, fill=fill, outline=color, width=width)
            else:
                draw.pieslice(bbox, start=start_angle, end=end_angle, outline=color, width=width)
    
    def _draw_chord(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill, shape: Dict):
        """Draw a chord segment (circular segment cut by a straight line)."""
        if len(coords) >= 3:
            cx, cy, radius = coords[0], coords[1], coords[2]
            start_angle = shape.get('start_angle', 0)
            end_angle = shape.get('end_angle', 90)
            
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            
            # chord draws the circular segment without lines to center
            if fill:
                draw.chord(bbox, start=start_angle, end=end_angle, fill=fill, outline=color, width=width)
            else:
                draw.chord(bbox, start=start_angle, end=end_angle, outline=color, width=width)
    
    def _draw_half_circle(self, draw: ImageDraw.Draw, coords: List, color, width: int, fill, shape: Dict):
        """Draw a half circle."""
        if len(coords) >= 3:
            cx, cy, radius = coords[0], coords[1], coords[2]
            start_angle = shape.get('start_angle', 0)
            end_angle = shape.get('end_angle', 180)
            
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            
            if fill:
                draw.pieslice(bbox, start=start_angle, end=end_angle, fill=fill, outline=color, width=width)
            else:
                draw.arc(bbox, start=start_angle, end=end_angle, fill=color, width=width)
    
    def _draw_curve(self, draw: ImageDraw.Draw, coords: List, color, width: int):
        """Draw a curve (polyline through points)."""
        if len(coords) >= 2:
            points = [(pt[0], pt[1]) for pt in coords]
            # Draw curve by connecting points
            for i in range(len(points) - 1):
                draw.line([points[i], points[i + 1]], fill=color, width=width)
    
    def _draw_spline(self, draw: ImageDraw.Draw, coords: List, color, width: int):
        """Draw a smooth spline through points using Catmull-Rom interpolation."""
        if len(coords) < 2:
            return
        
        points = [(pt[0], pt[1]) for pt in coords]
        
        if len(points) == 2:
            # Just a line for 2 points
            draw.line(points, fill=color, width=width)
            return
        
        # Catmull-Rom spline interpolation
        def catmull_rom_point(p0, p1, p2, p3, t):
            """Calculate point on Catmull-Rom curve."""
            t2 = t * t
            t3 = t2 * t
            
            x = 0.5 * ((2 * p1[0]) +
                      (-p0[0] + p2[0]) * t +
                      (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                      (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            
            y = 0.5 * ((2 * p1[1]) +
                      (-p0[1] + p2[1]) * t +
                      (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                      (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            
            return (x, y)
        
        # Generate smooth curve
        segments = 10  # Points per segment
        curve_points = []
        
        for i in range(len(points) - 1):
            p0 = points[max(0, i - 1)]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[min(len(points) - 1, i + 2)]
            
            for j in range(segments):
                t = j / segments
                pt = catmull_rom_point(p0, p1, p2, p3, t)
                curve_points.append(pt)
        
        curve_points.append(points[-1])
        
        # Draw the smooth curve
        for i in range(len(curve_points) - 1):
            draw.line([curve_points[i], curve_points[i + 1]], fill=color, width=width)
    
    def _draw_bezier(self, draw: ImageDraw.Draw, coords: List, color, width: int, shape: Dict):
        """Draw a cubic Bezier curve."""
        if len(coords) < 2:
            return
        
        control_points = shape.get('control_points', [])
        
        if len(control_points) < 2:
            # No control points, draw as straight line
            draw.line([(coords[0][0], coords[0][1]), (coords[1][0], coords[1][1])], 
                     fill=color, width=width)
            return
        
        # Cubic Bezier: P0, C1, C2, P1
        p0 = (coords[0][0], coords[0][1])
        p1 = (coords[1][0], coords[1][1])
        c1 = (control_points[0][0], control_points[0][1])
        c2 = (control_points[1][0], control_points[1][1])
        
        # Generate points along the Bezier curve
        def bezier_point(t):
            """Calculate point on cubic Bezier curve at parameter t."""
            u = 1 - t
            u2 = u * u
            u3 = u2 * u
            t2 = t * t
            t3 = t2 * t
            
            x = u3 * p0[0] + 3 * u2 * t * c1[0] + 3 * u * t2 * c2[0] + t3 * p1[0]
            y = u3 * p0[1] + 3 * u2 * t * c1[1] + 3 * u * t2 * c2[1] + t3 * p1[1]
            
            return (x, y)
        
        # Generate curve points
        steps = 50
        curve_points = [bezier_point(i / steps) for i in range(steps + 1)]
        
        # Draw the curve
        for i in range(len(curve_points) - 1):
            draw.line([curve_points[i], curve_points[i + 1]], fill=color, width=width)
    
    def load_and_draw(self, json_path: str, output_path: str):
        """
        Complete pipeline: load JSON and draw diagram.
        
        Args:
            json_path: Path to JSON specification
            output_path: Path to save output image
        """
        spec = self.load_specification(json_path)
        self.draw_diagram(spec, output_path)


def main():
    """
    Main function for Part 2: Drawing from JSON
    """
    print("="*60)
    print("PART 2: GEOMETRIC DIAGRAM DRAWER")
    print("="*60)
    print()
    
    # Input and output paths
    input_json = "diagram_specification.json"
    output_image = "recreated_diagram.png"
    
    # Check if JSON file exists
    if not os.path.exists(input_json):
        print(f"Error: JSON specification '{input_json}' not found")
        print("\nUsage:")
        print("  1. First run Part 1 to create the JSON specification")
        print("  2. Then run: python part2_draw_diagram.py")
        print("  3. Output will be saved as 'recreated_diagram.png'")
        print("\nOr modify the 'input_json' variable in the script.")
        return
    
    # Create drawer and process
    drawer = DiagramDrawer()
    
    try:
        drawer.load_and_draw(input_json, output_image)
        print("\n" + "="*60)
        print("SUCCESS! Diagram has been recreated.")
        print(f"Check the output: {output_image}")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during drawing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

