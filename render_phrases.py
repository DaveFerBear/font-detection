from playwright.sync_api import sync_playwright
from pathlib import Path
import pandas as pd
import random
import shutil
import colorsys
from urllib.parse import unquote
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from fonts import FONTS

def _process_font_worker(args):
    """Worker function to process a single font (runs in separate process)"""
    font_family, texts, output_dir = args

    # Each worker creates its own generator with its own browser
    generator = FontDatasetGenerator(output_dir=output_dir)

    try:
        # Start browser for this worker - load all fonts once to reuse browser
        generator.start_browser(FONTS)

        # Create font directory
        font_dir = generator.output_dir / font_family.replace(' ', '_')
        font_dir.mkdir(exist_ok=True, parents=True)

        # Generate samples for this font
        for text_idx, text in enumerate(texts):
            screenshot = generator.render_font_sample(text, font_family)

            filename = f"sample_{text_idx:04d}.png"
            filepath = font_dir / filename

            with open(filepath, 'wb') as f:
                f.write(screenshot)

    except Exception as e:
        print(f"Error processing {font_family}: {e}")
        return None
    finally:
        generator.stop_browser()

    return font_family

class FontDatasetGenerator:
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.playwright = None
        self.browser = None
        self.page = None
    
    def generate_contrasting_colors(self):
        """Generate background and text colors with sufficient contrast"""
        # Generate random background color
        bg_hue = random.random()
        bg_sat = random.uniform(0.1, 0.9)
        bg_val = random.uniform(0.2, 0.9)
        
        bg_r, bg_g, bg_b = colorsys.hsv_to_rgb(bg_hue, bg_sat, bg_val)
        bg_r, bg_g, bg_b = int(bg_r * 255), int(bg_g * 255), int(bg_b * 255)
        
        # Calculate luminance for contrast
        def luminance(r, g, b):
            r, g, b = [x/255.0 for x in (r, g, b)]
            r = r/12.92 if r <= 0.03928 else ((r+0.055)/1.055)**2.4
            g = g/12.92 if g <= 0.03928 else ((g+0.055)/1.055)**2.4
            b = b/12.92 if b <= 0.03928 else ((b+0.055)/1.055)**2.4
            return 0.2126*r + 0.7152*g + 0.0722*b
        
        bg_lum = luminance(bg_r, bg_g, bg_b)
        
        # Choose text color for good contrast (aim for 4.5:1 ratio minimum)
        if bg_lum > 0.5:
            # Light background, use dark text
            text_val = random.uniform(0.0, 0.3)
        else:
            # Dark background, use light text
            text_val = random.uniform(0.7, 1.0)
            
        text_hue = random.random()
        text_sat = random.uniform(0.0, 0.8)
        
        text_r, text_g, text_b = colorsys.hsv_to_rgb(text_hue, text_sat, text_val)
        text_r, text_g, text_b = int(text_r * 255), int(text_g * 255), int(text_b * 255)
        
        return f"rgb({bg_r},{bg_g},{bg_b})", f"rgb({text_r},{text_g},{text_b})"

    def categorize_fonts(self):
        """Categorize fonts into system, Google, and local fonts"""
        # System fonts (available on most systems)
        system_fonts = [
            "Arial", "Helvetica", "Times New Roman",
            "Georgia", "Courier New", "Verdana"
        ]

        # Local fonts (loaded from fonts.css)
        local_fonts = [
            "Tusker Grotesk", "Saveur Sans Round", "Eurotype BKL",
            "Extenda", "Bobby Jones Soft", "Bobby Jones Soft Outline",
            "Bobby Rough Soft", "Bobby Rough Soft Outline", "Ashing",
            "Bondjlo", "Dodo", "English 111 Presto", "Faylake",
            "Felt Tip", "Fontuna Stencil", "Merisca", "Natalic",
            "Kaylar", "Posterman", "Tokyo OneSolid Regular"
        ]

        # Google Fonts (everything else from FONTS list)
        google_fonts = [font for font in FONTS if font not in system_fonts and font not in local_fonts]

        return {
            'system': system_fonts,
            'google': google_fonts,
            'local': local_fonts
        }
    
    def start_browser(self, fonts):
        """Initialize browser instance with all fonts preloaded"""
        if not self.playwright:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()

            # Set up route to serve local font files
            fonts_dir = Path(__file__).parent / "fonts"

            def handle_font_route(route):
                # Extract the font path from the URL
                url = route.request.url
                if "/local-fonts/" in url:
                    # Get the relative path after "/local-fonts/" and URL-decode it
                    rel_path = url.split("/local-fonts/")[1]
                    rel_path = unquote(rel_path)  # Decode %20 to space, etc.
                    font_file = fonts_dir / rel_path

                    if font_file.exists():
                        route.fulfill(path=str(font_file))
                    else:
                        route.abort()
                else:
                    route.continue_()

            self.page.route("**/*", handle_font_route)
            self._setup_fonts(fonts)
    
    def stop_browser(self):
        """Clean up browser instance"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def _setup_fonts(self, fonts):
        """Setup HTML page with all fonts preloaded"""
        # Categorize fonts
        categorized = self.categorize_fonts()

        # Filter fonts to only include Google Fonts for API loading
        google_fonts_to_load = [font for font in fonts if font in categorized['google']]

        # Generate Google Fonts API links
        font_links = '\n'.join([
            f'<link href="https://fonts.googleapis.com/css2?family={font.replace(" ", "+")}:wght@400;700&display=swap" rel="stylesheet">'
            for font in google_fonts_to_load
        ])

        # Read local fonts.css file and convert relative paths to use our custom routing
        fonts_css_path = Path(__file__).parent / "fonts.css"
        local_fonts_css = ""
        if fonts_css_path.exists():
            with open(fonts_css_path, 'r') as f:
                local_fonts_css = f.read()
                # Convert relative URLs to use our custom HTTP-like scheme that Playwright can intercept
                # The fonts.css uses "../fonts/" which we'll convert to "http://local-fonts/"
                local_fonts_css = local_fonts_css.replace('url("../fonts/', 'url("http://local-fonts/')
                local_fonts_css = local_fonts_css.replace("url('../fonts/", "url('http://local-fonts/")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            {font_links}
            <style>
                {local_fonts_css}
                body {{ margin: 0; padding: 20px; }}
                #container {{ background: white; }}
            </style>
        </head>
        <body>
            <div id="container"></div>
            <script>
                function renderText(text, fontFamily, containerWidth, fontSize, paddingTop, paddingRight, paddingBottom, paddingLeft, textAlign, bgColor, textColor, fontWeight) {{
                    const container = document.getElementById('container');
                    container.style.width = containerWidth + 'px';
                    container.style.fontFamily = '"' + fontFamily + '", sans-serif';
                    container.style.fontSize = fontSize + 'px';
                    container.style.color = textColor || 'black';
                    container.style.backgroundColor = bgColor || 'white';
                    container.style.fontWeight = fontWeight || 'normal';
                    container.style.wordWrap = 'break-word';
                    container.style.padding = paddingTop + 'px ' + paddingRight + 'px ' + paddingBottom + 'px ' + paddingLeft + 'px';
                    container.style.textAlign = textAlign;
                    container.textContent = text;
                }}
            </script>
        </body>
        </html>
        """

        self.page.set_content(html_content)

        # Wait for fonts to load
        self.page.evaluate("""
            async () => {
                await document.fonts.ready;
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        """)
    
    def render_font_sample(self, text, font_family):
        """Render text with specified font in a container of random width and size"""
        
        font_size = random.randint(10, 100)
        
        # Random padding for each side
        padding_top = random.randint(0, 150)
        padding_right = random.randint(0, 150)
        padding_bottom = random.randint(0, 150)
        padding_left = random.randint(0, 150)

        # Adjust container width based on text length
        container_width = int(len(text) * font_size / 10. + random.randint(200, 800))
        
        # Random text alignment
        alignments = ['left', 'center', 'right']
        text_alignment = random.choice(alignments)
        
        # Use color 50% of the time
        use_color = random.random() < 0.5
        if use_color:
            bg_color, text_color = self.generate_contrasting_colors()
        else:
            bg_color, text_color = 'white', 'black'
        
        # Random font weight (50% chance of bold)
        font_weight = 'bold' if random.random() < 0.5 else 'normal'
        
        # Escape text for JavaScript
        escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
        
        # Render text in container
        self.page.evaluate(f'''
            renderText("{escaped_text}", "{font_family}", {container_width}, {font_size}, {padding_top}, {padding_right}, {padding_bottom}, {padding_left}, "{text_alignment}", "{bg_color}", "{text_color}", "{font_weight}")
        ''')
        
        # Take screenshot of container
        container = self.page.locator('#container')
        screenshot = container.screenshot()
        
        return screenshot
    
    def load_phrases(self, csv_path="phrases_10000.csv"):
        """Load phrases from CSV file"""
        df = pd.read_csv(csv_path)
        return df['phrase'].tolist()
    
    def generate_samples(self, texts=None, fonts=None, samples_per_font=500, num_workers=None):
        """Generate font samples and save as images using multiprocessing"""
        # Clear existing data folder
        if self.output_dir.exists():
            print(f"Clearing existing data folder: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if texts is None:
            all_phrases = self.load_phrases()
            texts = random.sample(all_phrases, min(samples_per_font, len(all_phrases)))

        if fonts is None:
            fonts = FONTS  # Use all fonts from fonts.py

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)  # Leave one core free

        print(f"Generating samples for {len(fonts)} fonts using {num_workers} workers...")

        # Create args for each font
        tasks = [(font, texts[:samples_per_font], str(self.output_dir)) for font in fonts]

        # Use multiprocessing pool
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap(_process_font_worker, tasks), total=len(tasks), desc="Fonts", unit="font"))

        print("Generation complete!")

    def generate_samples_single_thread(self, texts=None, fonts=None, samples_per_font=500):
        """Single-threaded version (for debugging)"""
        # Clear existing data folder
        if self.output_dir.exists():
            print(f"Clearing existing data folder: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)

        if texts is None:
            all_phrases = self.load_phrases()
            texts = random.sample(all_phrases, min(samples_per_font, len(all_phrases)))

        if fonts is None:
            fonts = FONTS  # Use all fonts from fonts.py

        print(f"Generating samples for {len(fonts)} fonts (single-threaded)...")

        try:
            self.start_browser(fonts)

            for font_family in tqdm(fonts, desc="Fonts", unit="font"):
                font_dir = self.output_dir / font_family.replace(' ', '_')
                font_dir.mkdir(exist_ok=True)

                for text_idx, text in enumerate(tqdm(texts[:samples_per_font], desc=f"  {font_family}", leave=False, unit="sample")):
                    screenshot = self.render_font_sample(text, font_family)

                    filename = f"sample_{text_idx:04d}.png"
                    filepath = font_dir / filename

                    with open(filepath, 'wb') as f:
                        f.write(screenshot)

        finally:
            self.stop_browser()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate font dataset')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count - 1)')
    parser.add_argument('--samples', type=int, default=500, help='Samples per font (default: 500)')
    parser.add_argument('--single-thread', action='store_true', help='Use single-threaded mode (for debugging)')
    args = parser.parse_args()

    generator = FontDatasetGenerator()

    if args.single_thread:
        generator.generate_samples_single_thread(samples_per_font=args.samples)
    else:
        generator.generate_samples(samples_per_font=args.samples, num_workers=args.workers)
