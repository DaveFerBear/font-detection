from playwright.sync_api import sync_playwright
from pathlib import Path
import pandas as pd
import random
import shutil

class FontDatasetGenerator:
    def __init__(self, output_dir="data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.playwright = None
        self.browser = None
        self.page = None
        
    def get_google_fonts(self, limit=50):
        """Get list of popular Google Fonts"""
        return [
            'Open Sans', 'Roboto', 'Lato', 'Montserrat', 'Source Sans Pro',
            'Roboto Condensed', 'Oswald', 'Roboto Mono', 'Raleway', 'Nunito',
            'Ubuntu', 'Playfair Display', 'Merriweather', 'Poppins', 'Inter'
        ][:limit]
    
    def start_browser(self, fonts):
        """Initialize browser instance with all fonts preloaded"""
        if not self.playwright:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()
            self._setup_fonts(fonts)
    
    def stop_browser(self):
        """Clean up browser instance"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def _setup_fonts(self, fonts):
        """Setup HTML page with all fonts preloaded"""
        font_links = '\n'.join([
            f'<link href="https://fonts.googleapis.com/css2?family={font.replace(" ", "+")}:wght@400&display=swap" rel="stylesheet">'
            for font in fonts
        ])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            {font_links}
            <style>
                body {{ margin: 0; padding: 20px; }}
                #container {{ background: white; }}
            </style>
        </head>
        <body>
            <div id="container"></div>
            <script>
                function renderText(text, fontFamily, containerWidth, fontSize, paddingTop, paddingRight, paddingBottom, paddingLeft, textAlign) {{
                    const container = document.getElementById('container');
                    container.style.width = containerWidth + 'px';
                    container.style.fontFamily = '"' + fontFamily + '", sans-serif';
                    container.style.fontSize = fontSize + 'px';
                    container.style.color = 'black';
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
                await new Promise(resolve => setTimeout(resolve, 1000));
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
        
        # Escape text for JavaScript
        escaped_text = text.replace('\\', '\\\\').replace('"', '\\"').replace("'", "\\'")
        
        # Render text in container
        self.page.evaluate(f'''
            renderText("{escaped_text}", "{font_family}", {container_width}, {font_size}, {padding_top}, {padding_right}, {padding_bottom}, {padding_left}, "{text_alignment}")
        ''')
        
        # Take screenshot of container
        container = self.page.locator('#container')
        screenshot = container.screenshot()
        
        return screenshot
    
    def load_phrases(self, csv_path="phrases_10000.csv"):
        """Load phrases from CSV file"""
        df = pd.read_csv(csv_path)
        return df['phrase'].tolist()
    
    def generate_samples(self, texts=None, fonts=None, samples_per_font=500):
        """Generate font samples and save as images"""
        # Clear existing data folder
        if self.output_dir.exists():
            print(f"Clearing existing data folder: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if texts is None:
            all_phrases = self.load_phrases()
            texts = random.sample(all_phrases, min(samples_per_font, len(all_phrases)))
        
        if fonts is None:
            fonts = self.get_google_fonts(20)
        
        print(f"Generating samples for {len(fonts)} fonts...")
        
        try:
            self.start_browser(fonts)
            
            for font_idx, font_family in enumerate(fonts):
                print(f"Processing {font_family} ({font_idx+1}/{len(fonts)})")
                
                font_dir = self.output_dir / font_family.replace(' ', '_')
                font_dir.mkdir(exist_ok=True)
                
                for text_idx, text in enumerate(texts[:samples_per_font]):
                    screenshot = self.render_font_sample(text, font_family)
                    
                    filename = f"sample_{text_idx:02d}.png"
                    filepath = font_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        f.write(screenshot)
                        
                    print(f"  Saved: {filename}")
        
        finally:
            self.stop_browser()

if __name__ == "__main__":
    generator = FontDatasetGenerator()
    generator.generate_samples()
