"""
Educational Article Generator using Google Gemini (gemini-1.5-flash)
Requirements:
    pip install gradio google-generativeai python-dotenv fpdf2

Usage:
    1. Create a .env file in the same directory with:
        GOOGLE_API_KEY=your_api_key_here
    2. Run: python app.py
    3. Open http://0.0.0.0:7860 locally
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import gradio as gr
from fpdf import FPDF
import textwrap
import uuid

# -------------------------
# Load API key from .env
# -------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found in environment. Create a .env file with GOOGLE_API_KEY=..."
    )

# Configure the google generative ai client
genai.configure(api_key=GOOGLE_API_KEY)

# -------------------------
# Helper: Build prompt
# -------------------------
def build_prompt(topic: str) -> str:
    """
    Construct a detailed prompt that instructs Gemini 1.5 Flash to
    produce an educational article with required sections.
    """
    prompt = f"""
You are an expert educator and technical writer. Produce a clear, well-structured, up-to-date educational article
on the following topic. The user wants a comprehensive but readable article suitable for learners and practitioners.

Topic: "{topic}"

Requirements:
- The article must include clearly labeled sections in this exact order:
  1) Title
  2) Introduction
  3) Key Concepts
  4) Practical Examples
  5) Further Reading
  6) Summary
- Use headings for each section (e.g., "Title:", "Introduction:", "Key Concepts:", etc.).
- Keep writing concise but informative. Use bullet points in Key Concepts and Practical Examples when helpful.
- Include at least 2 practical, concrete examples or mini-tutorial steps in Practical Examples.
- For Further Reading, include 4 suggestions (books, papers, or authoritative websites) with a 1-sentence reason for each.
- Aim for a total length ~600-1200 words. Use current best-practices where applicable.
- Don't include unrelated content or marketing fluff. No code other than short example snippets if relevant.

Return the article as plain text.
"""
    return prompt

# -------------------------
# Generate article via Gemini
# -------------------------
def generate_article_from_gemini(topic: str, temperature: float = 0.2, max_output_tokens: int = 1024):
    """
    Calls Gemini model to generate the article text.
    Returns: article_text (str)
    """
    prompt = build_prompt(topic)

    # Use generate() for text output
    try:
        response = genai.generate(
            model="gemini-1.5-flash",
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        # The google.generativeai library returns a Response-like object with .text
        article_text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
        # Some versions return structured fields; attempt several fallbacks:
        if isinstance(response, dict) and "candidates" in response and len(response["candidates"]) > 0:
            article_text = response["candidates"][0].get("content", response["candidates"][0].get("output", article_text))
        # final fallback cast
        article_text = str(article_text).strip()
        return article_text
    except Exception as e:
        # Provide an informative error message for the UI
        return f"ERROR: Failed to generate article from Gemini: {e}"

# -------------------------
# PDF generation
# -------------------------
class ArticlePDF(FPDF):
    def header(self):
        # Override header if desired (keeps simple)
        pass

    def footer(self):
        # Page number in footer
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def create_pdf_from_article(article_text: str, topic: str, out_dir: str = "outputs") -> str:
    """
    Generate a nicely formatted PDF with title and headings from the article text.
    Returns the filepath of the saved PDF.
    """
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "-", "_")).strip()[:40]
    filename = f"article_{safe_topic or 'topic'}_{timestamp}.pdf"
    filepath = os.path.join(out_dir, filename)

    # Parse article_text - we will keep the section headings as-is and render them
    lines = article_text.splitlines()
    wrapped_lines = []
    # We'll wrap long lines to reasonable width when putting into PDF
    wrapper = textwrap.TextWrapper(width=90, replace_whitespace=False)

    for line in lines:
        if not line.strip():
            wrapped_lines.append("")  # blank line
        else:
            # If line looks like a heading (ends with ':' or all-caps first words), keep as-is
            stripped = line.strip()
            if stripped.endswith(":") or (stripped == stripped.upper() and len(stripped.split()) <= 6):
                wrapped_lines.append(stripped)
            else:
                wrapped_lines.extend(wrapper.wrap(stripped))

    pdf = ArticlePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Title page header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(40, 40, 80)
    pdf.multi_cell(0, 10, topic.strip() or "Educational Article", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 110)
    generated_on = datetime.now().strftime("%B %d, %Y %H:%M:%S")
    pdf.multi_cell(0, 6, f"Generated on: {generated_on}", align="C")
    pdf.ln(8)

    # Now render content
    pdf.set_text_color(10, 10, 30)
    for line in wrapped_lines:
        if not line.strip():
            pdf.ln(4)
            continue

        # Detect headings by trailing ':' or common heading tokens
        if line.endswith(":") or line.lower().startswith(("title:", "introduction:", "key concepts:", "practical examples:", "further reading:", "summary:")):
            # Heading style
            heading = line.rstrip(":").strip()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(40, 40, 80)
            pdf.multi_cell(0, 8, heading)
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(10, 10, 30)
        elif line.startswith("- ") or line.startswith("• ") or line.startswith("* "):
            # Bullet point
            pdf.set_font("Helvetica", "", 11)
            # Small indent bullet
            pdf.cell(8)
            pdf.multi_cell(0, 6, line)
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, line)

    # Save PDF
    pdf.output(filepath)
    return filepath

# -------------------------
# Gradio UI callbacks
# -------------------------
def on_generate(topic: str):
    """
    Callback for Generate button. Produces article text and PDF, returns (article_text, pdf_file_path)
    """
    topic = (topic or "").strip()
    if not topic:
        return "Please enter a topic to generate an article.", None

    # Give the user quick acknowledgement
    # (We perform the actual generation synchronously)
    article_text = generate_article_from_gemini(topic)

    # If generation returned an ERROR string, do not attempt to create PDF
    if article_text.startswith("ERROR:"):
        return article_text, None

    # Ensure the generated article includes required sections (best-effort):
    # If the model didn't include section labels, we add them heuristically.
    # But we primarily trust the model because prompt enforces sections.
    required_sections = ["Title:", "Introduction:", "Key Concepts:", "Practical Examples:", "Further Reading:", "Summary:"]
    if not any(s in article_text for s in required_sections):
        # attempt to structure using simple heuristics
        # We'll create a minimal structured wrapper
        generated_title = f"{topic.strip().title()}"
        wrapped = [
            f"Title: {generated_title}",
            "",
            "Introduction:",
            article_text,
            "",
            "Key Concepts:",
            "- (See above — split into bullets for clarity)",
            "",
            "Practical Examples:",
            "- Example 1: (Please refine)",
            "- Example 2: (Please refine)",
            "",
            "Further Reading:",
            "- (Search current resources on the topic)",
            "",
            "Summary:",
            "- (Brief summary)"
        ]
        article_text = "\n".join(wrapped)

    # Create PDF
    try:
        pdf_path = create_pdf_from_article(article_text, topic)
    except Exception as e:
        pdf_path = None
        article_text = f"ERROR: Article generated but failed to create PDF: {e}\n\nArticle content below:\n\n{article_text}"

    return article_text, pdf_path

# -------------------------
# Gradio UI
# -------------------------
css = """
/* Simple purple-blue theme with modern feel */
:root {
  --bg1: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
  --card-bg: rgba(255,255,255,0.04);
  --accent: #9b59ff;
  --accent-2: #6dd5ed;
}

/* Apply background gradient */
body, .root {
  background: var(--bg1) !important;
}

/* Card style for blocks */
.gradio-container .card {
  background: rgba(255,255,255,0.06);
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.24);
}

/* Headings */
h1, h2, .title {
  color: white !important;
}

/* Buttons */
.gr-button {
  background: linear-gradient(90deg, #7b61ff, #4fc3ff) !important;
  color: white !important;
  border: none !important;
  box-shadow: 0 6px 12px rgba(79,195,255,0.18);
}

/* Text areas */
textarea, input[type="text"] {
  background: rgba(255,255,255,0.02) !important;
  color: #f1f1f1 !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* File download look */
.gr-file {
  background: rgba(255,255,255,0.02) !important;
  border-radius: 8px;
}
"""

with gr.Blocks(css=css, theme=None) as demo:
    gr.HTML("<h1 style='text-align:center; margin-bottom: -6px;'>Gemini Educational Article Generator</h1>")
    gr.Markdown(
        """
        <div style="text-align:center; color: #e8f0ff; margin-bottom: 20px;">
        Enter a topic and press **Generate** — Gemini 1.5 Flash will create a structured educational article that you can download as a PDF.
        </div>
        """,
        elem_id="subtitle"
    )

    with gr.Row(elem_id="controls_row"):
        with gr.Column(scale=2):
            topic_input = gr.Textbox(
                label="Topic",
                placeholder="e.g., 'Quantum Computing basics' or 'Introduction to REST APIs'",
                interactive=True,
                lines=1
            )
        with gr.Column(scale=1):
            generate_btn = gr.Button("Generate", variant="primary")

    # Output area
    article_output = gr.Textbox(
        label="Generated Article",
        placeholder="Generated article will appear here...",
        lines=25,
        interactive=False
    )

    # File component to show downloadable PDF (initially empty)
    pdf_file = gr.File(label="Download PDF", interactive=False)

    # Connect button to generation function
    generate_btn.click(
        fn=on_generate,
        inputs=[topic_input],
        outputs=[article_output, pdf_file],
        _js=None,
    )

    # Footer note
    gr.Markdown(
        """
        <small style="color: #dbe9ff;">
        Uses Google Gemini (gemini-1.5-flash). Make sure your GOOGLE_API_KEY is set in .env.
        </small>
        """
    )

# Launch the app as requested
if __name__ == "__main__":
    # Security: ensure not exposing share link
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
