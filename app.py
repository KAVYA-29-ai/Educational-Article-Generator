import gradio as gr
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fpdf import FPDF
import os
import datetime
import re

class ArticleGenerator:
    def __init__(self):
        """Initialize the T5 model for article generation"""
        self.model_name = "google/flan-t5-base"
        print("Loading model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        print("Model loaded successfully!")
    
    def generate_article(self, topic):
        """Generate a comprehensive article on the given topic"""
        # Create a detailed prompt for article generation
        prompt = f"""Write a comprehensive educational article about {topic}. 
        Include an introduction, main body with multiple sections, examples, and a conclusion. 
        Make it informative and well-structured for educational purposes."""
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate article
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=1024,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode the generated text
        article = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process the article to make it more structured
        article = self.format_article(article, topic)
        
        return article
    
    def format_article(self, article, topic):
        """Format the generated article with proper structure"""
        # Clean up the article
        article = article.strip()
        
        # Add title and structure
        formatted_article = f"""# {topic.title()}

## Introduction

{article}

## Key Points

This article covers the essential aspects of {topic}, providing readers with a comprehensive understanding of the subject matter.

## Further Reading

For more information on {topic}, consider exploring academic journals, reputable online resources, and educational materials from established institutions.

## Conclusion

Understanding {topic} is valuable for educational and practical purposes. This overview provides a foundation for further exploration of the subject.
"""
        
        return formatted_article

class PDFGenerator:
    def __init__(self):
        """Initialize PDF generator"""
        pass
    
    def create_pdf(self, title, content):
        """Create a PDF from the article content"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(10)
        
        # Add content
        pdf.set_font("Arial", size=11)
        
        # Split content into lines and handle special characters
        lines = content.split('\n')
        
        for line in lines:
            # Handle markdown-style headers
            if line.startswith('# '):
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, line[2:], ln=True)
                pdf.ln(3)
                pdf.set_font("Arial", size=11)
            elif line.startswith('## '):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 7, line[3:], ln=True)
                pdf.ln(2)
                pdf.set_font("Arial", size=11)
            else:
                # Handle regular text with word wrapping
                if line.strip():
                    # Encode to latin-1, replacing problematic characters
                    try:
                        line_encoded = line.encode('latin-1', 'replace').decode('latin-1')
                    except:
                        line_encoded = line.encode('ascii', 'replace').decode('ascii')
                    
                    # Split long lines
                    words = line_encoded.split(' ')
                    current_line = ""
                    
                    for word in words:
                        if len(current_line + word) < 80:
                            current_line += word + " "
                        else:
                            if current_line:
                                pdf.cell(0, 6, current_line.strip(), ln=True)
                            current_line = word + " "
                    
                    if current_line:
                        pdf.cell(0, 6, current_line.strip(), ln=True)
                else:
                    pdf.ln(3)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"article_{timestamp}.pdf"
        
        # Save PDF
        pdf.output(filename)
        
        return filename

# Initialize generators
print("Initializing Article Generator...")
article_gen = ArticleGenerator()
pdf_gen = PDFGenerator()

def generate_and_create_pdf(topic):
    """Main function to generate article and create PDF"""
    if not topic.strip():
        return "Please enter a topic to generate an article.", None
    
    try:
        # Generate article
        print(f"Generating article for topic: {topic}")
        article = article_gen.generate_article(topic)
        
        # Create PDF
        print("Creating PDF...")
        pdf_filename = pdf_gen.create_pdf(topic.title(), article)
        
        print(f"Article generated successfully! PDF saved as: {pdf_filename}")
        
        return article, pdf_filename
        
    except Exception as e:
        error_msg = f"Error generating article: {str(e)}"
        print(error_msg)
        return error_msg, None

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Educational Article Generator",
    css="""
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
    }
    """
) as demo:
    
    gr.HTML("""
    <div class="header">
        <h1>🎓 Educational Article Generator</h1>
        <p><strong>Created by [Your Name]</strong></p>
        <p>Generate comprehensive educational articles on any topic with AI-powered content creation</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label="Topic",
                placeholder="Enter any educational topic (e.g., 'Climate Change', 'Machine Learning', 'Ancient Rome')",
                lines=2
            )
            
            generate_btn = gr.Button(
                "Generate Article", 
                variant="primary"
            )
    
    with gr.Row():
        with gr.Column():
            article_output = gr.Textbox(
                label="Generated Article",
                lines=20,
                show_copy_button=True,
                placeholder="Your generated article will appear here..."
            )
    
    with gr.Row():
        with gr.Column():
            pdf_download = gr.File(
                label="Download PDF",
                visible=True
            )
    
    gr.HTML("""
    <div class="footer">
        <p>🤖 Powered by Google's FLAN-T5 model | Built with Gradio and Hugging Face Transformers</p>
        <p>Generate educational content instantly and download as PDF</p>
    </div>
    """)
    
    # Connect the button to the function
    generate_btn.click(
        fn=generate_and_create_pdf,
        inputs=[topic_input],
        outputs=[article_output, pdf_download],
        show_progress=True
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["Artificial Intelligence"],
            ["Climate Change and Global Warming"],
            ["The History of the Internet"],
            ["Quantum Physics Basics"],
            ["Renewable Energy Sources"],
            ["Ancient Egyptian Civilization"],
            ["Machine Learning Fundamentals"],
            ["Space Exploration"],
            ["DNA and Genetics"],
            ["The Industrial Revolution"]
        ],
        inputs=topic_input
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
