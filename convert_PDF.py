import os
import markdown
import pdfkit


def convert_markdown_to_pdf(input_md_path, output_pdf_path):
    """
    Convert a Markdown file to PDF and resolve issues related to Chinese garbled text
    and malformed table formatting.

    Parameters:
    - input_md_path: str, path to the Markdown file
    - output_pdf_path: str, path to the generated PDF file or output directory
    """

    # 1. Check and fix the output path
    if os.path.isdir(output_pdf_path) or not output_pdf_path.endswith('.pdf'):
        # If the output path is a directory, generate the PDF filename
        filename = os.path.splitext(os.path.basename(input_md_path))[0] + '.pdf'
        if os.path.isdir(output_pdf_path):
            output_pdf_path = os.path.join(output_pdf_path, filename)
        else:
            output_pdf_path = output_pdf_path + '\\' + filename

    print(f"Output PDF path: {output_pdf_path}")

    # 2. Check whether wkhtmltopdf exists
    wkhtmltopdf_path = r'wkhtmltopdf.exe'
    if not os.path.exists(wkhtmltopdf_path):
        # Try other common installation paths
        alternative_paths = [
            r'wkhtmltopdf.exe',
            r'wkhtmltopdf.exe'
        ]

        found = False
        for path in alternative_paths:
            if os.path.exists(path):
                wkhtmltopdf_path = path
                found = True
                break

        if not found:
            raise FileNotFoundError(f"wkhtmltopdf.exe was not found. Please check the installation path.")

    print(f"Using wkhtmltopdf: {wkhtmltopdf_path}")

    # 3. Configure pdfkit
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    try:
        # 4. Read the Markdown file
        with open(input_md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 5. Enable the tables extension to fix table display issues
        html_content = markdown.markdown(md_content, extensions=['tables', 'toc'])

        # 6. Improved CSS styles
        html_with_css = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Well Logging Interpretation Report</title>
            <style>
                body {{
                    font-family: "Microsoft YaHei", "SimSun", "Arial", sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }}

                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }}

                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #bdc3c7;
                    padding-bottom: 5px;
                    margin-top: 25px;
                }}

                h3 {{
                    color: #7f8c8d;
                    margin-top: 20px;
                }}

                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 15px auto;
                }}

                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                    font-size: 12px;
                }}

                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px 12px;
                    text-align: left;
                }}

                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                    text-align: center;
                }}

                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}

                td[style*="text-align: center"] {{
                    text-align: center !important;
                }}

                code {{
                    background-color: #f4f4f4;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: "Consolas", "Monaco", monospace;
                }}

                pre {{
                    background-color: #f8f8f8;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}

                ul, ol {{
                    margin-left: 20px;
                }}

                li {{
                    margin-bottom: 5px;
                }}

                p {{
                    margin-bottom: 12px;
                    text-align: justify;
                }}

                /* Print optimization */
                @media print {{
                    body {{
                        font-size: 12px;
                    }}

                    h1 {{
                        page-break-after: avoid;
                    }}

                    h2, h3 {{
                        page-break-after: avoid;
                    }}

                    table {{
                        page-break-inside: avoid;
                    }}

                    img {{
                        page-break-inside: avoid;
                    }}
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # 7. Save the temporary HTML file
        temp_html_path = os.path.join(os.path.dirname(output_pdf_path), 'temp_output.html')
        with open(temp_html_path, 'w', encoding='utf-8') as f:
            f.write(html_with_css)

        print("Temporary HTML file has been created.")

        # 8. Improved PDF options
        options = {
            'encoding': 'UTF-8',
            'page-size': 'A4',
            'margin-top': '15mm',
            'margin-right': '15mm',
            'margin-bottom': '15mm',
            'margin-left': '15mm',
            'enable-local-file-access': None,  # Use None instead of an empty string
            'print-media-type': None,
            'no-outline': None,
            'disable-smart-shrinking': None,
            'minimum-font-size': '10',
            'zoom': '1.0'
        }

        # 9. Convert to PDF
        print("Starting PDF conversion...")
        pdfkit.from_file(
            temp_html_path,
            output_pdf_path,
            configuration=config,
            options=options
        )

        print(f"✓ PDF conversion completed successfully: {output_pdf_path}")

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        raise

    finally:
        # 10. Clean up temporary files
        temp_html_path = os.path.join(os.path.dirname(output_pdf_path), 'temp_output.html')
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)
            print("Temporary file has been cleaned up.")


def main():
    """
    Main function.
    """

    input_md_path = r".\final_enhanced_well_logging_report_with_summary.md"
    output_pdf_path = r".\v2"

    # Check whether the input file exists
    if not os.path.exists(input_md_path):
        print(f"✗ Input file does not exist: {input_md_path}")
        return

    # Ensure that the output directory exists
    output_dir = output_pdf_path if os.path.isdir(output_pdf_path) else os.path.dirname(output_pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        convert_markdown_to_pdf(input_md_path, output_pdf_path)
    except Exception as e:
        print(f"Program execution failed: {e}")
        print("\nPossible solutions:")
        print("1. Check whether wkhtmltopdf is installed correctly.")
        print("2. Confirm whether the file paths are correct.")
        print("3. Check whether there is enough disk space.")
        print("4. Try running the program with administrator privileges.")


if __name__ == "__main__":
    main()