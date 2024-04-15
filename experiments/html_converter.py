from nbconvert import HTMLExporter, PDFExporter
import nbformat

# Load the notebook
input_nb_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/PDBBind/analysis.ipynb'
output_html_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/PDBBind/analysis.html'
output_pdf_path = '/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/COMPASS/experiments/PDBBind/analysis.pdf'

with open(input_nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Convert to HTML
html_exporter = HTMLExporter()
html_exporter.template_name = 'classic'
(body, resources) = html_exporter.from_notebook_node(nb)

with open(output_html_path, 'w', encoding='utf-8') as f:
    f.write(body)

# Convert to PDF
'''pdf_exporter = PDFExporter()
pdf_exporter.template_name = 'classic'
pdf_body, pdf_resources = pdf_exporter.from_notebook_node(nb)

with open(output_pdf_path, 'wb') as f:
    f.write(pdf_body)'''

output_html_path
