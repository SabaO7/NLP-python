## working with pdf file in python

import PyPDF2

# Open the PDF file
with open("capacity_building.pdf", mode='rb') as myfile:
    pdf_reader = PyPDF2.PdfReader(myfile) 

    # Get the number of pages
    num_pages = len(pdf_reader.pages)
    print(num_pages)

# Access a specific page
    # Example: Accessing the first page (page numbering starts from 0)
    if num_pages > 0:
        first_page = pdf_reader.pages[0]
        # Perform operations with first_page, like extracting text
        print(first_page.extract_text())



##adding a page to another pdf file

# Open the first PDF file and extract the first page
with open("capacity_building.pdf", mode='rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)
    page_one = pdf_reader.pages[0]

    # Create a PdfWriter object
    pdf_writer = PyPDF2.PdfWriter()

    # Add the first page from the first PDF to the writer
    pdf_writer.add_page(page_one)

# Open the second PDF file and add all its pages
with open("article.pdf", mode='rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)

    # Loop through all the pages in the second PDF and add them
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_writer.add_page(page)

# Write the combined pages to a new PDF file
with open("new_pdf.pdf", mode='wb') as f_output:
    pdf_writer.write(f_output)

##Checking the number of pages in the new PDF file

# Open the newly created PDF file
with open("new_pdf.pdf", mode='rb') as f:
    pdf_reader = PyPDF2.PdfReader(f)
    
    # Get the number of pages
    num_pages = len(pdf_reader.pages)
    print(f"The new PDF file contains {num_pages} pages.")
