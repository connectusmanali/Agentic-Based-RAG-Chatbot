import requests
import os
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup

PDF_LINKS = [
    "https://bylaws.vancouver.ca/zoning/zoning-by-law-consolidated.pdf",
    "https://council.vancouver.ca/20171128/documents/rr1appendixa.pdf",
    "https://vancouver.ca/files/cov/vancouver-plan.pdf",
    "https://bylaws.vancouver.ca/Bulletin/bulletin-green-buildings-policy-for-rezoning.pdf"
]

BROCHURE_PAGE = "https://vancouver.ca/home-property-development/development-and-building-services-centre-brochures.aspx"
BASE_FOLDER = "City_of_Vancouver_PDFs"
os.makedirs(BASE_FOLDER, exist_ok=True)

def download_pdf(url, folder):
    file_name = url.split("/")[-1]
    file_path = os.path.join(folder, file_name)
    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 200 and 'application/pdf' in resp.headers.get("Content-Type", ""):
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Skipped (Not PDF or error): {url}")
        resp.close()
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    time.sleep(1)

# Download static PDF links
for pdf_url in PDF_LINKS:
    download_pdf(pdf_url, BASE_FOLDER)

for pdf_url in PDF_LINKS:
    download_pdf(pdf_url, BASE_FOLDER)
# Scrape additional brochures from the city page
try:
    resp = requests.get(BROCHURE_PAGE, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_links = [a['href'] for a in soup.find_all("a", href=True) if a['href'].lower().endswith(".pdf")]
    for link in pdf_links:
        full_url = urljoin(BROCHURE_PAGE, link)
        download_pdf(full_url, BASE_FOLDER)
except Exception as e:
    print(f"Failed to scrape brochures: {e}")

print("\nPDF download complete.")