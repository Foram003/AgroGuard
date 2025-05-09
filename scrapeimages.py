import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests
import os


chrome_options = Options()
chrome_options.add_argument("--headless")  
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Set up WebDriver
driver = webdriver.Chrome(options=chrome_options)

# Function to search Google and scrape images
def scrape_images(query, num_images=10):
    # Create a folder to store images
    if not os.path.exists(query):
        os.mkdir(query)
    
    # Format the query to a Google image search URL
    search_url = f"https://www.google.com/search?hl=en&tbm=isch&q={query}"
    
    # Open the search URL in the browser
    driver.get(search_url)
    
    # Scroll the page to load images (optional)
    for _ in range(2):  # Adjust number of scrolls if needed
        driver.execute_script("window.scrollBy(0,1000);")
        time.sleep(2)
    
    # Get image elements on the page
    images = driver.find_elements(By.TAG_NAME, "img")
    
    image_urls = []
    for img in images:
        src = img.get_attribute("src")
        if src and src.startswith("http"):
            image_urls.append(src)
    
    # Download the images
    count = 0
    for url in image_urls:
        if count >= num_images:
            break
        try:
            img_data = requests.get(url).content
            with open(f"{query}/image_{count + 1}.jpg", 'wb') as f:
                f.write(img_data)
            print(f"Downloaded {query} image {count + 1}")
            count += 1
        except Exception as e:
            print(f"Failed to download image {count + 1}: {e}")

# Example: Scrape images for a specific pest
scrape_images('aphid pest', num_images=10)

# Close the driver
driver.quit()
