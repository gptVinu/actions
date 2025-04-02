import csv
import requests
import xml.etree.ElementTree as ET

def load_rss(url, filename):
    """Loads an RSS feed from a URL and saves it to a file."""
    resp = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(resp.content)
    print(f"RSS feed loaded and saved to '{filename}'.")

def parse_xml(xmlfile):
    """Parses an XML file and extracts news items."""
    try:
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        newsitems = []
        all_fields = set()

        for item in root.findall('.//item'):
            news = {}
            for child in item:
                tag = child.tag.split('}')[-1]  # Handle namespaces
                if tag not in ('thumbnail', 'content'):
                    news[tag] = child.text
                    all_fields.add(tag)
                elif tag == 'content':
                    media = child.attrib.get('url', '')
                    if media:
                        news['media'] = media
                        all_fields.add('media')
            newsitems.append(news)
        return newsitems, all_fields

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        with open(xmlfile, 'r', encoding='utf-8') as file:
            content = file.read()
            print("XML Content (first 1000 characters):")
            print(content[:1000])
        return [], set()

def save_to_csv(newsitems, fields, filename):
    """Saves news items to a CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(newsitems)
    print(f"Data saved to '{filename}'.")

def main():
    """Main function to execute the RSS feed processing."""
    rss_url = 'https://timesofindia.indiatimes.com/rssfeedstopstories.cms'
    xml_filename = 'timesOfIndia.xml'
    csv_filename = 'timesOfIndia.csv'

    load_rss(rss_url, xml_filename)
    newsitems, fields = parse_xml(xml_filename)
    if newsitems:
        save_to_csv(newsitems, fields, csv_filename)
    else:
        print("No news items to save due to XML parsing issues.")

if __name__ == "__main__":
    main()