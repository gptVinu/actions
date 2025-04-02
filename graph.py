import csv
import requests
import xml.etree.ElementTree as ET

def load_rss(url, filename):
    resp = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(resp.content)
    print(f"RSS feed loaded and saved to '{filename}'.")

def parse_xml(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    return [
        {child.tag: child.text for child in item if not child.tag.endswith(('thumbnail', 'content'))} |
        {'media': next((child.attrib['url'] for child in item if child.tag.endswith('content')), '')}
        for item in root.findall('.//item')
    ]

def save_to_csv(newsitems, filename):
    fields = ['guid', 'title', 'pubDate', 'description', 'link', 'media']
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(newsitems)
    print(f"Data saved to '{filename}'.")

def main():
    rss_url = 'http://feeds.bbci.co.uk/news/rss.xml'
    xml_filename = 'topnewsfeed.xml'
    csv_filename = 'topnews.csv'
    load_rss(rss_url, xml_filename)
    save_to_csv(parse_xml(xml_filename), csv_filename)

if __name__ == "__main__":
    main()