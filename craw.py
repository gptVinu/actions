from html.parser import HTMLParser
from urllib.request import urlopen
from urllib import parse
import json

class LinkParser(HTMLParser):
    def __init__(self, base_url):
        super().__init__()
        self.links = []
        self.base_url = base_url

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for key, value in attrs:
                if key == "href":
                    self.links.append(parse.urljoin(self.base_url, value))

    def get_links(self, url):
        try:
            response = urlopen(url)
            if "text/html" in response.getheader("Content-Type"):
                html_content = response.read().decode("utf-8")
                self.feed(html_content)
                return html_content, self.links
            return "", []
        except Exception:
            return "", []

def crawl(url, word):
    visited_urls = set()
    found_urls = []
    not_found_urls = []
    total_word_count = 0
    parser = LinkParser(url)
    links_to_crawl = [url]
    number_visited = 0

    while links_to_crawl:
        current_url = links_to_crawl.pop(0)
        if current_url not in visited_urls:
            visited_urls.add(current_url)
            number_visited += 1
            print(f"{str(number_visited).zfill(2)} --> Scanning URL {current_url}")
            data, links = parser.get_links(current_url)
            word_count = data.lower().count(word.lower())
            total_word_count += word_count

            if word_count > 0:
                found_urls.append(current_url)
                print(f"--> The word '{word}' was found {word_count} times at {current_url}")
            else:
                not_found_urls.append(current_url)
                print(f"Word '{word}' not found at {current_url}")

            links_to_crawl.extend(links)

    print(f"\nFinished, crawled {number_visited} pages.")
    print(f"Total matches: {len(found_urls)}")
    print(f"Total non-matches: {len(not_found_urls)}")
    print(f"Total occurrences of '{word}': {total_word_count}")
    print(f"Total occurrences of '{word}': {total_word_count}")
    print("URLs where the word was found:", json.dumps(found_urls, separators=(',', ':')))
    print("URLs where the word was NOT found:", json.dumps(not_found_urls, separators=(',', ':')))

crawl("https://www.facebook.com", "login")