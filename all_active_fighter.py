from bs4 import BeautifulSoup
import lxml
import requests
import json
import re
import sys

class ActiveFighterNameExtractor:
    def __init__(self):
        pass


    def split_name(self, name):
        cleaned_name = " ".join(name.split())

        parts = cleaned_name.split()

        if len(parts) == 1:
            return parts[0], "", ""
        elif len(parts) == 2:
            return parts[0], "", parts[1]
        else:
            return parts[0], " ".join(parts[1:-1]), parts[-1]
            
    def extractName(self, verbose=False):
        url = 'https://www.ufc.com/athletes/all'
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        result = []
        seen = set()
        page = 0
        while True:
            if verbose:
                print(f"Fetching page {page}...")
            response = requests.get(
                url, params={"page": page}, headers=headers, timeout=20
            )
            soup = BeautifulSoup(response.text, "html.parser")
            fighters = soup.find_all("div", class_="c-listing-athlete-flipcard white")
            # The listing is finite; an empty page means we've reached the end.
            if not fighters:
                break

            new_on_page = 0
            for fighter in fighters:
                fighter_name = fighter.find('span', class_="c-listing-athlete__name")
                if not fighter_name:
                    continue
                name = " ".join(fighter_name.get_text().split())
                if not name or name in seen:
                    continue
                seen.add(name)
                firstname, middlename, lastname = self.split_name(name)
                result.append([firstname, middlename, lastname])
                new_on_page += 1

            # Safety net: if a full page adds nothing new, stop rather than loop.
            if new_on_page == 0:
                break
            page += 1

        if verbose:
            print(f"{len(result)} fighters found")
        return result


if __name__ == '__main__':
    print(ActiveFighterNameExtractor().extractName(verbose=True))