import scrapy
import re
from bs4 import BeautifulSoup
from copy import deepcopy
states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming"
]
parties = ["Republican", "Democratic", "Other parties"]
bodies = ["U.S. House", "U.S. Senate"]


class BPSpider(scrapy.Spider):
    name = "bp"

    start_urls = [
        'https://ballotpedia.org/List_of_candidates_who_ran_in_U.S._Congress_elections,_2018',
        'https://ballotpedia.org/List_of_candidates_who_ran_in_U.S._Congress_elections,_2016'
        ]

    def parse(self, response):
        year = response.css('#firstHeading span::text').get()[-4:]
        place = {"State": "", "Body": "", "Party": "", "Year": year}
        loc_body_candidates = response.css(
            "h1 span::text,h1 ~ h2 span::text,h1 ~ h2 ~ h3 span::text, h1~h2~h3~div ol li a")
        for selector in loc_body_candidates:
            text_or_tag = selector.get()
            # print("\n", text_or_tag)
            if text_or_tag in states:
                place["State"] = text_or_tag
            elif text_or_tag in bodies:
                place["Body"] = text_or_tag
            elif text_or_tag in parties:
                place["Party"] = text_or_tag
            else:
                a_tag = BeautifulSoup(text_or_tag).find('a')
                if a_tag is not None and 'href' in a_tag.attrs:
                    # candidate_twitter = response.follow(
                    #     selector, callback=self.parse_candidate)
                    candidate_data = deepcopy(place)
                    candidate_data.update({"Name": a_tag['title']})
                    # print("basic:", candidate_data)
                    # def ind_parser(response): return self.parse_candidate(
                    #     response, candidate_data)
                    yield response.follow(selector, self.parse_candidate, meta={'cand_data': candidate_data})
        # candidate_links = response.css('h1~h2~h3~div ol li a')
        # for href in candidate_links:
        # print(href.attrib['title'])
        # response.folow(href, self.parse_candidate)

        # candidate_twitter_links = response.follow_all(
        #     candidate_links, self.parse_candidate)
        # for header in response.css('h1 span::text'):
        #     print(header.get())

    def parse_candidate(self, response):
        candidate_data = response.meta['cand_data']
        twitter_HTML_strs = response.css(
            'div p a[href*="twitter.com"]').getall()
        twitter_links = dict()
        for a_tag in twitter_HTML_strs:
            soup = BeautifulSoup(a_tag).a
            tag_text = str(soup.contents[0])
            if re.search('Twitter', tag_text):
                twitter_links[tag_text] = soup['href']
        candidate_data.update(twitter_links)
        # print("with twitter:", candidate_data)
        yield candidate_data
