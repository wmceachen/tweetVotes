import scrapy

class BPSpider(scrapy.Spider):
    name = "twitter"
    start_urls = ["https://twitter.com/Palmer4Alabama"]
    def parse(self, response):
        