# Trading을 위한 대체 데이터

대체 데이터 분류

1. individuals
2. business processes
3. sensors produce

## Content

1. [The Alternative Data Revolution](#the-alternative-data-revolution)
    * [Resources](#resources)
2. [Sources of alternative data](#sources-of-alternative-data)
3. [Criteria for evaluating alternative datasets](#criteria-for-evaluating-alternative-datasets)
    * [Resources](#resources-2)
4. [The Market for Alternative Data](#the-market-for-alternative-data)
5. [Working with Alternative Data](#working-with-alternative-data)
    * [Code Example: Open Table Web Scraping](#code-example-open-table-web-scraping)
    * [Code Example: SeekingAlpha Earnings Transcripts](#code-example-seekingalpha-earnings-transcripts)
    * [Python Libraries & Documentation](#python-libraries--documentation)

## 대체 데이터 혁신

기존에 사용할 수 없던 정보를 사용하거나, 더 빨리 데이터화 할 수 있게됨. 아래는 그 예시

- 인플레이션 측정 : 대표상품과 서비스의 온라인 가격 데이터 이용
- 회사/산업/경제에 대한 예측 : 매장 방문 또는 구매 횟수
- 광산, 석유 굴착, 농업 생산량 : 위성 데이터 이용

### Resources

- [The Digital Universe in 2020](https://www.emc.com/collateral/analyst-reports/idc-the-digital-universe-in-2020.pdf)
- [Big data: The next frontier for innovation, competition, and productivity](https://www.mckinsey.com/business-functions/digital-mckinsey/our-insights/big-data-the-next-frontier-for-innovation), McKinsey 2011
- [McKinsey on Artificial Intelligence](https://www.mckinsey.com/featured-insights/artificial-intelligence)

## 대체 데이터 출처

데이터 출처에 의한 분류
- Individuals : SNS의 posting, 물건 리뷰, 검색엔진 이용 내역
- Businesses : commercial transactions(credit card payments), 공급망
- Sensors : 이미지 데이터(위성, 보안 카메라), security cameras, cell phone towers 위치

대표 예시 : Baltic Dry Index (BDI), 화물 supply/demand 예측 가능

## 대체 데이터 평가 기준

알파 생성에 도움줘야한다 (그 자체로든, 다른 전략의 신호로서든)

### Resources

- [Big Data and AI Strategies](http://valuesimplex.com/articles/JPM.pdf), Kolanovic, M. and Krishnamachari, R., JP Morgan, May 2017

## 대체 데이터 시장

계속 커지고 있다.

 - [Alternative Data](https://alternativedata.org/)

## 대체 데이터 사용 연습

스크래핑 해보자

- [Quantifying Trading Behavior in Financial Markets Using Google Trends](https://www.nature.com/articles/srep01684), Preis, Moat and Stanley, Nature, 2013
- [Quantifying StockTwits semantic terms’ trading behavior in financial markets: An effective application of decision tree algorithms](https://www.sciencedirect.com/science/article/pii/S0957417415005473), Al Nasseri et al, Expert Systems with Applications, 2015

### Code Example: Open Table Web Scraping

This subfolder [01_opentable](01_opentable) contains the script [opentable_selenium](01_opentable/opentable_selenium.py) to scrape OpenTable data using Scrapy and Selenium.

- [How to View the Source Code of a Web Page in Every Browser](https://www.lifewire.com/view-web-source-code-4151702)

### Code Example: SeekingAlpha Earnings Transcripts

> Update: 막힘

The subfolder [02_earnings_calls](02_earnings_calls) contains the script [sa_selenium](02_earnings_calls/sa_selenium.py) to scrape earnings call transcripts from the [SeekingAlpha](www.seekingalpha.com) website.

## Python Libraries & Documentation
- requests [docs](http://docs.python-requests.org/en/master/)
- beautifulsoup [docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/﻿)
- Selenium [docs](https://www.seleniumhq.org/﻿)
- Scrapy [docs](https://scapy.readthedocs.io/en/latest/)

