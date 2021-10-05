# 트레이딩을 위한 머신러닝

얻어갈 것 : 트레이딩용 end-to-end 머신러닝 워크플로우

<p align="center">
<img src="https://i.imgur.com/kcgItgp.png" width="75%">
</p>


**알고리즘 트레이딩** : 알고리즘(목표를 달성하기 위한 규칙이나 절차)을 사용해 실제 주문을 집행. 끝에서 부터 보면,
 - 목표 예시. Active investment management : 알파(벤치마크를 초과하는 수익률) 달성
 - Evaluation 예시. 위험조정지표 : 샤프비율, 트레이너지수, 젠센알파, 정보비율(IR=IC*BR^0.5, 정보 계수 IC=(2×Proportion Correct)−1)] > 전략에 대한 평가 및 전략별 비교 가능 (ex. 종목 선택 전략 / 마켓 타이밍 전략)
 - 방법 예시. [대체데이터](#ml-and-alternative-data) 등 이용한 정보의 우위 또는 고도화된 데이터 분석

책 기준으로 보면, 각자 자신의 목표를 기준으로 평가방법(5, 8장) 세우고 데이터(2, 3장) 이용해서 피쳐(4, 13, 20장, 3부[자연어], 4부[이미지]) 뽑아내고 자산별 예측 모델(7, 9~12, 19장) 결과값 넣어서 포트폴리오 최적화(5장) 된 걸 시뮬레이션 등으로 테스트 해본 후(CV, 21장, GAN) 잘 주문(22장, 강화학습)


1. 투자자별 위험선호도(KYC)에 따른 매력적인 목표 수익률을 제공하고자
2. 시장 거래를 관찰
3. 포트폴리오 매니저의 매매행위
 하는 포트폴리오를 보유하기 위한 매수/매도 주문


## Content

1. [The rise of ML in the investment industry](#the-rise-of-ml-in-the-investment-industry)
    * [From electronic to high-frequency trading](#from-electronic-to-high-frequency-trading)
    * [Factor investing and smart beta funds](#factor-investing-and-smart-beta-funds)
    * [Algorithmic pioneers outperform humans](#algorithmic-pioneers-outperform-humans)
        - [ML driven funds attract $1 trillion AUM](#ml-driven-funds-attract-1-trillion-aum)
        - [The emergence of quantamental funds](#the-emergence-of-quantamental-funds)
    * [ML and alternative data](#ml-and-alternative-data)
2. [Designing and executing an ML-driven strategy](#designing-and-executing-an-ml-driven-strategy)
    * [Sourcing and managing data](#sourcing-and-managing-data)
    * [From alpha factor research to portfolio management](#from-alpha-factor-research-to-portfolio-management)
    * [Strategy backtesting](#strategy-backtesting)
3. [ML for trading in practice: strategies and use cases](#ml-for-trading-in-practice-strategies-and-use-cases)
    * [The evolution of algorithmic strategies](#the-evolution-of-algorithmic-strategies)
    * [Use cases of ML for trading](#use-cases-of-ml-for-trading)
        - [Data mining for feature extraction and insights](#data-mining-for-feature-extraction-and-insights)
        - [Supervised learning for alpha factor creation and aggregation](#supervised-learning-for-alpha-factor-creation-and-aggregation)
        - [Asset allocation](#asset-allocation)
        - [Testing trade ideas](#testing-trade-ideas)
        - [Reinforcement learning](#reinforcement-learning)
4. [Resources & References](#resources--references)
    * [Academic Research](#academic-research)
    * [Industry News](#industry-news)
    * [Books](#books)
        - [Machine Learning](#machine-learning)
    * [Courses](#courses)
    * [ML Competitions & Trading](#ml-competitions--trading)
    * [Python Libraries](#python-libraries)


## 썰풀기 시작 : 결국 트레이딩 워크플로우 각각에서 머신러닝이 사람보다 뛰어났다는 이야기

## 투자업계에서 머신러닝의 부상

왜?
..액티브 성과 안좋음..
1. 전자거래 확산, 시장구조 변화
2. risk-factor exposure 측면의 투자전략 개발
3. 컴퓨팅 파워, 데이터 측면 발전
4. 인간보다 성과좋음

### 전자거래에서 고빈도 거래까지

요약 : Execution 단에서 잘했다 (22장)
처음에는 시장충격 제한위해 시간에 걸쳐 주문 분산시키는 주문실행 목적으로 사용 - 이후 매수 쪽으로 진행 - 단기 가격 및 거래량 예측, 거래비용 및 유동성까지 고려
이후 고빈도 거래에도 사용(마이크로초 단위의 거래 ~ 패시브 : 차익 거래 / 액티브 : Momentum Ignition(다른 알고리즘 움직이게 만드는거), Liquidity Detection)

- [Dark Pool Trading & Finance](https://www.cfainstitute.org/en/advocacy/issues/dark-pools), CFA Institute
- [Dark Pools in Equity Trading: Policy Concerns and Recent Developments](https://crsreports.congress.gov/product/pdf/R/R43739), Congressional Research Service, 2014
- [High Frequency Trading: Overview of Recent Developments](https://fas.org/sgp/crs/misc/R44443.pdf), Congressional Research Service, 2016

### 팩터 투자와 스마트 투자

요약 : Feature 단에서 잘했다. (4, 13, 20장, 3부[자연어], 4부[이미지])
(마코비츠)모든 투자자는 자신의 포트폴리오를 최적화 시키려 한다 by 수익률 & 위험
-> 수익률은 불확실성과 위험의 함수 ex) 주식 : 회사의 사업위험, 채권 : 디폴트 위험
-> 리스크 요소별로 나누고 그 움직임을 예측해보자!

현대 포트폴리오 이론 : 체계적 위험, 비체계적 위험의 원천 구별 (risk 요인 구별, risk free, Unsystematic risk는 없앨 수 있다)
"시장" 포트폴리오가 efficient하다 > CAPM(Asset Pricing Model) : 특정한 자산의 가격은 그 자산의 "beta" 알면 구할 수 있다 (risk free + 시장위험) [아닌자산 있으면 돈벌 수 있다] : 주식가격 분석하는 좋은 툴
MPT + 제대로 작동하는 시장에선 아비트라지가 없다 > 다요인 모형 : APT모형[GDP, 인플레이션], 파마-프렌치 모형, 가치투자(벤자민 그래험, 워런 버핏 등), 모멘텀]

2008 금융위기 > 자산군으로만 나누는게 아니라 factor 기준으로 나누는게 필요

but! apt는 어떤 factor, 몇개의 factor가 필요한지 알기 어렵다 > 머신러닝 등장

### 알고리즘 개척자는 인간보다 우위

다이쇼, 시타델, 투시그마, 르네상스 테크놀로지, AQR

#### ML driven funds attract $1 trillion AUM

규모 늘어나고, 적용범위도 넓어지는중 -> ex. ETF 관리, 로보어드바이저 (from 아이디어 창출&리서치 to 거래 실행&위험 관리)

- [Global Algorithmic Trading Market to Surpass US$ 21,685.53 Million by 2026](https://www.bloomberg.com/press-releases/2019-02-05/global-algorithmic-trading-market-to-surpass-us-21-685-53-million-by-2026)
- [The stockmarket is now run by computers, algorithms and passive managers](https://www.economist.com/briefing/2019/10/05/the-stockmarket-is-now-run-by-computers-algorithms-and-passive-managers), Economist, Oct 5, 2019

#### The emergence of quantamental funds

systematic (or quant) : 알고리즘에만 의존
discretionary investing : 심층적 분석도 사용 (알고리즘으로 한번 걸러내고 리서치 추가 이용, 맨 플러스 머신)

### ML and alternative data

디지털 데이터 양 증가, 컴퓨팅 파워 증가, 데이터 분석 위한 머신러닝 기법 발달
전통적인 데이터 : 경제 통계, 거래 데이터, 기업 보고서
대체 데이터 : 실적관련[취업공고 감소, 임원 내부등급 평가, 해당 사이트 의류 평균가격 하락, 주차장 위성 이미지, 모바일 위치정보, 신용카드 판매 데이터], 감성분석, 웹사이트 스크래핑 등
> 데이터 커서 병렬처리 위한 하둡, 스파크 사용

- [Can We Predict the Financial Markets Based on Google's Search Queries?](https://onlinelibrary.wiley.com/doi/abs/10.1002/for.2446), Perlin, et al, 2016, Journal of Forecasting

### 크라우드 소싱 거래 알고리즘
퀀토피안은 망함..

## Designing and executing an ML-driven strategy

데이터 > 피쳐 엔지니어링 > 포트폴리오 관리 및 성과 추적(8장 백테스팅)

[Chapter 4, Alpha Factor Research](../04_alpha_factor_research) outlines a methodologically sound process to manage the risk of false discoveries that increases with the amount of data. [Chapter 5, Strategy Evaluation](../05_strategy_evaluation) provides the context for the execution and performance measurement of a trading strategy.

### Sourcing and managing data

1. Identify and evaluate market, fundamental, and alternative data sources containing alpha signals that do not decay too quickly.
2. DB 잘 짜서 빠르고 유연해야한다 (Hadoop or Spark)
3. Point-In-Time 기반한 데이터셋 만들어서 Look-ahead bias 피해야 한다

### From alpha factor research to portfolio management

알파 팩터 만들기 : 여러개 이용해서 만들수도 있음 > 차원축소, 클러스터링 등 가능 + 해야함 (단일은 이미 알파 다 빼먹었기 때문)

실행단에서는 포트폴리오 최적화 포함 (ex. 개별주식 수익률 및 변동성 예측 > 포트폴리오 구성) - 5장

### Strategy backtesting

백테스팅 거쳐서 통과하면 실제로 전략풀에 넣는다 (ex. 시뮬레이션)



## ML 적용 실제사례

### 썰풀기

1. In the 1980s and 1990s, signals often emerged from academic research and used a single or very few inputs derived from market and fundamental data. AQR, one of the largest quantitative hedge funds today, was founded in 1998 to implement such strategies at scale. These signals are now largely commoditized and available as ETF, such as basic mean-reversion strategies.
2. In the 2000s, factor-based investing proliferated based on the pioneering work by Eugene Fama and Kenneth French and others. Funds used algorithms to identify assets exposed to risk factors like value or momentum to seek arbitrage opportunities. Redemptions during the early days of the financial crisis triggered the quant quake of August 2007 that cascaded through the factor-based fund industry. These strategies are now also available as long-only smart beta funds that tilt portfolios according to a given set of risk factors.
3. The third era is driven by investments in ML capabilities and alternative data to generate profitable signals for repeatable trading strategies. Factor decay is a major challenge: the excess returns from new anomalies have been shown to drop by a quarter from discovery to publication, and by over 50 percent after publication due to competition and crowding.

Today, traders pursue a range of different objectives when using algorithms to execute rules:
- 주문집행
- HFT (차익거래)
- 행동 예측
- Asset Pricing 기반 전략

### 책에서 따라하는 실제사례

- ... 개별 신호를 전략을 통합 (메릴린치 방식) ...

#### Data mining for feature extraction and insights -> 피쳐 뽑기

- **Information theory** : 해당 피쳐 평가 통한 입력변수 추출에 활용
- **Unsupervised learning**  ex. 크래프트
    - In Chapter 13, [Unsupervised Learning: From Data-Driven Risk Factors to Hierarchical Risk Parity](../13_unsupervised_learning/README.md), we introduce clustering and dimensionality reduction to generate features from high-dimensional datasets. 
    - In Chapter 15, [Topic Modeling for Earnings Calls and Financial News](../15_topic_modeling/README.md), we apply Bayesian probability models to summarize financial text data.
    - In Chapter 20: [Autoencoders for Conditional Risk Factors](../20_autoencoders_for_conditional_risk_factors), we used deep learning to extract non-linear risk factors conditioned on asset characteristics and predict stock returns based on [Kelly et. al.](https://www.aqr.com/Insights/Research/Working-Paper/Autoencoder-Asset-Pricing-Models) (2020).
- **Model transparency**: 피쳐 중요도..

#### Supervised learning for alpha factor creation and aggregation -> 모델링 방식 : 타깃 바꿔보기(매크로, 변동성), 시계열 예측(RNN)
#### Asset allocation -> 자산배분 : 묶여있는 자산군이 아니라 특성에 따라 새로 묶고 배분해서 최적화 가능




## Resources & References

### Academic Research

- [The fundamental law of active management](http://jpm.iijournals.com/content/15/3/30), Richard C. Grinold, The Journal of Portfolio Management Spring 1989, 15 (3) 30-37
- [The relationship between return and market value of common stocks](https://www.sciencedirect.com/science/article/pii/0304405X81900180), Rolf Banz,Journal of Financial Economics, March 1981
- [The Arbitrage Pricing Theory: Some Empirical Results](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1981.tb00444.x), Marc Reinganum, Journal of Finance, 1981
- [The Relationship between Earnings' Yield, Market Value and Return for NYSE Common Stock](https://pdfs.semanticscholar.org/26ab/311756099c8f8c4e528083c9b90ff154f98e.pdf), Sanjoy Basu, Journal of Financial Economics, 1982
- [Bridging the divide in financial market forecasting: machine learners vs. financial economists](http://www.sciencedirect.com/science/article/pii/S0957417416302585), Expert Systems with Applications, 2016 
- [Financial Time Series Forecasting with Deep Learning : A Systematic Literature Review: 2005-2019](http://arxiv.org/abs/1911.13288), arXiv:1911.13288 [cs, q-fin, stat], 2019 
- [Empirical Asset Pricing via Machine Learning](https://doi.org/10.1093/rfs/hhaa009), The Review of Financial Studies, 2020 
- [The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns](http://academic.oup.com/rfs/article/30/12/4389/3091648), The Review of Financial Studies, 2017 
- [Characteristics are covariances: A unified model of risk and return](http://www.sciencedirect.com/science/article/pii/S0304405X19301151), Journal of Financial Economics, 2019 
- [Estimation and Inference of Heterogeneous Treatment Effects using Random Forests](https://doi.org/10.1080/01621459.2017.1319839), Journal of the American Statistical Association, 2018 
- [An Empirical Study of Machine Learning Algorithms for Stock Daily Trading Strategy](https://www.hindawi.com/journals/mpe/2019/7816154/), Mathematical Problems in Engineering, 2019 
- [Predicting stock market index using fusion of machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414006551), Expert Systems with Applications, 2015 
- [Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques](http://www.sciencedirect.com/science/article/pii/S0957417414004473), Expert Systems with Applications, 2015 
- [Deep Learning for Limit Order Books](http://arxiv.org/abs/1601.01987), arXiv:1601.01987 [q-fin], 2016 
- [Trading via Image Classification](http://arxiv.org/abs/1907.10046), arXiv:1907.10046 [cs, q-fin], 2019 
- [Algorithmic trading review](http://doi.org/10.1145/2500117), Communications of the ACM, 2013 
- [Assessing the impact of algorithmic trading on markets: A simulation approach](https://www.econstor.eu/handle/10419/43250), , 2008 
- [The Efficient Market Hypothesis and Its Critics](http://www.aeaweb.org/articles?id=10.1257/089533003321164958), Journal of Economic Perspectives, 2003 
- [The Arbitrage Pricing Theory Approach to Strategic Portfolio Planning](https://doi.org/10.2469/faj.v40.n3.14), Financial Analysts Journal, 1984 

### Industry News

- [The Rise of the Artificially Intelligent Hedge Fund](https://www.wired.com/2016/01/the-rise-of-the-artificially-intelligent-hedge-fund/#comments), Wired, 25-01-2016
- [Crowd-Sourced Quant Network Allocates Most Ever to Single Algo](https://www.bloomberg.com/news/articles/2018-08-02/crowd-sourced-quant-network-allocates-most-ever-to-single-algo), Bloomberg, 08-02-2018
- [Goldman Sachs’ lessons from the ‘quant quake’](https://www.ft.com/content/fdfd5e78-0283-11e7-aa5b-6bb07f5c8e12), Financial Times, 03-08-2017
- [Lessons from the Quant Quake resonate a decade later](https://www.ft.com/content/a7a04d4c-83ed-11e7-94e2-c5b903247afd), Financial Times, 08-18-2017
- [Smart beta funds pass $1tn in assets](https://www.ft.com/content/bb0d1830-e56b-11e7-8b99-0191e45377ec), Financial Times, 12-27-2017
- [BlackRock bets on algorithms to beat the fund managers](https://www.ft.com/content/e689a67e-2911-11e8-b27e-cc62a39d57a0), Financial Times, 03-20-2018
- [Smart beta: what’s in a name?](https://www.ft.com/content/d1bdabaa-a9f0-11e7-ab66-21cc87a2edde), Financial Times, 11-27-2017
- [Computer-driven hedge funds join industry top performers](https://www.ft.com/content/9981c870-e79a-11e6-967b-c88452263daf), Financial Times, 02-01-2017
- [Quants Rule Alpha’s Hedge Fund 100 List](https://www.institutionalinvestor.com/article/b1505pmf2v2hg3/quants-rule-alphas-hedge-fund-100-list), Institutional Investor, 06-26-2017
- [The Quants Run Wall Street Now](https://www.wsj.com/articles/the-quants-run-wall-street-now-1495389108), Wall Street Journal, 05-21-2017
- ['We Don’t Hire MBAs': The New Hedge Fund Winners Will Crunch The Better Data Sets](https://www.cbinsights.com/research/algorithmic-hedge-fund-trading-winners/), cbinsights, 06-28-2018
- [Artificial Intelligence: Fusing Technology and Human Judgment?](https://blogs.cfainstitute.org/investor/2017/09/25/artificial-intelligence-fusing-technology-and-human-judgment/), CFA Institute, 09-25-2017
- [The Hot New Hedge Fund Flavor Is 'Quantamental'](https://www.bloomberg.com/news/articles/2017-08-25/the-hot-new-hedge-fund-flavor-is-quantamental-quicktake-q-a), Bloomberg, 08-25-2017
- [Robots Are Eating Money Managers’ Lunch](https://www.bloomberg.com/news/articles/2017-06-20/robots-are-eating-money-managers-lunch), Bloomberg, 06-20-2017
- [Rise of Robots: Inside the World's Fastest Growing Hedge Funds](https://www.bloomberg.com/news/articles/2017-06-20/rise-of-robots-inside-the-world-s-fastest-growing-hedge-funds), Bloomberg, 06-20-2017
- [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
- [BlackRock bulks up research into artificial intelligence](https://www.ft.com/content/4f5720ce-1552-11e8-9376-4a6390addb44), Financial Times, 02-19-2018
- [AQR to explore use of ‘big data’ despite past doubts](https://www.ft.com/content/3a8f69f2-df34-11e7-a8a4-0a1e63a52f9c), Financial Times, 12-12-2017
- [Two Sigma rapidly rises to top of quant hedge fund world](https://www.ft.com/content/dcf8077c-b823-11e7-9bfb-4a9c83ffa852), Financial Times, 10-24-2017
- [When Silicon Valley came to Wall Street](https://www.ft.com/content/ba5dc7ca-b3ef-11e7-aa26-bb002965bce8), Financial Times, 10-28-2017
- [Artificial intelligence (AI) in finance - six warnings from a central banker](https://www.bundesbank.de/en/press/speeches/artificial-intelligence--ai--in-finance--six-warnings-from-a-central-banker-711602), Deutsche Bundesbank, 02-27-2018
- [Fintech: Search for a super-algo](https://www.ft.com/content/5eb91614-bee5-11e5-846f-79b0e3d20eaf), Financial Times, 01-20-2016
- [Barron’s Top 100 Hedge Funds](https://www.barrons.com/articles/top-100-hedge-funds-1524873705)
- [How high-frequency trading hit a speed bump](https://www.ft.com/content/d81f96ea-d43c-11e7-a303-9060cb1e5f44), FT, 01-01-2018

### Books

- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086), Marcos Lopez de Prado, 2018
- [Quantresearch](http://www.quantresearch.info/index.html) by Marcos López de Prado
- [Quantitative Trading](http://epchan.blogspot.com/), Ernest Chan
- [Machine Learning in Finance](https://www.springer.com/gp/book/9783030410674), Dixon, Matthew F., Halperin, Igor, Bilokon, Paul, Springer, 2020

#### Machine Learning

- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html), Tom Mitchell, McGraw Hill, 1997
- [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), Gareth James et al.
    - Excellent reference for essential machine learning concepts, available free online
- [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf), Barber, D., Cambridge University Press, 2012 (updated version available on author's website)

### Courses

- [Algorithmic Trading](http://personal.stevens.edu/~syang14/fe670.htm), Prof. Steve Yang, Stevens Institute of Technology
- [Machine Learning](https://www.coursera.org/learn/machine-learning), Andrew Ng, Coursera
- [Deep Learning Specialization](http://deeplearning.ai/), Andrew Ng
    - Andrew Ng’s introductory deep learning course
- Machine Learning for Trading Specialization, [Coursera](https://www.coursera.org/specializations/machine-learning-trading)
- Machine Learning for Trading, Georgia Tech CS 7646, [Udacity](https://www.udacity.com/course/machine-learning-for-trading--ud501
- Introduction to Machine Learning for Trading, [Quantinsti](https://quantra.quantinsti.com/course/introduction-to-machine-learning-for-trading)

### ML Competitions & Trading

- [IEEE Investment Ranking Challenge](https://www.crowdai.org/challenges/ieee-investment-ranking-challenge)
    - [Investment Ranking Challenge : Identifying the best performing stocks based on their semi-annual returns](https://arxiv.org/pdf/1906.08636.pdf)
- [Two Sigma Financial Modeling Challenge](https://www.kaggle.com/c/two-sigma-financial-modeling)
- [Two Sigma: Using News to Predict Stock Movements](https://www.kaggle.com/c/two-sigma-financial-news)
- [The Winton Stock Market Challenge](https://www.kaggle.com/c/the-winton-stock-market-challenge)
- [Algorithmic Trading Challenge](https://www.kaggle.com/c/AlgorithmicTradingChallenge)
   
### Python Libraries

- matplotlib [docs](https://github.com/matplotlib/matplotlib)
- numpy [docs](https://github.com/numpy/numpy)
- pandas [docs](https://github.com/pydata/pandas)
- scipy [docs](https://github.com/scipy/scipy)
- scikit-learn [docs](https://scikit-learn.org/stable/user_guide.html)
- LightGBM [docs](https://lightgbm.readthedocs.io/en/latest/)
- CatBoost [docs](https://catboost.ai/docs/concepts/about.html)
- TensorFlow [docs](https://www.tensorflow.org/guide)
- PyTorch [docs](https://pytorch.org/docs/stable/index.html)
- Machine Learning Financial Laboratory (mlfinlab) [docs](https://mlfinlab.readthedocs.io/en/latest/)
- seaborn [docs](https://github.com/mwaskom/seaborn)
- statsmodels [docs](https://github.com/statsmodels/statsmodels)
- [Boosting numpy: Why BLAS Matters](http://markus-beuckelmann.de/blog/boosting-numpy-blas.html)



















































