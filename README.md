# shopping-classification
-----
본 repository는 카카오 아레나의 쇼핑몰 상품 카테고리 분류 대회용 repository 입니다.
-----

- Team Name : error모르겠다
- Members : joonable(https://github.com/joonable), yamarae(https://github.com/yoonkt200)
-----
## 최종 예측 결과

- d
-----
## 소스 코드 실행 방법


0. Install
    - 본 repository의 소스코드를 내려받습니다.
        - `git clone https://github.com/dunna-error/shopping-classification.git`
    - Kakao arena의 데이터를 `../dataset/` 경로에 내려받습니다.
    - 필요한 패키지를 설치합니다.
        - `pip install -r requirements.txt`
    - 필요한 써드파티 라이브러리를 설치합니다.
        - Elastic Seach
            - 형태소 분석을 위해 elasticsearch의 한글 형태소 분석기인 nori를 사용합니다.
            - elaticsearch v6.5, nori-plugin 설치가 필수로 요구됩니다.
            - elaticsearch 설치 : https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
            - nori 설치 : http://joonable.tistory.com/10?category=682569
1. Data Preprocessing
    - 데이터셋을 생성하기 전, 필요한 전처리 작업들을 수행합니다.
        - 빠른 개발을 위해, Dataframe 형태로 데이터를 변환합니다.
            - `python preprocess.py make_df train`
            - `python preprocess.py make_df dev`
            - `python preprocess.py make_df test`
            - 변환된 데이터는 `../dataset/` 경로에 `train_df.csv` 형태로 저장됩니다.
        - 전처리에 필요한 중간 단계의 파일을 생성합니다.
            - `python preprocess.py make_dict`
            - `python preprocess.py make_b2v_model`
            - `python preprocess.py make_parsed_product`
            - `python preprocess.py make_d2v_model`
    - 훈련용 데이터셋을 생성합니다.
        - `python data.py make_db train`
            - 모델을 학습하기 위한 데이터셋을 (train:dev = 8:2) 비율로 생성합니다.
            - 생성된 훈련용 데이터셋은 `./data/train/` 경로에 `data.h5py` 이름으로 생성됩니다.
        - 모델 학습에 필요한 중간 단계의 파일을 생성합니다.
            - `python meta_reverse_dict.py reverse_meta ./data/ train`
            - `python meta_reverse_dict.py cate_index ./data/`
2. Train
    - 생성한 데이터셋을 이용하여, 아래 순서대로 4단계 학습을 진행합니다.
        - `python multilayer_classifier.py train ./data/train ./model/train b`
        - `python multilayer_classifier.py train ./data/train ./model/train m`
        - `python multilayer_classifier.py train ./data/train ./model/train s`
        - `python multilayer_classifier.py train ./data/train ./model/train d`
    - 완성된 모델은 `./model/train`에 위치합니다.
3. Predict
    - Train
        - `python multilayer_classifier.py predict ./data/train ./model/train ./data/train/ dev predict.tsv`
        - 훈련용 데이터셋에서 'dev' 데이터셋을 사용하여 분류 예측 결과를 `predict.tsv`에 생성합니다.
    - Dev
        - `python data.py make_db dev ./data/dev --train_ratio=0.0`
        - `python multilayer_classifier.py predict ./data/train ./model/train ./data/dev/ dev baseline.predict.tsv`
        - 리더보드 제출용 예측 파일을 생성합니다.
    - Test
        - `python data.py make_db test ./data/test --train_ratio=0.0`
        - `python multilayer_classifier.py predict ./data/train ./model/train ./data/test/ dev final.predict.tsv`
        - 최종 제출용 예측 파일을 생성합니다.
4. Evaluate(develop only)
    - `python evaluate.py evaluate predict.tsv ./data/train/data.h5py dev ./data/y_vocab.py3.cPickle`
        - 훈련용 데이터셋에서 'dev' 데이터셋으로 생성한 예측 결과를 평가합니다.
-----
## 로직 설명

0. EDA
    - (참고 : https://github.com/dunna-error/shopping-classification/blob/master/yamarae/eda/02_eda_dataframe.ipynb)
        - `y` : 'b>m>s>d'의 조합은 총 4215개로 구성, 세>소>중>대 순으로 skewed한 분포를 보임.
        - `pid` : primary key로 기능.
        - `brand`
            - 약 4%정도가 브랜드 쓰레기값 포함
            - 약 50%정도가 브랜드 결측
            - dev, test의 brand : 3%정도는 train에서 등장하지 않았던 브랜드
        - `maker` : 브랜드 정보는 없지만 제조사 정보만 있는 경우 : 약 30%
        - `price`
            - price 결측 비율 약 64%
            - quantile로 나누었을 때, 합리적인 분포로 보임
        - `updttm` : b,m,s,d 각각으로 groupby 하였을 때, 카테고리별 나이(현재 unixtime - 생성시 unixtime)에 차이를 보임.
1. Feature Engineering 
    - (참고 : https://github.com/dunna-error/shopping-classification/blob/master/yamarae/eda/03_feature_engineering_df_ver.ipynb)
        - `b2v` : 같은 카테고리로 분류된 브랜드를 idx로 치환하여 데이터셋으로 만든 뒤, Word2Vect 알고리즘을 이용하여 Brand2Vect 생성.
            - (참고 : https://github.com/dunna-error/shopping-classification/blob/master/yamarae/eda/04_brand2vect_proto.ipynb)
        - `'img_feat', 'pid'` : 원본 그대로 사용
        - `price_lev` : 결측값을 제외한 전체 가격 데이터를 기준으로 (quantile 0.3, 0.7)을 기준으로 가격을 3개의 카테고리로 분류. 결측값은 '중' 가격으로 치환
        - `aging` : 데이터의 updttm을 기준으로 아이템의 생성 나이를 구한 뒤, min-max scaling
        - `d2v` : product 피처 정보를 Doc2Vec 알고리즘으로 새롭게 생성.
2. Modeling
    - ![img](https://github.com/dunna-error/shopping-classification/blob/master/images/model_architecture.png)