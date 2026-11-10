# CB · EB 평가 및 델타 헤지

현대건설 제312회 사모 CB와 HD현대 제19회 EB를 Alenius (2022) 방법론으로 평가하고, 델타 헤지 손익을 분석함

우민성 (한국투자증권 투자공학2부 현장실습) · 2026-11

## 대상

- 현대건설 제312회 CB: 액면 50억, 실제 헤지 기간 2026-07-07 ~ 11-02 (79영업일)
- HD현대 제19회 EB: 교환대상 HD현대마린솔루션, 투자 100억, 1년 헤지 시뮬레이션 (1만 경로)

## 결과

**현대건설 CB**
- 트리 · FDM · MC 가격 일치 (트리 · FDM은 닫힌 식 대비 오차 0.004p 이내, MC는 표준오차 3배 이내)
- 내재변동성 35.25% (과거 1년 변동성 75.87%)
- 거래비용 반영 후 +1.048억, 청산 비용까지 반영 시 +0.842억
- 3년 국채선물로 금리 헤지 시 금리 요인 변동 68% 감소, 손익 +0.615억

**HD현대 EB**
- FDM · 트리 가격 일치 (권리를 단계별로 추가하며 대조)
- 내재변동성 38.92 ~ 39.84% (과거 1년 변동성 63.19%)
- 1년 헤지 시 비용 후 +2.021억, 청산 비용까지 반영 시 +1.321억 (손실 확률 2 ~ 3%)
- 부도 위험은 헤지 후에도 남으며, 금리 헤지에는 3년 국채선물 75계약 필요

근거, 가정, 결정이 필요한 사항은 [보고서](Valuation_and_Delta_Hedging_Report.md)에 정리함

## 파일

[보고서](Valuation_and_Delta_Hedging_Report.md) · [CB 노트북](analysis/10_HyundaiEnC_CB_Valuation_and_Hedging.ipynb) · [EB 노트북](analysis/11_HDHyundai_EB_Valuation_and_Hedging.ipynb)

```
Valuation_and_Delta_Hedging_Report.md           분석 보고서
analysis/
  10_HyundaiEnC_CB_Valuation_and_Hedging.ipynb  현대건설 CB
  11_HDHyundai_EB_Valuation_and_Hedging.ipynb   HD현대 EB
  cb_tree_model.py       CB 조건 · 트리 · 내재변동성
  numerical_engine.py    닫힌 식 · FDM · MC · 헤지 · 국채선물
  eb_model.py            EB 조건 · 교환 · 풋 · 콜 · 헤지
  plot_style.py          그림 서식
tools/
  build_notebook_10.py, build_notebook_11.py    노트북 생성 스크립트
  run_notebook.py                               노트북 실행기
```

노트북은 실행 결과(표 · 그림)가 저장되어 있어 바로 열람 가능함

## 방법

- 금리는 국고 커브 보간, 신용스프레드는 AA- 5년 − 국고 5년, 부도강도는 스프레드 ÷ (1 − 회수율 40%)
- 가격은 트리 · FDM · MC로 계산하고, CB는 닫힌 식으로 검증함
- 매 영업일 종가에 delta만큼 공매도하며, 반 틱 · 시장충격 · 매도세 0.20% · 대차료 · 조달비용 · 청산 비용을 반영함
- 금리 헤지는 CB 금리 민감도만큼 3년 국채선물을 매도함
- 점검 항목 CB 92건 · EB 48건 모두 통과

## 재실행

- Python 3.10 이상, numpy · pandas · matplotlib · xlrd
- 원자료(`data/`)와 입력 파일(`work/`)은 저장소에 포함하지 않음
- `python tools/run_notebook.py analysis/10_HyundaiEnC_CB_Valuation_and_Hedging.ipynb` (약 1분), 11번 노트북은 약 3분
- 비용 가정은 각 노트북 상단 `CFG`에서 변경

## 출처

- Alenius (2022), *Convertible Bonds*
- 현대건설 · HD현대 주요사항보고서 (DART)
- KIS Open API, 한국은행 ECOS, 금융투자협회, KRX
