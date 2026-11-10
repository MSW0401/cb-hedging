"""
eb_model.py — HD현대 제19회 EB (교환대상 HD현대마린솔루션 443060) 평가·헤지 함수 (11 노트북용)
  · 가격 엔진은 numerical_engine.fdm을 사용하며, 본 모듈은 EB 칸 규칙(교환·풋·콜) · 독립 대조용 트리 · 헤지 MC만 정의
  · 조건 출처: DART rcpNo 20261027000277 (work/51_eb_terms_dart.csv, § = 공시 항목 번호)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import cb_tree_model as cb
import numerical_engine as sl


@dataclass(frozen=True)
class EBTerms:
    face: float = 100.0                     # 평가 단위: 액면 100
    exch_price: float = 247_000.0           # 교환가액(원)                      §9
    issue: date = date(2026, 11, 26)        # 납입일(발행일)                    §11
    maturity: date = date(2031, 11, 26)     # 만기 100% 일시상환, 표면·만기이자 0%  §4·5·7
    exch_start: date = date(2026, 11, 27)   # 교환청구기간                      §9
    exch_end: date = date(2031, 10, 26)
    puts: tuple = (date(2029, 11, 26), date(2030, 2, 26), date(2030, 5, 27), date(2030, 8, 26),   # 풋 지급일 8회 100% §19 표
                   date(2030, 11, 26), date(2031, 2, 26), date(2031, 5, 26), date(2031, 8, 26))   # 3차 5/26(일) → 다음 영업일 5/27
    put_claim: tuple = (date(2029, 10, 29), date(2030, 1, 28), date(2030, 4, 26), date(2030, 7, 29),   # 청구기간 말일(지급일 30일 전)
                        date(2030, 10, 28), date(2031, 1, 27), date(2031, 4, 28), date(2031, 7, 28))   # 비영업일이면 익영업일 §19 4.
    call_start: date = date(2029, 11, 26)   # 발행사 콜 행사기간                §9-1·§19
    call_end: date = date(2031, 10, 26)
    call_trigger: float = 1.30              # 직전 10거래일 연속 종가 > 교환가 × 130% → 100% 상환
    recovery: float = 0.40                  # 회수율 40% (논문 Table 1 가정, CB와 동일)
    issue_total: float = 588_667_443_000.0  # 권면총액 §2


EB = EBTerms()


def weekdays(d0, d1):
    """d0~d1 평일 (종가가 나오는 날의 근사 — 공휴일은 빼지 않음)"""
    out, d = [], d0
    while d <= d1:
        if d.weekday() < 5: out.append(d)
        d += timedelta(days=1)
    return out


class EBRule:
    """칸 규칙 (시간 t = 평가일부터 년, V = 액면 100당 값). 같은 시점에 적용하는 순서:
       ① 콜(발행사): 콜 기간의 평일 종가 시점에만, 주가 ≥ 트리거면 V ≤ max(100, c·S)
       ② 교환(투자자): 교환청구기간 매 칸 V ≥ c·S
       ③ 풋(투자자): 청구기간 말일에 결정 → V ≥ 지급일까지 남은 채권값(100을 받기 전 부도 위험 반영)
          청구한 사채는 상환 대상으로 묶이므로 이후 교환은 포기한 것으로 봄
       ①→②·③ 순서 = '공고 후에도 투자자의 교환·풋이 콜보다 우선'(§19 3.)
       콜 조건 부등호: 공시는 '종가 > 트리거'. 연속 모형에서는 경계 한 점이라 '≥'와 값이 같고, 격자에서는 트리거 칸을 포함('≥')해야
       경계가 트리거에 정확히 놓임 ('>'로 두면 경계가 한 칸 위로 밀려 격자 오차가 생김 — 11번 2절 참고 기록)
       call=True: 트리거를 넘은 종가 하루 만에 콜(10거래일·2주 공고 무시) → 실제 값의 하한 / call=False: 상한
       tol: 날짜 맞춤 허용 폭 (FDM은 날짜가 칸에 정확히 있으므로 1e-9, 트리는 Δt/2)"""
    def __init__(self, ev, c, trig, amer=True, puts=True, call=True, tol=1e-9):
        self.ev, self.c, self.trig, self.amer, self.puts, self.call, self.tol = ev, c, trig, amer, puts, call, tol
    def _on(self, t, ts):                                             # t가 날짜 목록 ts(오름차순) 중 하나와 맞는가
        i = int(np.searchsorted(ts, t))
        return any(0 <= j < len(ts) and abs(ts[j] - t) <= self.tol for j in (i - 1, i))
    def __call__(self, t, S, V):
        e = self.ev
        if self.call and self._on(t, e['mon']):
            V = np.where(S >= self.trig * (1 - 1e-12), np.minimum(V, np.maximum(100.0, self.c * S)), V)
        if self.amer and e['xs'] - self.tol <= t <= e['xe'] + self.tol:
            V = np.maximum(V, self.c * S)
        if self.puts:
            for tp, bv in zip(e['puts'], e['put_val']):
                if abs(t - tp) <= self.tol: V = np.maximum(V, bv)
        return V


def setup(S, sigma, r, q, s, val, terms=EB, eta=0, rec='face'):
    """모형 입력 묶음. λ = s/(1−R), k = r+λ.
       eta=0: HD현대 부도 시 마린솔루션 주가는 그대로 (기준)   → 주가 추세 r−q
       eta=1: 부도 시 교환대상 주가도 0 (CB식 가정, 민감도)     → 생존 조건부 추세 r−q+λ
       rec='face': 부도 시 40 / 'shares': max(40, c·S) (교환대상주식 예탁원 신탁 §9 7) 반영 민감도)
       풋 값 = 청구 마감부터 지급일까지 g년: 100·e^(−k g) + 40·λ/k·(1−e^(−k g))"""
    R = terms.recovery; lam = s / (1 - R); c = terms.face / terms.exch_price; k = r + lam
    yr = lambda d: cb.yf(val, d)
    gap = [yr(pp) - yr(pc) for pp, pc in zip(terms.puts, terms.put_claim)]
    ev = dict(T=yr(terms.maturity), xs=max(yr(terms.exch_start), 0.0), xe=yr(terms.exch_end),
              cs=yr(terms.call_start), ce=yr(terms.call_end),
              puts=[yr(d) for d in terms.put_claim],                                  # 풋 결정 시점 = 청구기간 말일
              put_val=[100 * math.exp(-k * g) + R * 100 * lam / k * (1 - math.exp(-k * g)) for g in gap],
              mon=np.array([yr(d) for d in weekdays(terms.call_start, terms.call_end)]))   # 콜 감시 = 평일 종가
    if rec == 'shares' and eta == 0:
        src = lambda X: lam * np.maximum(R * terms.face, c * X)
    else:
        src = lam * R * terms.face
    return dict(S=S, sigma=sigma, r=r, q=q, lam=lam, c=c, trig=terms.call_trigger * terms.exch_price, ev=ev,
                drift=r - q + (lam if eta == 1 else 0.0), rho=r + lam, src=src, rec=rec, eta=eta, R=R)


def _term(p, exch):
    """만기 지급: 'euro'(만기 하루 교환)면 max(100, c·S), 그 외 100 (교환청구 종료 10/26 < 만기 11/26)"""
    return (lambda X: np.maximum(100.0, p['c'] * X)) if exch == 'euro' else (lambda X: np.full_like(X, 100.0))


def price_fdm(p, exch='amer', puts=True, call=True, M=1600, NT=2520, keep=False):
    """FDM. exch: 'amer'(교환청구기간 매 칸) / 'euro'(만기 하루, 닫힌 식 대조용) / 'none'(교환권 없음 = 채권)
       시간 칸: 풋 결정일·콜 감시일(평일)을 모두 칸 위에 놓음 → 격자를 바꿔도 같은 계약을 풂"""
    amer = exch == 'amer'
    rule = EBRule(p['ev'], p['c'], p['trig'], amer, puts, call) if (amer or puts or call) else None
    ev = p['ev']; knots = [ev['xs'], ev['xe'], ev['cs'], ev['ce'], *ev['puts']] + (list(ev['mon']) if call else [])
    o = sl.fdm(p['S'], p['sigma'], ev['T'], p['drift'], p['rho'], p['src'], _term(p, exch), rule, knots=knots, M=M, NT=NT,
               keep=keep, align=p['trig'] if call else None)              # 콜 트리거를 주가 칸 위에 → 불연속 오차 제거
    o['delta'] = o['dVdS'] / p['c']                                   # 교환주식 수 대비 비율
    return o


def price_tree(p, exch='amer', puts=True, call=True, N=4000):
    """독립 대조용 이항 트리 + 부도 가지 (FDM과 코드를 공유하지 않음; 칸 규칙만 공통)
       V_i = e^(−rΔt) [ e^(−λΔt)(pu V⁺ + pd V⁻) + (1 − e^(−λΔt)) × 회수 ]   (Δt < 1일 → 날짜마다 칸 하나)"""
    T = p['ev']['T']; dt = T / N; u = math.exp(p['sigma'] * math.sqrt(dt)); d = 1 / u
    assert dt < 1 / 365, '트리 Δt는 하루보다 짧아야 날짜가 칸에 하나씩 맞음'
    pu = (math.exp(p['drift'] * dt) - d) / (u - d); pd_ = 1 - pu
    surv, disc = math.exp(-p['lam'] * dt), math.exp(-p['r'] * dt)
    rule = EBRule(p['ev'], p['c'], p['trig'], exch == 'amer', puts, call, tol=dt / 2)
    rec = lambda X: (np.maximum(p['R'] * 100, p['c'] * X) if (p['rec'] == 'shares' and p['eta'] == 0) else p['R'] * 100.0)
    S_N = p['S'] * u ** (N - 2.0 * np.arange(N + 1))
    V = rule(T, S_N, _term(p, exch)(S_N))
    for i in range(N - 1, -1, -1):
        Si = p['S'] * u ** (i - 2.0 * np.arange(i + 1))
        V = disc * (surv * (pu * V[:-1] + pd_ * V[1:]) + (1 - surv) * rec(Si))
        V = rule(i * dt, Si, V)
        if i == 1: S1, V1 = Si.copy(), V.copy()
    return dict(price=float(V[0]), delta=float((V1[0] - V1[1]) / (S1[0] - S1[1]) / p['c']))


def euro_exact(p):
    """유럽형(만기 하루 교환)·풋·콜 없음의 닫힌 식:
       eta=0: e^(−λT)[100e^(−rT) + c·Call(S, 100/c, T, r, q)] + 40·λ/k·(1−e^(−kT)),  k = r+λ
       eta=1: 100e^(−kT) + c·Call(S, 100/c, T, k, q) + 같은 회수 항  (생존 조건부 추세 k−q)"""
    T, r, q, lam, c, R = p['ev']['T'], p['r'], p['q'], p['lam'], p['c'], p['R']; k = r + lam
    recv = R * 100 * lam / k * (1 - math.exp(-k * T))
    if p['eta'] == 0:
        return math.exp(-lam * T) * (100 * math.exp(-r * T) + c * cb.bsm_call(p['S'], 100 / c, T, r, q, p['sigma'])) + recv
    return 100 * math.exp(-k * T) + c * cb.bsm_call(p['S'], 100 / c, T, k, q, p['sigma']) + recv


krx_tick = sl.krx_tick                                                # 호가단위 표 (numerical_engine에 한 번만 정의)


def hedge_mc(o, p, notional, n=10_000, steps=252, dt=1 / 252, seed=20261107, sigma_real=None, boot=None,
             sell_tax=0.0020, k_tick=0.5, borrow=0.005, Y=1.0, sd_d=None, adv=None):
    """EB 매수 + 교환대상 주식 공매도, 매 영업일 종가에 delta 재조정 (평가 σ 고정: 매일 같은 내재변동성으로 평가)
       o: price_fdm(keep=True) 결과 — 시간·주가 격자 위 V를 보간해 매일 평가가격·delta를 읽음
       주가 경로: sigma_real(GBM) 또는 boot(과거 일별 로그수익률 표본 복원추출). 추세는 둘 다 r−q로 맞춤
       부도: 하루 확률 1−e^(−λΔt), 부도일에 EB = 회수 40, 공매도는 시장에서 상환, 경로 종료
       조기 교환: 주가가 그날 격자의 교환 경계(평가 주가 = 격자 중심 위에서 V = c·S가 되는 첫 주가) 이상이면 교환,
                  받은 주식으로 공매도 상환(차이만 시장 거래), 경로 종료
       손익(원) = 액면/100 × ΔV − h·ΔS + h·S·(r−q)·Δt − 액면/100 × V·r·Δt − 거래비용
                  (공매도 대금 이자 r 수취 · 배당 q 지급 · EB 매입대금 r 조달 — 모형과 같은 금리 → S1 내재변동성 시나리오의 기대손익 0)
       거래비용(10번 노트북 기준안과 같은 항목): 반 틱 + 시장충격 eq.54 (Y·σ_D·√(Q/거래량), 거래량 = 60일 평균 adv)
                  + 매도세(공매도를 늘리는 매도) + 대차 수수료(공매도 잔고 × borrow × Δt)
       청산 비용(마지막 날 EB 매도 · 공매도 상환)은 여기서 빼지 않음 → 반환한 마지막 날 상태로 numerical_engine.mc_exit_cost에서 따로 계산"""
    assert p['eta'] == 0 and p['rec'] == 'face', '헤지 MC는 기준 가정(η=0, 회수 40)만 지원'
    rng = np.random.default_rng(seed); rng_d = np.random.default_rng([seed, 1]); units = notional / 100.0   # 부도 난수는 따로 → 세 시나리오가 같은 부도 경로 (같은 시드라 S1 · S2는 주가 난수도 공통)
    x, tg, G = np.log(o['S']), o['t'], o['grid']
    r, q, lam, c = p['r'], p['q'], p['lam'], p['c']
    dG = np.gradient(G, o['S'], axis=1)                               # 격자 위 ∂V/∂S
    i0 = len(x) // 2
    def bnd(j):                                                       # 교환 경계: 평가 주가(격자 중심) 위쪽에서 V − c·S ≤ 1e-10 인 첫 칸
        hit = np.nonzero(G[j, i0:] - c * o['S'][i0:] <= 1e-10)[0]
        return o['S'][i0 + hit[0]] if len(hit) else np.inf
    def at(t, S):                                                     # 시간 선형보간 → 주가(로그) 보간 + 그날 교환 경계
        j = min(np.searchsorted(tg, t, side='right') - 1, len(tg) - 2); w = (t - tg[j]) / (tg[j + 1] - tg[j])
        Vt, Dt = (1 - w) * G[j] + w * G[j + 1], (1 - w) * dG[j] + w * dG[j + 1]
        return np.interp(np.log(S), x, Vt), np.interp(np.log(S), x, Dt), max(bnd(j), bnd(j + 1))
    def tcost(Q, S_):                                                 # |거래 주식수| Q의 반 틱 + 시장충격 (eq.54)
        imp = Y * sd_d * np.sqrt(Q / adv) if (sd_d is not None and adv) else 0.0
        return Q * (k_tick * krx_tick(S_) + S_ * imp)
    S = np.full(n, p['S']); V, D, _ = at(0.0, S); h = units * D
    alive = np.ones(n, bool); pnl = np.zeros(n); dflt_all = np.zeros(n, bool); exch_all = np.zeros(n, bool)
    cst = tcost(np.abs(h), S) + h * S * sell_tax                     # 첫날 공매도
    cost0 = cst.copy()                                                # 첫날 공매도 비용 (거래비용 중 첫날 몫)
    trades = [h.copy()]; lr = []
    if boot is not None:
        X = np.asarray(boot, float); X = X - math.log(np.mean(np.exp(X))) + (r - q) * dt   # 표본 분포 그대로, 평균만 r−q에 맞춤
    for k in range(1, steps + 1):
        z = rng.choice(X, n) if boot is not None else (r - q - 0.5 * sigma_real ** 2) * dt + sigma_real * math.sqrt(dt) * rng.standard_normal(n)
        S1 = S * np.exp(z); dflt = alive & (rng_d.random(n) < 1 - math.exp(-lam * dt))
        V1, D1, sb = at(k * dt, S1)
        ex = alive & ~dflt & (S1 >= sb)                               # 조기 교환 (격자 경계 기준)
        V1 = np.where(dflt, p['R'] * 100, np.where(ex, c * S1, V1))
        step = units * (V1 - V) - h * (S1 - S) + h * S * (r - q) * dt - units * V * r * dt
        pnl += np.where(alive, step, 0.0)
        cst += np.where(alive, h * S * borrow * dt, 0.0)               # 대차 수수료 (전일 잔고)
        h1 = np.where(dflt | ex | ~alive, 0.0, units * D1)
        tr = np.where(alive & ex, units * c - h, np.where(alive, h1 - h, 0.0))   # 시장 거래 (교환 시 = 받은 주식 − 공매도 잔고)
        cst += tcost(np.abs(tr), S1) + np.where(tr > 0, tr * S1 * sell_tax, 0.0)
        trades.append(tr); lr.append(np.where(alive, z, np.nan))
        dflt_all |= dflt; exch_all |= ex
        alive &= ~(dflt | ex); S, V, h = S1, V1, h1
    lr = np.array(lr)[:, alive]                                        # 실현 변동성은 부도·교환 없이 끝까지 간 경로만
    return dict(pnl=pnl, pnl_cost=pnl - cst, cost=cst, trades=np.array(trades), h0=trades[0],
                vol=lr.std(axis=0, ddof=1) * math.sqrt(1 / dt), defaults=int(dflt_all.sum()), exchanged=int(exch_all.sum()),
                dflt=dflt_all, exch=exch_all, cost0=cost0, V_end=V, h_end=h, S_end=S, alive=alive)   # 마지막 날 상태 → numerical_engine.mc_exit_cost
