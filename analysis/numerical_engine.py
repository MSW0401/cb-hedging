"""
numerical_engine.py — 10(현대건설 CB)·11(HD현대 EB) 노트북 공용 계산 함수
  · cb_tree_model.py(트리·채권가치·역산·점검기)는 수정 없이 그대로 사용
  · scipy 없이 numpy만 사용 (정규분포 = math.erf, 연립방정식 = 직접 구현한 삼중대각 해법)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import cb_tree_model as cb


def ncdf(x: float) -> float:
    """표준정규 누적분포 N(x)"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────
# 1. 닫힌 식 (현대건설 CB: 배당·이자 0, 부도 시 주가 0)
# ─────────────────────────────────────────────────────────────
def cb_exact(S, sigma, r, s, val, terms=cb.TERMS, convertible=True):
    """중간 전환 이득이 없으므로 전환청구 종료일 t_e에 한 번 판단하는 옵션으로 정확히 풀림.
    가격 = B·e^(−k t_e) + c·Call(S, K=B/c, 금리 k, t_e) + 40·λ/k·(1−e^(−k t_e)),  k = r+λ
    B = t_e 시점에 남은 채권값(만기까지 약 1개월, 부도 포함) = cb.bf_recovery
    delta(전환주식 대비) = N(d1)"""
    R = terms.recovery; lam = s / (1 - R); k = r + lam
    T = cb.yf(val, terms.maturity); te = cb.yf(val, terms.conv_end)
    c = terms.face / terms.conv_price
    B = cb.bf_recovery(r, s, T - te, R, terms.face)
    dflt = R * terms.face * lam / k * (1 - math.exp(-k * te))       # t_e 전에 부도날 때 받는 40의 현재가치
    if not convertible:
        return dict(price=B * math.exp(-k * te) + dflt, delta=0.0)
    K = B / c                                                        # 주식 1주로 환산한 행사가
    v = sigma * math.sqrt(te)
    d1 = (math.log(S / K) + (k + 0.5 * sigma ** 2) * te) / v; d2 = d1 - v
    call = S * ncdf(d1) - K * math.exp(-k * te) * ncdf(d2)
    return dict(price=B * math.exp(-k * te) + c * call + dflt, delta=ncdf(d1))


# ─────────────────────────────────────────────────────────────
# 2. FDM (완전 음함수, x = ln S) — CB·EB 공용 엔진
# ─────────────────────────────────────────────────────────────
def _factor(lo, di, up):
    """삼중대각 행렬 LU 분해(토마스) — 같은 Δt면 재사용"""
    n = len(di); cp = [0.0] * n; m = [0.0] * n
    m[0] = di[0]; cp[0] = up[0] / m[0]
    for i in range(1, n):
        m[i] = di[i] - lo[i] * cp[i - 1]
        cp[i] = up[i] / m[i] if i < n - 1 else 0.0
    return lo, cp, m


def _solve(f, d):
    lo, cp, m = f; n = len(m); x = [0.0] * n
    x[0] = d[0] / m[0]
    for i in range(1, n):
        x[i] = (d[i] - lo[i] * x[i - 1]) / m[i]
    for i in range(n - 2, -1, -1):
        x[i] -= cp[i] * x[i + 1]
    return x


def fdm(S0, sigma, T, drift, rho, src, terminal, rule=None, knots=(), M=800, NT=1000, nsd=6.0, keep=False, align=None):
    """V_τ = ½σ² V_xx + (drift − ½σ²) V_x − ρ V + src   (τ = 잔존 기간, 만기에서 평가일 방향으로 후진 계산)
       CB: drift = r+λ, ρ = r+λ, src = 40λ  /  EB(η=0): drift = r−q, ρ = r+λ, src = 40λ
       src: 숫자 또는 함수 src(S) (주가별 부도 회수 × λ)
       rule(t, S, V) → V : 시간 t(평가일부터 년)에 적용할 칸 규칙(전환·교환·풋·콜)
       knots: 시간 칸에 정확히 와야 하는 날짜(년) — 구간별 등간격으로 배치
       align: 주가 칸에 정확히 와야 하는 값(EB 콜 트리거) — 불연속 지점이 칸 사이에 끼면 오차가 칸 폭만큼 생김
       경계: 아래쪽 V_x = 0 (주가가 아주 낮으면 V가 S에 둔감), 위쪽 V_SS = 0 (V가 S에 직선)"""
    if M % 2: M += 1
    L = nsd * sigma * math.sqrt(T)
    kk = 0 if align is None else round(abs(math.log(align / S0)) / (2 * L / M))
    if kk == 0:                                                      # 맞출 값이 없거나 현재가와 반 칸 이내면 맞추지 않음 (격자 축소 방지)
        x = math.log(S0) + np.linspace(-L, L, M + 1)
    else:                                                            # 주가 align(예: 콜 트리거)이 정확히 칸에 오도록 칸 폭 조정 (폭 변화 < 반 칸)
        x = math.log(S0) + (np.arange(M + 1) - M // 2) * (abs(math.log(align / S0)) / kk)
    dx = x[1] - x[0]; S = np.exp(x)
    pts = sorted({0.0, T, *[k for k in knots if 1e-12 < k < T - 1e-12]})
    tg = [0.0]
    for a, b in zip(pts, pts[1:]):                                   # 사건 날짜가 칸에 정확히 오도록 구간별 등간격
        n = max(1, int(round(NT * (b - a) / T)))
        tg += list(a + (b - a) * np.arange(1, n + 1) / n)
    tg = np.array(tg); tg[-1] = T
    mu = drift - 0.5 * sigma ** 2
    a_ = 0.5 * sigma ** 2 / dx ** 2 - mu / (2 * dx)                  # V_{i-1} 계수
    b_ = -sigma ** 2 / dx ** 2 - rho                                 # V_i
    c_ = 0.5 * sigma ** 2 / dx ** 2 + mu / (2 * dx)                  # V_{i+1}
    g = math.exp(dx)
    cache = {}
    def fac(dt):
        key = round(dt, 14)
        if key not in cache:
            n = M + 1
            lo = [-dt * a_] * n; di = [1 - dt * b_] * n; up = [-dt * c_] * n
            up[0] = -dt * (a_ + c_)                                  # 아래 경계: 가상점 V_{-1} = V_1
            lo[-1] = -dt * (a_ - c_ * g); di[-1] = 1 - dt * (b_ + c_ * (1 + g))   # 위 경계: S에 대해 직선
            cache[key] = _factor(lo, di, up)
        return cache[key]
    if callable(src): src = np.asarray(src(S), dtype=float)            # 부도 시 받는 값이 주가에 따라 다를 때(EB 회수 max(40, c·S))
    V = np.asarray(terminal(S), dtype=float)
    if rule is not None: V = rule(T, S, V)
    grid = [V.copy()] if keep else None
    for j in range(len(tg) - 2, -1, -1):
        dt = tg[j + 1] - tg[j]
        V = np.array(_solve(fac(dt), list(V + dt * src)))
        if rule is not None: V = rule(tg[j], S, V)
        if keep: grid.append(V.copy())
    i = M // 2
    out = dict(price=float(V[i]), dVdS=float((V[i + 1] - V[i - 1]) / (S[i + 1] - S[i - 1])), S=S, t=tg, V=V)
    if keep: out["grid"] = np.array(grid[::-1])                      # grid[j] = 시간 tg[j]의 V
    return out


class CBRule:
    """현대건설 CB 칸 규칙: 전환청구기간 [ts, te] 안에서 V = max(V, c·S).
    점검용: 종료일 전에 전환이 걸린 칸 수를 '현재 주가 ± 4σ√T' 범위 안에서만 셈 (격자 끝의 경계 효과 제외, 이론값 0)"""
    def __init__(self, ts, te, c, band=None):
        self.ts, self.te, self.c, self.band = ts, te, c, band; self.early = 0
    def __call__(self, t, S, V):
        if self.ts - 1e-12 <= t <= self.te + 1e-12:
            cv = self.c * S
            if t < self.te - 1e-9:
                hit = cv > V + 1e-12
                if self.band is not None: hit &= (S >= self.band[0]) & (S <= self.band[1])
                self.early += int(hit.sum())
            return np.maximum(V, cv)
        return V


def cb_fdm(S, sigma, r, s, val, terms=cb.TERMS, M=800, NT=1000):
    R = terms.recovery; lam = s / (1 - R)
    T = cb.yf(val, terms.maturity); ts = max(cb.yf(val, terms.conv_start), 0.0); te = cb.yf(val, terms.conv_end)
    c = terms.face / terms.conv_price
    band = (S * math.exp(-4 * sigma * math.sqrt(T)), S * math.exp(4 * sigma * math.sqrt(T)))
    rule = CBRule(ts, te, c, band)
    o = fdm(S, sigma, T, r + lam, r + lam, lam * R * terms.face, lambda x: np.full_like(x, terms.face), rule,
            knots=(ts, te), M=M, NT=NT)
    return dict(price=o["price"], delta=o["dVdS"] / c, early=rule.early)


# ─────────────────────────────────────────────────────────────
# 3. MC (현대건설 CB: 마감일 한 번 판단)
# ─────────────────────────────────────────────────────────────
def cb_mc(S, sigma, r, s, val, n=1_000_000, seed=20261107, terms=cb.TERMS):
    """부도 시점 τ ~ 지수분포(λ). τ ≤ t_e → 40·e^(−rτ),  아니면 e^(−r t_e)·max(c·S_te, B).
    분산 축소: 대칭 난수(Z,−Z / U,1−U) + 통제변수 X = e^(−r t_e)·1{생존}·c·S_te (기댓값 = c·S, 정확히 알려짐).
    표준오차는 쌍 평균으로 계산(가격·delta 모두). delta = 경로 미분(∂지급액/∂S ÷ c)."""
    R = terms.recovery; lam = s / (1 - R)
    T = cb.yf(val, terms.maturity); te = cb.yf(val, terms.conv_end)
    c = terms.face / terms.conv_price; B = cb.bf_recovery(r, s, T - te, R, terms.face)
    rng = np.random.default_rng(seed); h = n // 2
    Z = rng.standard_normal(h); U = rng.random(h)
    Z = np.concatenate([Z, -Z]); U = np.concatenate([U, 1 - U])
    tau = -np.log(U) / lam; alive = tau > te
    ST = S * np.exp((r + lam - 0.5 * sigma ** 2) * te + sigma * math.sqrt(te) * Z)
    pay = np.where(alive, math.exp(-r * te) * np.maximum(c * ST, B), R * terms.face * np.exp(-r * np.minimum(tau, te)))
    X = np.where(alive, math.exp(-r * te) * c * ST, 0.0)            # 통제변수, E[X] = c·S
    b = np.cov(pay, X)[0, 1] / X.var(ddof=1)
    Y = pay - b * (X - c * S)
    pair = 0.5 * (Y[:h] + Y[h:])
    dl = np.where(alive & (c * ST > B), math.exp(-r * te) * ST / S, 0.0)
    dpair = 0.5 * (dl[:h] + dl[h:])
    return dict(price=float(Y.mean()), se=float(pair.std(ddof=1) / math.sqrt(h)),
                delta=float(dl.mean()), delta_se=float(dpair.std(ddof=1) / math.sqrt(h)))


# ─────────────────────────────────────────────────────────────
# 4. 일별 헤지 루프 (기존 시뮬레이션과 같은 계산 순서 + 추가 비용)
# ─────────────────────────────────────────────────────────────
def hedge_run(D, P, delta, notional, nsh, dps, thr=0.0, borrow=0.005, rebate="ktb1", tick=100.0, k_tick=0.5,
              Y=1.0, sell_tax=0.0, funding=None):
    """D: 일별 시장값(일자, val, S, ktb1, 거래량, sd60_daily) / P: CB 평가가격(액면 100당) / delta: 전환주식 대비
    t−1 종가 보유분으로 t일 손익을 계산하고, t일 종가에 수량을 맞춤.
    k_tick: 반 틱=0.5(기존) · funding: None=미반영, 숫자=국고1년에 더할 가산금리(조달 = 액면 × (국고1년 + funding) × d/365) · rebate: 'ktb1' 또는 연율(0=미수취)"""
    rows = []; h = 0.0; last = None
    for t in range(len(D)):
        row = D.iloc[t]; S = row["S"]; dl = float(delta[t])
        out = dict(일자=row["일자"], S=S, delta=dl)
        if t == 0:
            out.update(CB손익=0.0, 주식손익=0.0, 이자=0.0, 대차=0.0, 배당=0.0, 조달=0.0)
        else:
            prev = D.iloc[t - 1]; d = (row["val"] - prev["val"]).days
            rb = prev["ktb1"] / 100 if rebate == "ktb1" else float(rebate)
            out.update(CB손익=notional * (P[t] - P[t - 1]) / 100,
                       주식손익=-h * (S - prev["S"]),                       # 공매도: 주가 하락 시 이익
                       이자=h * prev["S"] * rb * d / 365,                    # 공매도 대금 이자 (수익)
                       대차=-h * prev["S"] * borrow * d / 365,
                       배당=-h * dps / 252,                                  # 논문 eq.53 (매일 1/252)
                       조달=0.0 if funding is None else -notional * (prev["ktb1"] / 100 + funding) * d / 365)
        target = min(dl, 1.0) * nsh; trade = 0.0
        if last is None or abs(dl - last) > thr:                         # eq.49 리밸런싱 규칙
            trade = target - h; last = dl
        Q = abs(trade)
        impact = Y * row["sd60_daily"] * math.sqrt(Q / row["거래량"]) if Q > 0 else 0.0   # eq.54 시장충격
        cost = Q * S * ((k_tick * tick) / S + impact)
        tax = trade * S * sell_tax if trade > 0 else 0.0                 # 공매도를 늘리는 거래 = 매도 → 과세
        h += trade
        out.update(거래주식수=trade, 보유공매도=h, 거래비용=-cost, 매도세=-tax, 리밸런싱=bool(Q > 0), 거래량대비=Q / row["거래량"])
        out["합계"] = sum(out[k] for k in ("CB손익", "주식손익", "이자", "대차", "배당", "조달", "거래비용", "매도세"))
        rows.append(out)
    return pd.DataFrame(rows)


def exit_cost(df, D, P_last, notional, cb_bidask=0.0037, tick=100.0, k_tick=0.5, Y=1.0):
    """마지막 날(11/2) 청산 가정: CB 호가 차이 절반 + 공매도 상환(반 틱 + 시장충격 eq.54, 그날 거래량 · σ_D). 기존 시뮬레이션과 같은 정의"""
    last = D.iloc[-1]; hl = df["보유공매도"].iloc[-1]
    cbx = notional * P_last / 100 * cb_bidask / 2
    cover = hl * last["S"] * ((k_tick * tick) / last["S"] + Y * last["sd60_daily"] * math.sqrt(hl / last["거래량"])) if hl > 0 else 0.0
    return cbx + cover


def krx_tick(S):
    """유가증권시장 호가가격단위 (KRX 업무규정 시행세칙, 2023-01-25 시행): 5만~20만 100원 · 20만~50만 500원 등"""
    S = np.asarray(S, dtype=float)
    return np.select([S < 2e3, S < 5e3, S < 2e4, S < 5e4, S < 2e5, S < 5e5], [1, 5, 10, 50, 100, 500], 1000.0)


# ─────────────────────────────────────────────────────────────
# 5. CB 헤지 시뮬레이션 (주가 경로를 모른다고 가정, 11번 EB 헤지 MC와 같은 틀)
# ─────────────────────────────────────────────────────────────
def cb_grid(S, sigma, r, s, val, terms=cb.TERMS, M=1600, NT=2000, knots=()):
    """CB 가격표: cb_fdm과 같은 식(부도 시 주가 0, η = 1)을 풀되 시간 × 주가 격자 전체를 보관 (keep=True)
       knots: 헤지 날짜(년) — 시간 칸에 정확히 놓음"""
    R = terms.recovery; lam = s / (1 - R)
    T = cb.yf(val, terms.maturity); ts = max(cb.yf(val, terms.conv_start), 0.0); te = cb.yf(val, terms.conv_end)
    c = terms.face / terms.conv_price
    o = fdm(S, sigma, T, r + lam, r + lam, lam * R * terms.face, lambda x: np.full_like(x, terms.face), CBRule(ts, te, c),
            knots=(ts, te, *knots), M=M, NT=NT, keep=True)
    o.update(c=c, r=r, lam=lam, R=R, S0=S, ts=ts, delta=o['dVdS'] / c)
    return o


def cb_hedge_mc(o, notional, steps, dt=1 / 252, n=10_000, seed=20261107, sigma_real=None, boot=None, path=None, dps=0.0,
                sell_tax=0.0020, k_tick=0.5, borrow=0.005, Y=1.0, sd_d=None, adv=None):
    """CB 매수 + 현대건설 주식 공매도, 매 영업일 종가에 delta 재조정 (평가 σ 고정: 매일 같은 내재변동성으로 평가)
       o: cb_grid 결과 — 시간·주가 격자 위 V를 보간해 매일 평가가격·delta를 읽음
       주가 경로(부도 전): sigma_real(GBM) · boot(과거 일별 로그수익률 복원추출) · path(실제 종가 1개 경로, 부도 없음)
         추세 r + λ: 부도 시 주가가 0이 되는 모형(η = 1)에서 부도 전 상승률 → 부도까지 포함한 기대수익률 = r
       부도: 하루 확률 1 − e^(−λΔt), 부도일에 CB = 회수 40 · 주가 = 0 (공매도 상환 비용 없음), 경로 종료
       모형 손익(원) = 액면/100 × ΔV − h·ΔS + h·S·r·Δt − 액면/100 × V·r·Δt   (공매도 대금 이자 · CB 매입대금 조달 = 모형 금리 r)
       배당 = h × 주당 배당금 / 252 (논문 eq.53, 평가 모형은 배당 0이라 모형 밖 비용으로 따로 집계)
       거래비용 = 반 틱 + 시장충격 eq.54 (Y·σ_D·√(Q/거래량), 거래량 = 60일 평균 adv) + 매도세 + 대차료(잔고 × borrow × Δt)"""
    rng = np.random.default_rng(seed); rng_d = np.random.default_rng([seed, 1]); units = notional / 100.0
    x, tg, G = np.log(o['S']), o['t'], o['grid']; r, lam = o['r'], o['lam']
    dG = np.gradient(G, o['S'], axis=1)                               # 격자 위 ∂V/∂S
    def at(t, S):                                                     # 시간 선형보간 → 주가(로그) 보간
        j = min(np.searchsorted(tg, t, side='right') - 1, len(tg) - 2); w = (t - tg[j]) / (tg[j + 1] - tg[j])
        Vt, Dt = (1 - w) * G[j] + w * G[j + 1], (1 - w) * dG[j] + w * dG[j + 1]
        return np.interp(np.log(S), x, Vt), np.interp(np.log(S), x, Dt)
    def tcost(Q, S_):                                                 # |거래 주식수| Q의 반 틱 + 시장충격 (eq.54)
        imp = Y * sd_d * np.sqrt(Q / adv) if (sd_d is not None and adv) else 0.0
        return Q * (k_tick * krx_tick(S_) + S_ * imp)
    if path is not None:
        path = np.asarray(path, float); n = 1; steps = len(path) - 1
        assert abs(path[0] - o['S0']) < 1e-9, '실제 경로 시작가 = 가격표 기준 주가'
    assert steps * dt < o['ts'], '헤지 기간 안에 전환청구 기간이 없어야 함 (조기 전환 없음)'
    S = np.full(n, o['S0']); V, D = at(0.0, S); h = units * D
    alive = np.ones(n, bool); pnl = np.zeros(n); div = np.zeros(n); dflt_all = np.zeros(n, bool)
    cst = tcost(np.abs(h), S) + h * S * sell_tax                     # 첫날 공매도
    cost0 = cst.copy()                                                # 첫날 공매도 비용 (거래비용 중 첫날 몫)
    trades = [h.copy()]; lr = []; cum = [-cst.copy()]               # 누적 손익: 첫날은 첫 공매도 비용만
    if boot is not None:
        X = np.asarray(boot, float); X = X - math.log(np.mean(np.exp(X))) + (r + lam) * dt   # 표본 분포 그대로, 평균만 r + λ에 맞춤
    for k in range(1, steps + 1):
        if path is not None:
            S1 = np.array([path[k]]); z = np.log(S1 / S); dflt = np.zeros(1, bool)
        else:
            z = rng.choice(X, n) if boot is not None else (r + lam - 0.5 * sigma_real ** 2) * dt + sigma_real * math.sqrt(dt) * rng.standard_normal(n)
            dflt = alive & (rng_d.random(n) < 1 - math.exp(-lam * dt))
            S1 = np.where(dflt, 0.0, S * np.exp(z))
        Sx = np.where(alive & ~dflt, S1, o['S0'])                     # 부도 · 종료 경로는 계산용 임시값 (결과에 쓰지 않음)
        V1, D1 = at(k * dt, Sx); V1 = np.where(dflt, o['R'] * 100, V1)
        step = units * (V1 - V) - h * (S1 - S) + h * S * r * dt - units * V * r * dt
        pnl += np.where(alive, step, 0.0)
        div += np.where(alive, h * dps / 252, 0.0)                     # 배당 (eq.53)
        cst += np.where(alive, h * S * borrow * dt, 0.0)               # 대차료 (전일 잔고)
        h1 = np.where(dflt | ~alive, 0.0, units * D1)
        tr = np.where(alive & ~dflt, h1 - h, 0.0)                     # 부도 시 주가 0 → 공매도 상환 비용 없음
        cst += tcost(np.abs(tr), Sx) + np.where(tr > 0, tr * Sx * sell_tax, 0.0)
        trades.append(tr); lr.append(np.where(alive & ~dflt, z, np.nan))
        dflt_all |= dflt; alive &= ~dflt; S, V, h = S1, V1, h1
        cum.append(pnl - div - cst)
    lr = np.array(lr)[:, alive]
    return dict(pnl=pnl, div=div, cost=cst, pnl_cost=pnl - div - cst, cum=np.array(cum), trades=np.array(trades), h0=trades[0],
                vol=lr.std(axis=0, ddof=1) * math.sqrt(1 / dt), defaults=int(dflt_all.sum()), dflt=dflt_all,
                cost0=cost0, V_end=V, h_end=h, S_end=S, alive=alive)   # 마지막 날 상태 → mc_exit_cost


def mc_exit_cost(m, notional, sd_d, adv, bidask=0.0037, k_tick=0.5, Y=1.0):
    """헤지 MC 마지막 날 청산 가정 비용(원, 경로별) — 10번 4절 exit_cost와 같은 항목:
       ① 증권(CB·EB) 매도: 액면/100 × 평가가격 × 호가 차이의 절반 (논문 p.34, 0.37%)
       ② 남은 공매도 상환(매수, 매도세 없음): 반 틱 + 시장충격 eq.54
       호가단위 = 그날 주가 기준 krx_tick, 거래량 = 60일 평균 adv (MC의 다른 거래비용과 같은 기준)
       부도 · 조기 교환으로 이미 끝난 경로는 0 (해당 비용은 그날 거래비용에 이미 반영)"""
    a = m['alive']; Q = np.where(a, np.abs(m['h_end']), 0.0); Sx = np.where(a, m['S_end'], 1.0)
    cover = Q * (k_tick * krx_tick(Sx) + Sx * Y * sd_d * np.sqrt(Q / adv))
    return np.where(a, notional / 100 * m['V_end'] * bidask / 2 + cover, 0.0)


# ─────────────────────────────────────────────────────────────
# 6. 국채선물 금리 헤지 (CB 실제 경로: 금리 위험만, 거래소 국채선물 가격식)
# ─────────────────────────────────────────────────────────────
def ktbf_price(y, n_years, cpn=5.0):
    """국채선물 가격식: 표면 연 cpn%·6개월 이자·만기 n_years년 가상 국채를 수익률 y(연, 6개월 복리)로 할인 (액면 100)
       P = Σ_{i=1..2n} (cpn/2)/(1+y/2)^i + 100/(1+y/2)^{2n}"""
    v = 1 / (1 + y / 2); m = 2 * n_years
    return sum(cpn / 2 * v ** i for i in range(1, m + 1)) + 100 * v ** m


def ktbf_yield(P, n_years, lo=-0.05, hi=0.30, n_iter=100):
    """정산가 → 수익률 (이분법: 수익률이 오르면 가격은 내려감)"""
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if ktbf_price(mid, n_years) > P: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


def ktbf_dv01(P, n_years, mult=1_000_000):
    """선물 1계약의 수익률 1bp 상승 시 손실액(원): [P(y−1bp) − P(y+1bp)] ÷ 2 × 거래승수(1pt = 100만원)"""
    y = ktbf_yield(P, n_years)
    return (ktbf_price(y - 1e-4, n_years) - ktbf_price(y + 1e-4, n_years)) / 2 * mult


def ktbf_hedge(px, held, dv01_cb, n_years=3, mult=1_000_000, tick=0.01, k_tick=0.5):
    """px: 날짜(행) × 월물(열) 정산가 · held: 날짜별 그날 종가 이후 보유 월물 · dv01_cb: 날짜별 CB 금리 1bp 상승 시 손실(원)
       매일 종가에 계약 수 = round(CB DV01 ÷ 선물 1계약 DV01)만큼 매도 → 손익은 전날 보유 월물의 정산가 변화로 계산
       롤오버 날: 전날 월물 전량 환매 + 새 월물 매도 → 두 거래 모두 비용 (계약당 반 틱 = 0.5 × 0.01pt × 100만원)"""
    rows = []; n_prev = 0; c_prev = None
    for t in range(len(held)):
        c = held[t]; F = float(px[c].iloc[t])
        dv01_f = ktbf_dv01(F, n_years, mult); n = int(round(dv01_cb[t] / dv01_f))
        pnl = 0.0 if t == 0 else -n_prev * (float(px[c_prev].iloc[t]) - float(px[c_prev].iloc[t - 1])) * mult   # 매도 포지션
        q = n if t == 0 else (abs(n - n_prev) if c == c_prev else n_prev + n)
        rows.append(dict(월물=c, 정산가=F, 선물DV01=dv01_f, CB_DV01=float(dv01_cb[t]), 계약수=n, 선물손익=pnl,
                         거래계약=q, 선물비용=-q * k_tick * tick * mult))
        n_prev, c_prev = n, c
    return pd.DataFrame(rows)
