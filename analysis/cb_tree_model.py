"""
cb_tree_model.py — 현대건설 제312회 사모 CB: 평가·헤지 시뮬레이션 공통 함수

근거
  - Alenius (2022) LUT 석사논문: eq.4(채권가치), eq.21~35(Milanov et al. 2013 신용위험 트리),
    eq.36~37(이표), eq.47(스프레드 = PD×LGD), eq.48(delta), eq.49(리밸런싱), eq.51~54(수익·비용)
  - 비교용 일반 방식: 채권가치 + BSM 콜 (5-1절, 논문 방식 아님)
  - 현대건설 주요사항보고서(2026-06-09) p.2: 만기·전환가·전환청구기간·청약/납입일

원칙
  - 각 함수는 기존 분석(02 노트북, 저장소 미포함)에서 손계산과 대조 검증함.
  - 수치 입력은 노트북이 원자료에서 읽어 전달하며, 본 모듈에는 공시 조건과 논문 가정만 정의함.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0. 계약 조건 (공시 원문) · 논문 가정
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CBTerms:
    face: float = 100.0                    # 평가 단위: 액면 100
    conv_price: float = 150_607.0          # 전환가액(원)           공시 p.2 §9
    issue: date = date(2026, 7, 7)         # 청약일·납입일          공시 p.3 §11·12
    maturity: date = date(2031, 7, 7)      # 만기일 (100% 일시상환)  공시 p.2 §5·7
    conv_start: date = date(2027, 7, 7)    # 전환청구 시작          공시 p.2 §9
    conv_end: date = date(2031, 6, 7)      # 전환청구 종료          공시 p.2 §9
    coupon: float = 0.0                    # 표면·만기이자율 0%     공시 p.2 §4
    recovery: float = 0.40                 # 회수율 40%             논문 Table 1
    sim_notional: float = 5_000_000_000    # 헤지 시뮬레이션 투자액 50억 (액면 기준)


TERMS = CBTerms()


def yf(d0: date, d1: date) -> float:
    """기간(년) = 실제 일수 / 365"""
    return (d1 - d0).days / 365.0


# ---------------------------------------------------------------------------
# 1. 금리
# ---------------------------------------------------------------------------
KTB_TENORS = (1, 2, 3, 5, 10)


def interp_ktb(curve_pct: dict, T: float) -> float:
    """국고 커브(연%) 직선 보간. 논문 2.3절: 알려진 만기 사이를 보간."""
    ks = sorted(curve_pct)
    if not (ks[0] <= T <= ks[-1]):
        raise ValueError(f"T={T:.4f}년이 커브 범위 {ks[0]}~{ks[-1]}년 밖")
    for a, b in zip(ks, ks[1:]):
        if a <= T <= b:
            return curve_pct[a] + (T - a) / (b - a) * (curve_pct[b] - curve_pct[a])
    raise RuntimeError("unreachable")


def semi_to_cont(y_pct: float) -> float:
    """6개월 복리 호가(연%) → 연속복리(소수). 논문에는 없는 보정(비교용)."""
    return 2.0 * math.log(1.0 + y_pct / 200.0)


# ---------------------------------------------------------------------------
# 2. 역사적 변동성
# ---------------------------------------------------------------------------
def hv(close: pd.Series, end: str, n: int, days: int = 252):
    """end 일자까지 n개 로그수익률의 표본표준편차 × √days. close: index=일자(str), 오름차순"""
    idx = list(close.index)
    i = idx.index(end)
    if i < n:
        raise ValueError(f"{end}: 수익률 {n}개 필요, {i}개뿐")
    px = close.iloc[i - n: i + 1].to_numpy(dtype=float)
    lr = np.log(px[1:] / px[:-1])
    return float(lr.std(ddof=1) * math.sqrt(days)), idx[i - n + 1]


# ---------------------------------------------------------------------------
# 3. 트리 (Milanov et al. 2013, 논문 eq.21~35)
# ---------------------------------------------------------------------------
def tree_params(sigma: float, r: float, s: float, dt: float, q: float = 0.0, R: float = TERMS.recovery) -> dict:
    lam = s / (1.0 - R)                                   # eq.47 역산: λ = s/(1−R)
    u = math.exp(sigma * math.sqrt(dt))                   # eq.21
    d = 1.0 / u                                           # eq.22
    g = math.exp((r - q) * dt)                            # q=0이면 논문 원형 e^{rδt}
    pu = (g - math.exp(-lam * dt) * d) / (u - d)          # eq.23
    pd = -(g - math.exp(-lam * dt) * u) / (u - d)         # eq.24
    p0 = 1.0 - math.exp(-lam * dt)                        # eq.25
    if not (pu > 0 and pd > 0):
        raise ValueError(f"확률 음수: pu={pu}, pd={pd} (dt를 줄이거나 입력 확인)")
    return dict(u=u, d=d, pu=pu, pd=pd, p0=p0, lam=lam, disc=math.exp(-r * dt), dt=dt)


def cb_price(S: float, sigma: float, r: float, s: float, val_date: date, q: float = 0.0,
             steps: int = 400, convertible: bool = True, terms: CBTerms = TERMS, full: bool = False):
    """
    무이표 CB 가격(액면 100 기준).
      만기 노드  : 액면 100 (전환청구 종료가 만기 1개월 전이므로 eq.26의 max 대신)
      후진 계산  : V = e^{-rδt}(pu V⁺ + pd V⁻ + p0 X), X = R×액면  (eq.35, eq.28 η=1)
      전환       : 전환청구기간 안의 노드에서만 V = max(V, 액면×S/전환가)  (eq.27)
    full=True  : dict(price, delta48, nodes) — nodes는 칸별 (주가, 값, 전환여부)
    """
    T = yf(val_date, terms.maturity)
    p = tree_params(sigma, r, s, T / steps, q, terms.recovery)
    u, pu, pd, p0, disc, dt = p["u"], p["pu"], p["pd"], p["p0"], p["disc"], p["dt"]
    X = terms.recovery * terms.face
    ts = max(yf(val_date, terms.conv_start), 0.0)
    te = yf(val_date, terms.conv_end)
    cr = terms.face / terms.conv_price                    # 액면 100당 전환주식수

    V = np.full(steps + 1, terms.face)
    nodes = {} if full else None
    if full:
        nodes[steps] = (S * u ** (steps - 2.0 * np.arange(steps + 1)), V.copy(), np.zeros(steps + 1, bool))
    lvl1 = None
    for i in range(steps - 1, -1, -1):
        Si = S * u ** (i - 2.0 * np.arange(i + 1))
        V = disc * (pu * V[:-1] + pd * V[1:] + p0 * X)
        conv = np.zeros(i + 1, bool)
        if convertible and ts - 1e-12 <= i * dt <= te + 1e-12:
            cv = cr * Si
            conv = cv > V
            V = np.where(conv, cv, V)
        if i == 1:
            lvl1 = (Si.copy(), V.copy())
        if full:
            nodes[i] = (Si.copy(), V.copy(), conv.copy())
    if not full:
        return float(V[0])
    S1, V1 = lvl1
    d48 = (V1[0] - V1[1]) / (S1[0] - S1[1])              # eq.48: ∂CB/∂S (액면 100당, 주가 1원당)
    return dict(price=float(V[0]), delta48=float(d48 / cr), nodes=nodes, params=p)


def delta48(S, sigma, r, s, val_date, q=0.0, steps=400, terms=TERMS) -> float:
    """eq.48 delta를 전환주식수 대비 비율로 (논문 hedge ratio δ = Cr × Δ의 Δ)"""
    return cb_price(S, sigma, r, s, val_date, q, steps, terms=terms, full=True)["delta48"]


def tf_price(S, sigma, r, s, val_date, q=0.0, steps=400, terms: CBTerms = TERMS) -> float:
    """Tsiveriotis–Fernandes(1998) 비교 모형. 주식 부분은 r, 현금 부분은 r+s로 할인 (논문 Table 1의 TF 가격)."""
    T = yf(val_date, terms.maturity)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt)); d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    ts = max(yf(val_date, terms.conv_start), 0.0); te = yf(val_date, terms.conv_end)
    cr = terms.face / terms.conv_price
    E = np.zeros(steps + 1); B = np.full(steps + 1, terms.face)
    for i in range(steps - 1, -1, -1):
        Si = S * u ** (i - 2.0 * np.arange(i + 1))
        E = math.exp(-r * dt) * (p * E[:-1] + (1 - p) * E[1:])
        B = math.exp(-(r + s) * dt) * (p * B[:-1] + (1 - p) * B[1:])
        if ts - 1e-12 <= i * dt <= te + 1e-12:
            cv = cr * Si; ex = cv > E + B
            E = np.where(ex, cv, E); B = np.where(ex, 0.0, B)
    return float(E[0] + B[0])


# ---------------------------------------------------------------------------
# 4. 채권가치
# ---------------------------------------------------------------------------
def bf_eq4(r: float, s: float, T: float, face: float = 100.0) -> float:
    """논문 eq.4 (무이표): 액면 × e^{-(r+s)T}. 논문 p.19: '발행일 커브 + 합성 스프레드'로 할인."""
    return face * math.exp(-(r + s) * T)


def bf_recovery(r: float, s: float, T: float, R: float = TERMS.recovery, face: float = 100.0) -> float:
    """트리에서 전환권을 뺀 값의 닫힌 식: 생존 시 액면 + 부도 시 R×액면 (λ = s/(1−R))"""
    lam = s / (1 - R); k = r + lam
    return face * math.exp(-k * T) + R * face * lam / k * (1 - math.exp(-k * T))


# ---------------------------------------------------------------------------
# 5. 역산 변동성 (이분법)
# ---------------------------------------------------------------------------
def implied_vol(price_fn, target: float, lo: float = 0.01, hi: float = 1.50, tol: float = 1e-8, max_iter: int = 100):
    """price_fn(σ)=target 인 σ. 가격은 σ에 대해 증가함수. 반환: (σ, 반복 기록 DataFrame)"""
    flo, fhi = price_fn(lo) - target, price_fn(hi) - target
    if flo > 0 or fhi < 0:
        raise ValueError(f"해 없음: f(lo)={flo:.6f}, f(hi)={fhi:.6f}")
    log = []
    for k in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f = price_fn(mid) - target
        log.append((k, lo, hi, mid, f))
        if abs(f) < tol:
            break
        if f > 0:
            hi = mid
        else:
            lo = mid
    if abs(f) >= tol:
        raise RuntimeError(f"수렴 실패: {max_iter}회 후 가격−목표 {f:.3e}")
    return mid, pd.DataFrame(log, columns=["반복", "하한σ", "상한σ", "중간σ", "가격−목표"])


# ---------------------------------------------------------------------------
# 5-1. 일반적인 방식: 채권가치 + BSM 콜 (실무 비교용 · 논문 방식 아님)
# ---------------------------------------------------------------------------
def norm_cdf(x: float) -> float:
    """표준정규 누적분포 (scipy 없이 math.erf 사용)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bsm_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """블랙-숄즈-머튼 유럽형 콜 (연속 배당수익률 q)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bsm_cb_price(S: float, sigma: float, r: float, s: float, val_date: date, q: float = 0.0, terms: CBTerms = TERMS) -> float:
    """CB = 채권가치(eq.4와 같이 r+s로 할인) + 액면 100당 전환주식수 × BSM 콜.
    전환권을 만기 행사 유럽형 콜로 보고, 부도는 채권 할인율(스프레드)에만 반영한다.
    트리와 달리 부도 경로 · 전환청구기간을 따로 다루지 않는 단순한 방식으로, 논문 방식과의 비교용이다."""
    T = yf(val_date, terms.maturity)
    return bf_eq4(r, s, T, terms.face) + terms.face / terms.conv_price * bsm_call(S, terms.conv_price, T, r, q, sigma)


# ---------------------------------------------------------------------------
# 6. 이표 CB (논문 Table 1 재현용: 반기 이표, 전 기간 American, eq.26·36·37)
# ---------------------------------------------------------------------------
def cb_price_coupon(S, Cr, cpn, T, sigma, r, lam, R=0.40, steps=400, convertible=True, freq=2) -> float:
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt)); d = 1 / u
    pu = (math.exp(r * dt) - math.exp(-lam * dt) * d) / (u - d)
    pd = -(math.exp(r * dt) - math.exp(-lam * dt) * u) / (u - d)
    p0 = 1 - math.exp(-lam * dt)
    if not (pu > 0 and pd > 0):
        raise ValueError(f"확률 음수: pu={pu}, pd={pd} (σ가 너무 작거나 dt가 큼)")
    N = 100.0; c = N * cpn / freq; X = R * N; disc = math.exp(-r * dt)
    ctimes = np.arange(1, int(round(T * freq)) + 1) / freq
    ST = S * u ** (steps - 2.0 * np.arange(steps + 1))
    V = np.maximum(Cr * ST, N + c) if convertible else np.full(steps + 1, N + c)      # eq.26
    for i in range(steps - 1, -1, -1):
        t = i * dt; Si = S * u ** (i - 2.0 * np.arange(i + 1))
        V = disc * (pu * V[:-1] + pd * V[1:] + p0 * X)                                  # eq.35
        for tc in ctimes[:-1]:                                                         # 마지막 이표는 만기 노드에 포함
            if t < tc <= t + dt + 1e-12:
                V = V + c * math.exp(-r * (tc - t) - lam * dt)                          # eq.37 원문: c·e^{-r(t_i−t)−λδt}
        if convertible:
            V = np.maximum(V, Cr * Si)                                                  # eq.27
    return float(V[0])


def bf_eq4_coupon(cpn, T, y, freq=2, face=100.0) -> float:
    ts = np.arange(1, int(round(T * freq)) + 1) / freq
    return float(sum(face * cpn / freq * math.exp(-y * t) for t in ts) + face * math.exp(-y * T))


def tf_price_coupon(S, Cr, cpn, T, sigma, r, s, steps=400, freq=2) -> float:
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt)); d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    N = 100.0; c = N * cpn / freq
    ctimes = np.arange(1, int(round(T * freq)) + 1) / freq
    ST = S * u ** (steps - 2.0 * np.arange(steps + 1)); conv = Cr * ST
    E = np.where(conv >= N + c, conv, 0.0); B = np.where(conv >= N + c, 0.0, N + c)
    for i in range(steps - 1, -1, -1):
        t = i * dt; Si = S * u ** (i - 2.0 * np.arange(i + 1))
        E = math.exp(-r * dt) * (p * E[:-1] + (1 - p) * E[1:])
        B = math.exp(-(r + s) * dt) * (p * B[:-1] + (1 - p) * B[1:])
        for tc in ctimes[:-1]:
            if t < tc <= t + dt + 1e-12:
                B = B + c * math.exp(-(r + s) * (tc - t))
        cv = Cr * Si; ex = cv > E + B
        E = np.where(ex, cv, E); B = np.where(ex, 0.0, B)
    return float(E[0] + B[0])


# ---------------------------------------------------------------------------
# 7. 점검표
# ---------------------------------------------------------------------------
class Checker:
    """check(이름, 계산값, 기대값, 허용오차, 근거) → ✓/✗ 출력·기록. kind='info'는 합격 판정 없이 기록만.
    결과는 frame()/save()로 집계 (셀 출력 간결화를 위해 반환값 없음)."""

    def __init__(self, notebook: str):
        self.nb = notebook
        self.rows = []

    def check(self, name, value, expected=None, tol=0.0, basis="", kind="test"):
        if kind == "info":
            ok, mark = None, "ℹ"
        elif isinstance(expected, bool) or expected is None:
            ok = bool(value) if expected is None else (bool(value) == expected)
            mark = "✓" if ok else "✗"
        elif isinstance(expected, (int, float)) and isinstance(value, (int, float, np.floating, np.integer)):
            ok = abs(float(value) - float(expected)) <= tol
            mark = "✓" if ok else "✗"
        else:
            ok = value == expected
            mark = "✓" if ok else "✗"
        self.rows.append(dict(노트북=self.nb, 항목=name, 계산값=value, 기대값=expected, 허용오차=tol,
                              결과=mark, 근거=basis))
        exp_txt = "" if expected is None else f" | 기대 {expected}" + (f" ±{tol}" if tol else "")
        print(f"{mark} {name}: {value}{exp_txt}" + (f"  ({basis})" if basis else ""))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path: Path):
        df = self.frame()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        n_fail = int((df["결과"] == "✗").sum())
        print(f"\n점검 {len(df)}건: ✓ {int((df['결과']=='✓').sum())} · ✗ {n_fail} · ℹ {int((df['결과']=='ℹ').sum())} → {path.name}")
        return n_fail
