"""Mini NPU Simulator - MAC(Multiply-Accumulate) 연산으로 Cross/X 패턴을 판정한다.

실행: python main.py
외부 라이브러리 없이 표준 라이브러리(json, time, os, re)만 사용한다.
"""

import json
import os
import re
import time

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
EPSILON = 1e-9          # 점수 비교 허용오차: |a-b| < EPSILON 이면 동점
REPEAT = 10             # 성능 측정 반복 횟수
CROSS, X, UNDECIDED = "Cross", "X", "UNDECIDED"
LINE = "#" + "-" * 40

# 다양한 표기를 표준 라벨(Cross / X)로 바꾸는 표
LABEL_MAP = {
    "+": CROSS, "cross": CROSS, "plus": CROSS, "십자가": CROSS,
    "x": X, "×": X, "cross_x": X,
}


# ------------------------------------------------------------------
# 1. 데이터 구조 / 라벨 정규화
# ------------------------------------------------------------------
class Matrix:
    """n×n 2차원 배열. 특정 위치 값 저장/읽기 기능을 제공한다."""

    def __init__(self, rows):
        self.rows = [[float(v) for v in row] for row in rows]
        self.n = len(self.rows)

    @classmethod
    def zeros(cls, n):
        return cls([[0.0] * n for _ in range(n)])

    def get(self, r, c):
        return self.rows[r][c]

    def set(self, r, c, value):
        self.rows[r][c] = float(value)

    def is_square(self):
        return self.n > 0 and all(len(row) == self.n for row in self.rows)

    def shape(self):
        cols = {len(row) for row in self.rows}
        col_text = str(cols.pop()) if len(cols) == 1 else "가변"
        return f"{self.n}×{col_text}"


def normalize_label(raw):
    """'+', 'cross', 'x', 'X' 등을 표준 라벨(Cross/X)로 바꾼다. 모르는 값은 None."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    return None


# ------------------------------------------------------------------
# 2. MAC 연산 / 판정
# ------------------------------------------------------------------
def mac(pattern, flt):
    """같은 위치끼리 곱한 뒤 모두 더한다 (반복문 직접 구현)."""
    n = pattern.n
    total = 0.0
    for r in range(n):
        p_row = pattern.rows[r]
        f_row = flt.rows[r]
        for c in range(n):
            total += p_row[c] * f_row[c]
    return total


def mac_flat(p_flat, f_flat):
    """[보너스] 1차원(길이 N²) 배열 버전 MAC."""
    total = 0.0
    for i in range(len(p_flat)):
        total += p_flat[i] * f_flat[i]
    return total


def flatten(matrix):
    return [v for row in matrix.rows for v in row]


def decide(score_a, score_b, label_a, label_b):
    """epsilon 기반 비교. 차이가 EPSILON 미만이면 UNDECIDED."""
    if abs(score_a - score_b) < EPSILON:
        return UNDECIDED
    return label_a if score_a > score_b else label_b


def measure_ms(func, *args, repeat=REPEAT):
    """연산 함수 호출 구간만 repeat 회 측정해 평균(ms)을 돌려준다."""
    start = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    return (time.perf_counter() - start) * 1000 / repeat


# ------------------------------------------------------------------
# 3. 패턴 생성기 (보너스)
# ------------------------------------------------------------------
def make_cross(n):
    mid = n // 2
    return Matrix([[1 if (r == mid or c == mid) else 0 for c in range(n)] for r in range(n)])


def make_x(n):
    return Matrix([[1 if (r == c or r + c == n - 1) else 0 for c in range(n)] for r in range(n)])


# ------------------------------------------------------------------
# 4. 출력 도우미
# ------------------------------------------------------------------
def header(title):
    print()
    print(LINE)
    print(f"# {title}")
    print(LINE)


def fmt(score):
    return repr(score)  # 부동소수점 원래 값을 그대로 보여준다


def print_performance(rows):
    print(f"{'크기':<10}{'평균 시간(ms)':>14}{'연산 횟수(N²)':>14}")
    print("-" * 40)
    for n, ms in rows:
        print(f"{f'{n}×{n}':<10}{ms:>14.4f}{n * n:>14}")


def performance_rows(sizes):
    """크기별로 Cross 패턴 × Cross 필터 MAC 평균 시간을 잰다."""
    rows = []
    for n in sizes:
        pattern, flt = make_cross(n), make_cross(n)
        rows.append((n, measure_ms(mac, pattern, flt)))
    return rows


# ------------------------------------------------------------------
# 5. 모드 1: 사용자 입력 (3×3)
# ------------------------------------------------------------------
def read_line(prompt=""):
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 프로그램을 종료합니다.")
        raise SystemExit(0)


def read_matrix(title, n=3):
    """n줄을 공백 구분 숫자로 입력받는다. 형식 오류면 처음부터 다시 받는다."""
    while True:
        print(f"{title} ({n}줄 입력, 공백 구분)")
        rows = []
        ok = True
        for i in range(n):
            parts = read_line().split()
            if len(parts) != n:
                print(f"입력 형식 오류: {i + 1}번째 줄 - 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요.")
                ok = False
                break
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                print(f"입력 형식 오류: {i + 1}번째 줄 - 숫자만 입력하세요. (입력값: {' '.join(parts)})")
                ok = False
                break
        if ok:
            return Matrix(rows)
        print("다시 입력해 주세요.\n")


def mode_user_input():
    header("[1] 필터 입력")
    filter_a = read_matrix("필터 A")
    print()
    filter_b = read_matrix("필터 B")
    print("\n✓ 필터 A, B 저장 완료")

    header("[2] 패턴 입력")
    pattern = read_matrix("패턴")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    elapsed = measure_ms(mac, pattern, filter_a)
    result = decide(score_a, score_b, "A", "B")

    header("[3] MAC 결과")
    print(f"A 점수: {fmt(score_a)}")
    print(f"B 점수: {fmt(score_b)}")
    print(f"연산 시간(평균/{REPEAT}회): {elapsed:.4f} ms")
    if result == UNDECIDED:
        print(f"판정: 판정 불가 (|A-B| < {EPSILON})")
    else:
        print(f"판정: {result}")

    header(f"[4] 성능 분석 (평균/{REPEAT}회)")
    print_performance([(3, elapsed)])


# ------------------------------------------------------------------
# 6. 모드 2: data.json 분석
# ------------------------------------------------------------------
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ data.json 파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError as e:
        print(f"✗ data.json 형식 오류: {e}")
    return None


def size_from_key(key):
    """'size_13' 또는 'size_13_2' 에서 13 을 꺼낸다."""
    m = re.match(r"^size_(\d+)(?:_|$)", str(key).strip())
    return int(m.group(1)) if m else None


def load_filters(data):
    """filters 에서 크기별 {Cross: Matrix, X: Matrix} 를 만든다."""
    header("[1] 필터 로드")
    filters = {}
    raw_filters = data.get("filters", {}) if isinstance(data, dict) else {}
    if not isinstance(raw_filters, dict) or not raw_filters:
        print("✗ filters 항목이 없습니다.")
        return filters
    for key, group in raw_filters.items():
        n = size_from_key(key)
        if n is None or not isinstance(group, dict):
            print(f"✗ {key}: 필터 키/구조를 해석할 수 없어 건너뜁니다.")
            continue
        pair = {}
        for label_key, values in group.items():
            label = normalize_label(label_key)
            if label is None:
                print(f"✗ {key}.{label_key}: 알 수 없는 필터 라벨")
                continue
            try:
                matrix = Matrix(values)
            except (TypeError, ValueError):
                print(f"✗ {key}.{label_key}: 숫자 배열이 아닙니다.")
                continue
            if not matrix.is_square() or matrix.n != n:
                print(f"✗ {key}.{label_key}: 크기 불일치 (키 {n}×{n}, 실제 {matrix.shape()})")
                continue
            pair[label] = matrix
        if CROSS in pair and X in pair:
            filters[n] = pair
            print(f"✓ {key:<8} 필터 로드 완료 (Cross, X)")
        else:
            missing = [lb for lb in (CROSS, X) if lb not in pair]
            print(f"✗ {key}: {', '.join(missing)} 필터가 없어 사용할 수 없습니다.")
    return filters


def analyze_pattern(key, item, filters):
    """패턴 1개를 검증·판정한다. (pass 여부, 실패 사유) 를 돌려준다."""
    print(f"--- {key} ---")
    n = size_from_key(key)
    if n is None:
        return False, "키에서 크기(N)를 추출할 수 없음"
    if not isinstance(item, dict) or "input" not in item:
        return False, "input 항목 없음 (스키마 오류)"
    expected = normalize_label(item.get("expected"))
    if expected is None:
        return False, f"expected 라벨을 해석할 수 없음 ({item.get('expected')!r})"
    if n not in filters:
        return False, f"size_{n} 필터가 없음"
    try:
        pattern = Matrix(item["input"])
    except (TypeError, ValueError):
        return False, "input 이 숫자 2차원 배열이 아님"
    if not pattern.is_square() or pattern.n != n:
        return False, f"크기 불일치 (필터 {n}×{n}, 패턴 {pattern.shape()})"

    cross_score = mac(pattern, filters[n][CROSS])
    x_score = mac(pattern, filters[n][X])
    result = decide(cross_score, x_score, CROSS, X)
    passed = result == expected
    print(f"Cross 점수: {fmt(cross_score)}")
    print(f"X 점수: {fmt(x_score)}")
    status = "PASS" if passed else "FAIL"
    note = " (동점 규칙)" if result == UNDECIDED else ""
    print(f"판정: {result} | expected: {expected} | {status}{note}")
    if passed:
        return True, ""
    if result == UNDECIDED:
        return False, f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| < {EPSILON})"
    return False, f"판정 {result} ≠ expected {expected}"


def mode_json():
    data = load_json(DATA_FILE)
    if data is None:
        return
    filters = load_filters(data)

    header("[2] 패턴 분석 (라벨 정규화 적용)")
    patterns = data.get("patterns", {}) if isinstance(data, dict) else {}
    if not isinstance(patterns, dict):
        print("✗ patterns 항목 구조 오류")
        patterns = {}
    passed, failures = 0, []
    for key, item in patterns.items():
        try:
            ok, reason = analyze_pattern(key, item, filters)
        except Exception as e:  # 예상 못 한 오류도 케이스 단위 FAIL 처리
            ok, reason = False, f"처리 중 오류: {type(e).__name__}: {e}"
        if ok:
            passed += 1
        else:
            failures.append((key, reason))
            if "동점" not in reason and "≠" not in reason:
                print(f"FAIL: {reason}")

    header(f"[3] 성능 분석 (평균/{REPEAT}회)")
    sizes = [3] + sorted(filters.keys() or [5, 13, 25])
    print_performance(performance_rows(sizes))

    header("[4] 결과 요약")
    total = len(patterns)
    print(f"총 테스트: {total}개")
    print(f"통과: {passed}개")
    print(f"실패: {len(failures)}개")
    if failures:
        print("\n실패 케이스:")
        for key, reason in failures:
            print(f"- {key}: {reason}")
    print('\n(상세 원인 분석 및 복잡도 설명은 README.md 의 "결과 리포트" 섹션 참고)')


# ------------------------------------------------------------------
# 7. 보너스: 1차원 배열 최적화 비교
# ------------------------------------------------------------------
def mode_bonus():
    header("[보너스] 2차원 vs 1차원(N²) MAC 성능 비교 (평균/100회)")
    print(f"{'크기':<8}{'2D(ms)':>12}{'1D(ms)':>12}{'개선율':>10}")
    print("-" * 42)
    for n in (3, 5, 13, 25, 51):
        p, f = make_cross(n), make_x(n)
        pf, ff = flatten(p), flatten(f)
        t2 = measure_ms(mac, p, f, repeat=100)
        t1 = measure_ms(mac_flat, pf, ff, repeat=100)
        assert abs(mac(p, f) - mac_flat(pf, ff)) < EPSILON
        print(f"{f'{n}×{n}':<8}{t2:>12.4f}{t1:>12.4f}{(1 - t1 / t2) * 100:>9.1f}%")
    print("\n패턴 생성기 예시 (5×5 Cross / X)")
    for a, b in zip(make_cross(5).rows, make_x(5).rows):
        print(" ".join(str(int(v)) for v in a), "   ", " ".join(str(int(v)) for v in b))


# ------------------------------------------------------------------
def main():
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]")
    print("1. 사용자 입력 (3x3)")
    print("2. data.json 분석")
    print("3. (보너스) 1차원 최적화 비교 / 패턴 생성기")
    while True:
        choice = read_line("선택: ").strip()
        if choice in ("1", "2", "3"):
            break
        print("1, 2, 3 중 하나를 입력하세요.")
    {"1": mode_user_input, "2": mode_json, "3": mode_bonus}[choice]()


if __name__ == "__main__":
    main()
