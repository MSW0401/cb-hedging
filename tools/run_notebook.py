"""간이 노트북 실행기 (검증용): 코드 셀을 순서대로 실행하고 출력을 .ipynb에 기록. Jupyter 미설치 환경용."""
import ast, base64, io, json, os, sys, time, traceback, contextlib
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run(path):
    path = Path(path).resolve(); os.chdir(path.parent); sys.path.insert(0, str(path.parent))
    nb = json.loads(path.read_text(encoding="utf-8")); ns = {"__name__": "__main__"}; n = 0
    outs_holder = []; state = {}
    def display(obj):
        b = state.get("buf")
        if b is not None and b.getvalue():
            outs_holder.extend(_stream(b.getvalue())); b.seek(0); b.truncate(0)   # print 순서 보존
        outs_holder.append(_rich(obj))
    ns["display"] = display
    t0 = time.time()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code": continue
        n += 1; src = "".join(cell["source"]); outs_holder.clear(); buf = io.StringIO(); state["buf"] = buf
        try:
            tree = ast.parse(src); last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {n}>", "exec"), ns)
                res = eval(compile(last, f"<cell {n}>", "eval"), ns) if last else None
        except Exception:
            cell["outputs"] = _stream(buf.getvalue()) + [{"output_type": "error", "ename": "Error", "evalue": "", "traceback": traceback.format_exc().splitlines()}]
            cell["execution_count"] = n
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{path.name}] 셀 {n} 오류\n{src[:300]}\n" + traceback.format_exc()); return False
        outs = list(outs_holder) + _stream(buf.getvalue())
        for num in plt.get_fignums():
            fig = plt.figure(num); b = io.BytesIO(); fig.savefig(b, format="png", dpi=fig.dpi, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
        plt.close("all")
        if res is not None:
            r = _rich(res); r["output_type"] = "execute_result"; r["execution_count"] = n; outs.append(r)
        cell["outputs"] = outs; cell["execution_count"] = n
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{path.name}] 코드 셀 {n}개 정상 실행 ({time.time()-t0:.1f}초)"); return True

def _stream(t):
    return [{"output_type": "stream", "name": "stdout", "text": t}] if t else []

def _rich(obj):
    data = {"text/plain": repr(obj)}
    if hasattr(obj, "to_html"):
        try: data["text/html"] = obj.to_html()
        except Exception: pass
    return {"output_type": "display_data", "metadata": {}, "data": data}

if __name__ == "__main__":
    ps = [os.path.abspath(p) for p in sys.argv[1:]]; ok = all(run(p) for p in ps); sys.exit(0 if ok else 1)
