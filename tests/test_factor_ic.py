# tests/test_factor_ic.py
import subprocess, os, csv

REPO = "/Users/wangbo/VSCodeProjects/Stock"

def test_factor_ic_csv_written():
    out = os.path.join(REPO, "results", "factor_ic.csv")
    if os.path.exists(out):
        os.remove(out)
    subprocess.run(["python3", "compute_factor_ic.py"], cwd=REPO, check=True)
    assert os.path.exists(out), "compute_factor_ic.py 应写 results/factor_ic.csv"
    with open(out) as f:
        header = next(csv.reader(f))
    assert header == ["date", "factor", "ic", "n"]
    # 至少含 fund_flow_rank 与 composite 两行
    txt = open(out).read()
    assert "fund_flow_rank" in txt and "composite" in txt
