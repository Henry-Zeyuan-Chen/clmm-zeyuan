import runpy
import sys
from pathlib import Path
import pytest


def test_actual_slurm_time_bound(monkeypatch,tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/format_training_pipeline.py'
    monkeypatch.setattr(sys,'argv',[str(path),'--base',str(tmp_path)])
    parse=runpy.run_path(str(path),run_name='resource_test')['slurm_time_seconds']
    for value,expected in [('2:00:00',7200),('04:00:00\n',14400),('59:59',3599),('0:04',4),('1-02:03:04',93784)]:
        assert parse(value)==expected
    for value in ['UNLIMITED','N/A','Unknown','1:60:00','1:01:60','1-2-3']:
        with pytest.raises(ValueError):parse(value)
