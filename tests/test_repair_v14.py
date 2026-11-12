"""Checks for repair_selection_v14."""
import os, sys
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from safe_motion.data import asdpose as D
from safe_motion import eventlog


def test_cleaned_validation_excludes_alias():
    y = yaml.safe_load(open("configs/protocols/repair_selection_v14.yaml"))
    assert y["val_exclude_ids"] == [322]
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    tr, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    assert any(r["child_id"] == 196 for r in tr) and any(r["child_id"] == 322 for r in va)
    keep = np.array([int(r["child_id"]) not in set(y["val_exclude_ids"]) for r in va])
    assert (~keep).sum() == sum(r["child_id"] == 322 for r in va) == 19
    print("ok test_cleaned_validation_excludes_alias")


def test_eventlog_draws_nothing():
    st = np.random.get_state()[1].copy()
    os.environ["SM_EVENT_LOG"] = "/tmp/_sm_event_test.log"
    eventlog.log_event("test", a=1)
    del os.environ["SM_EVENT_LOG"]
    assert (np.random.get_state()[1] == st).all()
    print("ok test_eventlog_draws_nothing")


if __name__ == "__main__":
    test_cleaned_validation_excludes_alias()
    test_eventlog_draws_nothing()
