"""The bench's own records (`fsla_bench.results`) on a temporary store root: the fleet's sink and completion reader
(`fsla_bench.fleet.jobs.fleet_store`) and the local grid's store (`fsla_bench.mqar.local_grid.grid_store`) write at their
targets' labels, the cell named in the semantics; the readers (`fsla_bench.mqar.analysis.graph._load_rows`, and
`fsla_bench.mqar.winners.pick_winners`, the winners `seed_confirm` re-runs) read both labels' samples of one bench and
config, the newest per run id. `python -m unittest tests.results.test_records`"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fsla_devtools.results import Store

from fsla_bench.fleet.jobs import fleet_store
from fsla_bench.mqar.analysis.graph import _load_rows
from fsla_bench.mqar.local_grid import grid_store
from fsla_bench.mqar.winners import pick_winners
from fsla_bench.results import FLEET, LOCAL_GRID, completed

RID = "grid-routed-dense-nc64_L128_st100_lr1e-3_s0-base"
CODE = {"fsla_bench": {"git_sha": "a" * 40, "diff_sha256": None}, "fsla": {"git_sha": "b" * 40, "diff_sha256": None}}


class Records(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "records"

    def test_the_fleet_stores_each_pulled_row_once_at_its_label_and_a_cell_is_done_for_its_image(self):
        store_result, done_ids = fleet_store("mqar", "canonical", "img:1", "mqar-canonical", root=self.root)
        row = {"run_id": RID, "ok": True, "max_acc": 0.9}
        store_result(row)
        store_result(row)  #: pulled again: the same row is stored once
        store_result({"run_id": "grid-mha-x", "ok": False, "error": "OOM"})
        samples = Store(FLEET, self.root).samples()
        self.assertEqual(sorted((s["semantics"]["cell"], s["ok"]) for s in samples), [("grid-mha-x", False), (RID, True)])
        self.assertEqual({s["label"] for s in samples}, {FLEET})
        self.assertEqual(samples[0]["semantics"].keys(), {"bench", "config", "cell", "image"})
        self.assertEqual(done_ids(), {RID})  #: a failed cell is not done
        _store, other_image = fleet_store("mqar", "canonical", "img:2", "mqar-canonical", root=self.root)
        self.assertEqual(other_image(), set())  #: an image bump runs the grid again

    def test_the_local_grid_stores_at_its_label_and_a_cell_is_done_for_its_code(self):
        store, semantics = grid_store("canonical", CODE, root=self.root)
        self.assertEqual(store.label, LOCAL_GRID)
        self.assertFalse(completed(store, semantics(RID)))
        store.put(semantics(RID), error="the cell reported ok=false")
        self.assertFalse(completed(store, semantics(RID)))
        store.put(semantics(RID), output={"run_id": RID, "ok": True, "max_acc": 0.5})
        self.assertTrue(completed(store, semantics(RID)))
        _other, moved = grid_store("canonical", {**CODE, "fsla": {"git_sha": "c" * 40, "diff_sha256": None}},
                                   root=self.root)
        self.assertFalse(completed(store, moved(RID)))  #: other code: not done

    def plant(self):
        fleet_result, _done = fleet_store("mqar", "graph_grid", "img:1", "j", root=self.root)
        fleet_result({"run_id": RID, "ok": True, "max_acc": 0.25})
        store, semantics = grid_store("graph_grid", CODE, root=self.root)
        store.put(semantics(RID), output={"run_id": RID, "ok": True, "max_acc": 0.75})  #: newer: it is the one read
        other, other_semantics = grid_store("canonical", CODE, root=self.root)
        other.put(other_semantics("grid-mha-y"), output={"run_id": "grid-mha-y", "ok": True, "max_acc": 0.5})

    def test_the_graph_reads_both_labels_of_its_config_the_newest_per_run_id(self):
        self.plant()
        self.assertEqual(_load_rows("graph_grid", self.root), [{"run_id": RID, "ok": True, "max_acc": 0.75}])
        self.assertEqual(_load_rows("presence_grid", self.root), [])

    def test_the_seed_confirmation_reads_every_mqar_config(self):
        self.plant()
        self.assertEqual(pick_winners(root=self.root), {RID, "grid-mha-y"})
        self.assertEqual(pick_winners({"grid-mha-y"}, root=self.root), {"grid-mha-y"})


if __name__ == "__main__":
    unittest.main()
