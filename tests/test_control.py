"""漂移与决策：时间回放、自适应选版、证据校验和回滚。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gas_a import adaptive
from gas_a.cloud_records import Audit
from gas_a.controller_policy import (
    build_planning_prompt,
    build_selection_prompt,
    class_metrics,
    measured_evidence,
    planning_contract,
    protect_decision,
    stopping_evidence,
    validate_plan,
)
from gas_a.data import evaluate, load_model, load_pair, score_predictions
from gas_a.io import CLASSES, digest, history, read_json, write_json
from gas_a.monitor import diagnose, monitor
from gas_a.optimize import fit_weighted, train_candidate
from gas_a.pipeline import ordered_specs
from fixtures import legacy_config as load_config
from gas_a.registry import Registry
from gas_a.temporal import (
    dat_inputs,
    export_temporal,
    optimize_window,
    performance_rollback,
    reveal_labels,
    run_temporal,
    window_gate,
)

from fixtures import ReplayModelFixture, fixture_plan

ROOT = Path(__file__).resolve().parents[1]


class TemporalTests(unittest.TestCase):
    def config(self, root):
        cfg = load_config(ROOT / "configs/a.live.json")
        cfg["runtime"] = str(root / "result")
        cfg["experiments"] = [{**cfg["experiments"][0], "epochs": 1}]
        cfg["optimization"]["max_selected_experiments"] = 1
        return cfg

    def make_windows(self, directory):
        cfg = load_config(ROOT / "configs/a.live.json")
        train, validation = load_pair(cfg["train"], cfg["validation"])
        for batch, start in ((3, 0), (4, 40)):
            lines = []
            for x, y in zip(validation["x"][start:start+40], validation["y"][start:start+40]):
                lines.append(str(int(y) + 1) + " " + " ".join(f"{i+1}:{v:.17g}" for i, v in enumerate(x)))
            (directory / f"batch{batch}.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return train, validation

    def test_features_do_not_parse_labels_and_reveal_checks_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, _ = self.make_windows(root)
            path = root / "batch3.dat"
            raw = path.read_text(encoding="utf-8")
            path.write_text("not-a-label " + raw.split(" ", 1)[1], encoding="utf-8")
            features = dat_inputs(path, train["paths"])
            self.assertEqual(features["x"].shape, (40, 128))
            self.assertNotIn("y", features)
            with self.assertRaises(ValueError):
                reveal_labels(path, features)
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed after prediction"):
                reveal_labels(path, features)

    def test_frozen_scores_equal_existing_evaluation_and_validate_labels(self):
        cfg = load_config(ROOT / "configs/a.live.json")
        train, val = load_pair(cfg["train"], cfg["validation"])
        model = load_model(cfg["baseline"])
        frozen = score_predictions(model.predict(val["x"]), val["y"])
        existing = evaluate(model, val)
        self.assertEqual(frozen["confusion_matrix"], existing["confusion_matrix"])
        self.assertEqual(frozen["macro_f1"], existing["macro_f1"])
        with self.assertRaises(ValueError):
            score_predictions(model.predict(val["x"]), np.full(len(val["y"]), 6))

    def test_full_replay_commits_before_labels_and_never_trains_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, validation = self.make_windows(root)
            cfg = self.config(root)
            client = ReplayModelFixture()
            reveals = []

            def checked_reveal(path, features):
                batch = int(Path(path).stem.replace("batch", ""))
                window = Path(cfg["runtime"]) / "windows" / f"batch{batch:02d}"
                commit = read_json(window / "prediction_commit.json")
                self.assertFalse(commit["labels_revealed"])
                self.assertEqual(digest(window / "predictions.npz"), commit["predictions_sha256"])
                reveals.append(batch)
                return reveal_labels(path, features)

            with patch("gas_a.temporal.reveal_labels", side_effect=checked_reveal):
                summary = run_temporal(cfg, root, (3, 4), cloud=False, client=client)
            output = Path(cfg["runtime"])
            self.assertEqual(reveals, [3, 4])
            self.assertEqual(len(client.requests), 2)
            selection = client.requests[-1]
            self.assertIn("current_active_artifact_id", selection)
            self.assertTrue(all("new_candidate_artifact_id" in trial for trial in selection["experiments"]))
            self.assertEqual(summary["finished_windows"], 2)
            self.assertEqual(summary["costs"]["calls"], 2)
            self.assertEqual(summary["costs"]["tokens_known"], 20)
            self.assertEqual(summary["costs"]["revealed_public_labels"], 80)
            self.assertTrue(all(read_json(output / "verification.json").values()))
            rows = history(output / "history.jsonl")
            self.assertEqual(rows[-1]["status"], "terminal_evaluation")
            self.assertFalse(rows[-1]["experiments"])
            for trial in rows[0]["experiments"]:
                self.assertEqual(trial["report"]["train_sha256"], train["sha256"])
                self.assertEqual(trial["report"]["validation_sha256"], validation["sha256"])
                self.assertNotEqual(trial["report"]["validation_sha256"], rows[0]["prediction_commit"]["source_sha256"])
                resumed = load_model(Path(trial["evidence_directory"]) / "checkpoint.npz")
                parent = load_model(cfg["baseline"])
                np.testing.assert_array_equal(resumed.mean, parent.mean)
                np.testing.assert_array_equal(resumed.scale, parent.scale)
                self.assertGreater(resumed.meta["steps"], parent.meta["steps"])
            self.assertTrue(read_json(output / "rollback_drill/result.json")["actual_deployment_unchanged"])
            with self.assertRaisesRegex(ValueError, "fresh runtime"):
                run_temporal(cfg, root, (3, 4), cloud=False, client=client)
            retry = {**cfg, "runtime": str(root / "retry"), "prior_attempts": [str(output)]}
            retried = run_temporal(retry, root, (3, 4), cloud=False, client=ReplayModelFixture())
            self.assertTrue(retried["success"])
            self.assertEqual(retried["costs"]["calls"], 4)
            self.assertEqual(retried["costs"]["tokens_known"], 40)

    def test_invalid_controller_recovers_to_observe_with_errors_still_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_windows(root)
            cfg = self.config(root)
            summary = run_temporal(cfg, root, (3, 4), cloud=False, client=ReplayModelFixture(invalid=True))
            self.assertEqual(summary["controller_failed_windows"], 0)
            self.assertEqual(summary["controller_recovered_windows"], 1)
            self.assertFalse(summary["all_model_decisions_valid_without_recovery"])
            self.assertEqual(summary["promotions"], 0)
            self.assertTrue(summary["cloud"]["session_succeeded"])
            self.assertTrue(summary["success"])
            self.assertEqual(summary["costs"]["calls"], 2)

    def test_performance_rollback_compares_same_window_and_keeps_safe_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, validation = self.make_windows(root)
            cfg = self.config(root)
            registry = Registry(root / "registry")
            registry.initialize(cfg["baseline"], {"macro_f1": .99})
            model = load_model(cfg["baseline"])
            cp = root / "copy.npz"
            model.save(cp)
            registry.register("identical", cp, "baseline", {"macro_f1": 1.0}, {})
            registry.activate("identical", "fixture")
            metric = evaluate(model, validation)
            policy = {**cfg["gate"], "minimum_class_recall": {}}
            self.assertIsNone(performance_rollback(registry, validation, metric, metric, validation, policy))
            bad = copy.deepcopy(metric)
            bad["macro_f1"] = 0
            decision = performance_rollback(registry, validation, bad, metric, validation, policy)
            self.assertEqual(decision["to"], "baseline")
            self.assertEqual(registry.active_id(), "identical")  # 调用方先写审计，再提交版本切换。

    def test_absent_recall_target_is_not_fabricated(self):
        cfg = load_config(ROOT / "configs/a.live.json")
        _, validation = load_pair(cfg["train"], cfg["validation"])
        metric = evaluate(load_model(cfg["baseline"]), validation)
        metric["per_class"]["Acetaldehyde"]["support"] = 0
        decision = window_gate(metric, metric, metric, {**cfg["gate"], "min_macro_f1_gain": 0})
        self.assertIn("Acetaldehyde", decision["unobserved_recall_targets"])
        self.assertNotIn("class_recall_below_minimum:Acetaldehyde", decision["reasons"])


def metric(f1, recall=.8):
    return {"macro_f1": f1, "accuracy": .9, "per_class": {
        name: {"support": 100, "recall": recall, "precision": recall, "f1": recall} for name in CLASSES}}


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "configs/a.adaptive.json")

    def test_fixed_validation_is_a_floor_and_targets_are_explicit(self):
        baseline, current, candidate = metric(.714), metric(.827), metric(.788, .4)
        policy = adaptive.policy(self.cfg)
        result = adaptive.fixed_gate(candidate, current, baseline, policy)
        self.assertTrue(result["accepted"])
        self.assertIn("class_target_unmet:Acetaldehyde", result["warnings"])
        hard = {**policy, "hard_minimum_class_recall": {"Acetaldehyde": .5}}
        self.assertFalse(adaptive.fixed_gate(candidate, current, baseline, hard)["accepted"])
        self.assertFalse(adaptive.fixed_gate(metric(.70), current, baseline, policy)["accepted"])

    def test_history_searches_all_published_but_never_unpublished_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = Registry(root / "registry")
            registry.initialize(self.cfg["baseline"], metric(.714))
            for version in ("old_good", "middle", "active", "unpublished"):
                registry.register(version, self.cfg["baseline"], "baseline", metric(.8), {})
                if version != "unpublished":
                    registry.activate(version, "fixture_promotion")
            current, fixed = {"key": "current"}, {"key": "fixed"}

            def score(path, dataset):
                name = Path(path).parent.name
                if dataset["key"] == "fixed":
                    return metric(.714 if name == "baseline" else .8)
                return metric({"baseline": .4, "old_good": .9, "middle": .7, "active": .6, "unpublished": .99}[name])

            with patch("gas_a.adaptive.load_model", side_effect=lambda path: str(path)), patch("gas_a.adaptive.evaluate", side_effect=score):
                event, review = adaptive.select_history(registry, current, [], fixed, self.cfg)
            self.assertEqual(event["to"], "old_good")
            self.assertNotIn("unpublished", review["assessed"])
            self.assertEqual(registry.active_id(), "active")  # The caller audits then commits.

    def test_forward_feedback_counts_windows_not_duplicate_probes(self):
        entries = [{"family": "head_adapt", "version": "one", "batch": 4, "macro_f1_gain": .1}] * 10
        entries += [{"family": "head_adapt", "version": "one", "batch": 5, "macro_f1_gain": -.3}]
        stats = adaptive.feedback_stats([{"forward_feedback": entries}])["head_adapt"]
        self.assertEqual(stats["windows"], 2)
        self.assertAlmostEqual(stats["recent_mean_gain"], -.1)
        self.assertEqual(stats["observations"], 11)

    def test_program_binding_is_distinct_from_raw_model_reply(self):
        raw = {"action": "experiment", "experiment_names": ["head_adapt"], "hypothesis": "可能改善迁移，需检验。",
               "reason": "进行受控实验。", "claims": [{"kind": "untried", "family": "head_adapt", "scope": "all_parents"}]}
        normalized = adaptive.normalize_plan(raw)
        self.assertNotIn("evidence_ids", raw)
        self.assertIn("H_head_adapt", normalized["evidence_ids"])
        evidence = {"facts": {"C": {}, "F": {}, "H_head_adapt": {"all_parents": {"attempts": 0}}}}
        validate_plan(normalized, [{"name": "head_adapt"}], 1, evidence)

    def test_parameter_proposals_are_bounded_and_cannot_reset_state(self):
        base = next(x for x in self.cfg["experiments"] if x["name"] == "head_adapt")
        plan = {"experiment_names": ["head_adapt"], "parameter_changes": {"head_adapt": {"learning_rate": .0005, "feature_lr_scale": .1}}}
        updated = adaptive.proposed_specs([base], plan, self.cfg["optimization"], [], "parent")[0]
        self.assertEqual(updated["learning_rate"], .0005)
        self.assertEqual(base["learning_rate"], .001)
        for change in ({"seed": 1}, {"epochs": 100}, {"input_transform": "signed_log"}, {"learning_rate": .003}):
            plan["parameter_changes"] = {"head_adapt": change}
            with self.assertRaises(ValueError):
                adaptive.proposed_specs([base], plan, self.cfg["optimization"], [], "parent")
        plan["parameter_changes"] = {"head_adapt": {"learning_rate": .0005, "feature_lr_scale": .1}}
        past = [{"parent_checkpoint_sha256": "parent", 'train_sha256':'train','validation_sha256':'validation',
                 "experiments": [{"configuration": updated}]}]
        with self.assertRaisesRegex(ValueError, "already recorded"):
            adaptive.proposed_specs([base], plan, self.cfg["optimization"], past, "parent",train_sha='train',validation_sha='validation')

    def test_provider_format_variants_preserve_raw_and_enforce_bounds(self):
        base = next(x for x in self.cfg["experiments"] if x["name"] == "balanced")
        raw = {"action": "experiment", "experiment_names": [base["name"]],
               "reason": "根据当前窗口进行受控实验。" * 9, "hypothesis": "待检验后续窗口收益。",
               "parameter_changes": {"learning_rate": .0015, "balance": .5}}
        normalized = adaptive.normalize_plan(raw)
        self.assertEqual(normalized["parameter_changes"], {base["name"]: raw["parameter_changes"]})
        self.assertEqual(normalized["reason"], raw["reason"])
        evidence = {"facts": {"C": {}, "F": {}}}
        validate_plan(normalized, [base], 1, evidence, text_limits={"hypothesis": 256, "reason": 512})
        self.assertEqual(adaptive.proposed_specs([base], normalized, self.cfg["optimization"], [], "p")[0]["balance"], .5)
        normalized["reason"] += "确认传感器故障。"
        with self.assertRaisesRegex(ValueError, "causal diagnosis"):
            validate_plan(normalized, [base], 1, evidence, text_limits={"hypothesis": 256, "reason": 512})
        forbidden = adaptive.normalize_plan({**raw, "parameter_changes": {"seed": 1}})
        with self.assertRaisesRegex(ValueError, "outside permitted bounds"):
            adaptive.proposed_specs([base], forbidden, self.cfg["optimization"], [], "p")

    def test_history_current_hard_targets_precede_ranking_and_report_degradation(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry(Path(tmp) / "registry")
            registry.initialize(self.cfg["baseline"], metric(.714))
            registry.register("active", self.cfg["baseline"], "baseline", metric(.8), {})
            registry.activate("active", "fixture_promotion")
            cfg = copy.deepcopy(self.cfg)
            cfg["adaptive"]["hard_minimum_class_recall"] = {"Acetaldehyde": .5}
            def score(path, dataset):
                if dataset["key"] == "fixed":
                    return metric(.8)
                return metric(.9, .2) if Path(path).parent.name == "active" else metric(.6, .8)
            with patch("gas_a.adaptive.load_model", side_effect=lambda p: str(p)), patch("gas_a.adaptive.evaluate", side_effect=score):
                event, review = adaptive.select_history(registry, {"key": "current"}, [], {"key": "fixed"}, cfg)
                self.assertEqual(event["to"], "baseline")
                self.assertFalse(review["degraded"])
            with patch("gas_a.adaptive.load_model", side_effect=lambda p: str(p)), patch("gas_a.adaptive.evaluate", return_value=metric(.8, .2)):
                _, review = adaptive.select_history(registry, {}, [], {}, cfg)
                self.assertTrue(review["degraded"])
                self.assertFalse(review["hard_targets_satisfied"])

    def test_retry_costs_include_prior_probe_and_history_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, previous = Path(tmp) / "retry", Path(tmp) / "previous"
            for directory in (root, previous):
                (directory / "session/model_calls").mkdir(parents=True)
            write_json(root / "configuration.json", {"batches": [], "prior_attempts": [str(previous)], "gate": {}})
            write_json(previous / "summary.json", {"controller_failed_windows": 1, "costs": {
                "candidate_cpu_seconds_known": 2, "candidate_wall_seconds_known": 3, "candidates_with_unknown_resources": 0,
                "probe_cpu_seconds_known": .25, "history_selection_cpu_seconds_known": .5}})
            result = export_temporal(root, [], {}, {"session_succeeded": True})
            self.assertEqual(result["costs"]["probe_cpu_seconds_known"], .25)
            self.assertEqual(result["costs"]["history_selection_cpu_seconds_known"], .5)

    def test_next_window_probes_are_committed_before_labels_and_never_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = TemporalTests()
            train, val = helper.make_windows(root)
            cfg = copy.deepcopy(self.cfg)
            cfg["runtime"] = str(root / "adaptive")
            cfg["experiments"] = [{**cfg["experiments"][0], "epochs": 2}]
            cfg["optimization"]["max_selected_experiments"] = 1
            seen = []

            def checked_reveal(path, inputs):
                number = int(Path(path).stem[5:])
                directory = Path(cfg["runtime"]) / "windows" / f"batch{number:02d}"
                commit = read_json(directory / "prediction_commit.json")
                self.assertEqual(digest(directory / "predictions.npz"), commit["predictions_sha256"])
                if number == 4:
                    self.assertTrue(commit["probe_bindings"])
                    with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                        self.assertEqual(saved["probe_values"].shape[1:], (40, 6))
                seen.append(number)
                return reveal_labels(path, inputs)

            with patch("gas_a.temporal.reveal_labels", side_effect=checked_reveal):
                summary = run_temporal(cfg, root, (3, 4), cloud=False, client=adaptive.RulePlanner())
            self.assertEqual(seen, [3, 4])
            self.assertTrue(summary["success"])
            self.assertEqual(summary["mode"], "historical_cpu_rule_screen")
            self.assertEqual(summary["costs"]["calls"], 0)
            rows = history(Path(cfg["runtime"]) / "history.jsonl")
            self.assertTrue(rows[-1]["forward_feedback"])
            for trial in rows[0]["experiments"]:
                self.assertEqual(trial["report"]["train_sha256"], train["sha256"])
                self.assertEqual(trial["report"]["validation_sha256"], val["sha256"])
                model = load_model(Path(trial["evidence_directory"]) / "checkpoint.npz")
                parent = load_model(cfg["baseline"])
                np.testing.assert_array_equal(model.mean, parent.mean)
                np.testing.assert_array_equal(model.scale, parent.scale)
                self.assertEqual(model.meta["seed"], parent.meta["seed"])


class ControllerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/a.live.json")
        cls.train, cls.val = load_pair(cls.config["train"], cls.config["validation"])
        cls.base = load_model(cls.config["baseline"])
        cls.metric = evaluate(cls.base, cls.val)
        cls.train_metric = evaluate(cls.base, cls.train)
        cls.drift = monitor(cls.train["x"], cls.val["x"], cls.config["monitor"])
        cls.diagnosis = diagnose(cls.train, cls.val, cls.metric, cls.drift, train_metrics=cls.train_metric)
        cls.parent = digest(cls.config["baseline"])

    def evidence(self, past=None):
        return measured_evidence(self.metric, self.metric, self.train_metric, self.diagnosis, self.drift,
            past or [], self.parent, [s["name"] for s in self.config["experiments"]], validation_sha=self.val["sha256"])

    def plan(self, evidence, names=None, action="experiment"):
        menu = self.config["experiments"]
        prompt = {"allowed_experiments": menu, "budget": {"max_experiments": 2}}
        return fixture_plan(prompt, names or [menu[0]["name"]], action=action)

    def test_planning_uses_actual_errors_and_keeps_contract_and_menu(self):
        evidence = self.evidence()
        menu = [s for s in self.config["experiments"] if s["name"] in ("gentle", "head_adapt")]
        prompt = build_planning_prompt(menu, evidence, self.config["gate"], 2, 30, mode="sdk", index=1)
        table = prompt["evidence"]
        self.assertEqual(table["overall"]["training"]["macro_f1"], evidence["facts"]["T"]["macro_f1"])
        self.assertNotIn("facts", table)
        self.assertEqual(prompt["allowed_experiments"], menu)
        self.assertEqual(set(prompt["experiment_guide"]), {"gentle", "head_adapt"})
        self.assertEqual(prompt["plan_contract"], planning_contract(2))
        self.assertEqual(evidence["facts"]["D"]["top_confusions"], self.diagnosis["top_confusions"][:3])
        expected = sorted((name for name in CLASSES if self.metric["per_class"][name]["support"]),
                          key=lambda name: self.metric["per_class"][name]["f1"])[:2]
        self.assertEqual([row["label"] for row in prompt["decision_context"]["low_f1_classes"]], expected)
        for row in prompt["decision_context"]["low_f1_classes"]:
            self.assertEqual(row["support"], self.metric["per_class"][row["label"]]["support"])
            self.assertAlmostEqual(row["f1"], self.metric["per_class"][row["label"]]["f1"], places=5)
        validate_plan(fixture_plan(prompt), menu, 2, evidence)

    def test_planning_respects_pause_and_effective_hard_targets(self):
        evidence = self.evidence()
        evidence["facts"]["S"] = {"pause": True}
        menu = self.config["experiments"][:2]
        prompt = build_planning_prompt(menu, evidence, self.config["gate"], 2, 30, mode="sdk", index=2)
        self.assertFalse(prompt["decision_context"]["training_allowed"])
        validate_plan(fixture_plan(prompt, action="observe"), menu, 2, evidence)
        evidence["facts"]["S"]["pause"] = False
        prompt = build_planning_prompt([], evidence, self.config["gate"], 0, 30, mode="sdk", index=3)
        self.assertFalse(prompt["decision_context"]["training_allowed"])
        validate_plan(fixture_plan(prompt, action="observe"), [], 0, evidence)
        adaptive_gate = {"hard_minimum_class_recall": {}}
        prompt = build_planning_prompt(menu, evidence, adaptive_gate, 2, 30, mode="temporal", index=4)
        self.assertTrue(prompt["decision_context"]["training_allowed"])
        self.assertEqual(prompt["decision_context"]["recall_below_configured_minimum"], [])

    def test_compact_planning_retains_training_and_confusion_evidence(self):
        from gas_a.adaptive import compact_evidence
        evidence = self.evidence()
        compact = compact_evidence(evidence, self.config["experiments"][:2], {})
        self.assertEqual(compact["overall"]["training"]["macro_f1"], evidence["facts"]["T"]["macro_f1"])
        self.assertEqual(compact["top_confusions"], self.diagnosis["top_confusions"][:3])
        self.assertEqual(len(compact["class_rows"]), 6)
        self.assertTrue(all(row[-2] is None for row in compact["history_rows"]))

    def test_selection_ranks_common_host_scores_and_preserves_local_class_regressions(self):
        parent, candidate = copy.deepcopy(self.metric), copy.deepcopy(self.metric)
        parent["macro_f1"], candidate["macro_f1"] = .7, .9
        parent["per_class"]["Ammonia"].update(support=100, recall=.8, f1=.8)
        candidate["per_class"]["Ammonia"].update(support=100, recall=.7, f1=.75)
        eligible = {"parent-complete-id": {"kind": "retain", "local": parent, "host": {"macro_f1": .82}},
                    "candidate-complete-id": {"kind": "promote", "local": candidate, "host": {"macro_f1": .81}}}
        original = copy.deepcopy(eligible)
        trial = {"name": "gentle", "host_receipt": {"artifact_id": "candidate-complete-id"},
                 "resources": {"cpu_seconds": 4.2}, "host_gate_passed": False}
        prompt = build_selection_prompt(eligible, [trial], "parent-complete-id", mode="sdk", index=2,
                                        history_ids=["historical-record"])
        comparison = prompt["comparison"]
        self.assertEqual(comparison["basis"], "host")
        self.assertEqual(comparison["measured_preference"], "parent-complete-id")
        row = next(row for row in comparison["rows"] if row["kind"] == "promote")
        self.assertAlmostEqual(row["delta_vs_current_same_scope"], -.01)
        self.assertEqual(row["class_change_scope"], "local")
        change = next(change for change in row["class_changes_vs_current"] if change["label"] == "Ammonia")
        self.assertEqual(change, {"label": "Ammonia", "support": 100, "recall_delta": -.1, "f1_delta": -.05})
        self.assertEqual(prompt["experiments"][0]["resources_already_spent"]["cpu_seconds"], 4.2)
        self.assertFalse(prompt["experiments"][0]["host_gate_passed"])
        self.assertEqual(prompt["experiments"][0]["new_candidate_artifact_id"], "candidate-complete-id")
        self.assertNotIn("history_ids", prompt)  # 宿主记录ID由程序提交，不参与模型业务决策。
        self.assertEqual(eligible, original)

    def test_selection_missing_host_uses_local_scores_for_every_artifact(self):
        parent, candidate = copy.deepcopy(self.metric), copy.deepcopy(self.metric)
        parent["macro_f1"], candidate["macro_f1"] = .8, .7
        eligible = {"current": {"kind": "retain", "local": parent},
                    "new": {"kind": "promote", "local": candidate, "host": {"macro_f1": .99}}}
        prompt = build_selection_prompt(eligible, [], "current", mode="sdk", index=1)
        self.assertEqual(prompt["comparison"]["basis"], "local")
        self.assertEqual(prompt["comparison"]["measured_preference"], "current")
        self.assertEqual([row["score"] for row in prompt["comparison"]["rows"]], [.8, .7])
        self.assertEqual(prompt["eligible_artifacts"]["new"]["host"]["macro_f1"], .99)

    def test_selection_tie_retains_current_and_single_restore_does_not_invent_parent_score(self):
        eligible = {"new": {"kind": "promote", "local": self.metric},
                    "current": {"kind": "retain", "local": self.metric}}
        tied = build_selection_prompt(eligible, [], "current", mode="sdk", index=1)
        self.assertEqual(tied["comparison"]["measured_preference"], "current")
        restore = build_selection_prompt({"baseline": {"kind": "rollback", "local": self.metric}},
                                         [{"name": "gentle", "error": "timeout"}], "unsafe-current", mode="sdk", index=3)
        self.assertEqual(restore["comparison"]["measured_preference"], "baseline")
        row = restore["comparison"]["rows"][0]
        self.assertIsNone(row["delta_vs_current_same_scope"])
        self.assertEqual(row["class_changes_vs_current"], [])
        self.assertEqual(restore["experiments"][0]["resources_already_spent"], {})

    def test_temporal_selection_uses_configured_ranking_before_current_shadow_score(self):
        current, candidate = class_metrics(self.metric), class_metrics(self.metric)
        current["macro_f1"], candidate["macro_f1"] = .8, .9
        eligible = {"current": {"kind": "retain", "current_window_shadow": current, "selection_score": .85},
                    "new": {"kind": "promote", "current_window_shadow": candidate, "selection_score": .82}}
        prompt = build_selection_prompt(eligible, [], "current", mode="temporal", index=4)
        self.assertEqual(prompt["comparison"]["basis"], "selection_score")
        self.assertEqual(prompt["comparison"]["measured_preference"], "current")
        self.assertAlmostEqual(prompt["comparison"]["rows"][1]["delta_vs_current_same_scope"], -.03)
        for value in eligible.values():
            value.pop("selection_score")
        prompt = build_selection_prompt(eligible, [], "current", mode="temporal", index=4)
        self.assertEqual(prompt["comparison"]["basis"], "current_window_shadow")
        self.assertEqual(prompt["comparison"]["measured_preference"], "new")

    def test_no_history_cannot_claim_highest_gain(self):
        evidence = self.evidence()
        plan = self.plan(evidence, ["weak_class"])
        plan.update(reason="该方向历史增益最高。", evidence_ids=["C", "H_weak_class"],
                    claims=[{"kind": "best_fixed_gain", "family": "weak_class", "scope": "all_parents"}])
        with self.assertRaisesRegex(ValueError, "best-gain claim is unsupported"):
            validate_plan(plan, self.config["experiments"], 2, evidence)
        plan["claims"] = []
        with self.assertRaisesRegex(ValueError, "ranking assertion"):
            validate_plan(plan, self.config["experiments"], 2, evidence)

    def test_histories_are_scoped_by_parent_dataset_and_shadow_delta(self):
        report = {"parent_checkpoint_sha256": "different-parent", "validation_sha256": self.val["sha256"],
                  "validation_before": {"macro_f1": .5}, "validation_after": {"macro_f1": .6}}
        past = [{"round": 1, "after": self.metric, "shadow_before": {"macro_f1": .8},
                 "experiments": [{"name": "weak_class", "report": report,
                                  "shadow_metrics": {"macro_f1": .7}, "eligible": False}]}]
        evidence = self.evidence(past)
        stat = evidence["facts"]["H_weak_class"]
        self.assertEqual(stat["all_parents"]["fixed_best_gain"], .1)
        self.assertEqual(stat["all_parents"]["window_best_paired_gain"], -.1)
        self.assertEqual(stat["same_parent"]["attempts"], 0)
        plan = self.plan(evidence, ["weak_class"])
        plan.update(evidence_ids=["C", "H_weak_class"],
                    claims=[{"kind": "untried", "family": "weak_class", "scope": "all_parents"}])
        with self.assertRaisesRegex(ValueError, "already has recorded"):
            validate_plan(plan, self.config["experiments"], 2, evidence)
        plan["claims"][0]["scope"] = "same_parent"
        validate_plan(plan, self.config["experiments"], 2, evidence)
        plan["claims"] = [{"kind": "best_fixed_gain", "family": "weak_class", "scope": "all_parents"}]
        validate_plan(plan, self.config["experiments"], 2, evidence)
        report["validation_sha256"] = "changed-validation"
        self.assertIsNone(self.evidence(past)["facts"]["H_weak_class"]["all_parents"]["fixed_best_gain"])

    def test_unknown_ids_weakened_gates_and_unproved_causality_are_rejected(self):
        evidence = self.evidence()
        self.assertEqual(len(evidence["facts"]["C"]["classes"]), 6)
        for changes in ({"evidence_ids": ["fabricated"]}, {"success_condition": "ignore_class_recall"},
                        {"reason": "已确认严重过拟合，必须重新标准化。"}):
            plan = self.plan(evidence)
            plan.update(changes)
            with self.assertRaises(ValueError):
                validate_plan(plan, self.config["experiments"], 2, evidence)

    def test_two_new_family_claims_have_room_for_current_fixed_and_drift_citations(self):
        evidence = self.evidence()
        plan = self.plan(evidence, ["anchored", "sensor_robust"])
        plan.update(evidence_ids=["C", "F", "D", "H_anchored", "H_sensor_robust"],
                    claims=[{"kind": "untried", "family": name, "scope": "all_parents"}
                            for name in ("anchored", "sensor_robust")])
        validate_plan(plan, self.config["experiments"], 2, evidence)
        plan["evidence_ids"].remove("H_sensor_robust")
        with self.assertRaisesRegex(ValueError, "H_sensor_robust"):
            validate_plan(plan, self.config["experiments"], 2, evidence)

    def test_observe_has_no_training_or_second_selection_call(self):
        class ObserveFixture(ReplayModelFixture):
            def complete(self, messages, **kwargs):
                import json
                result = super().complete(messages, **kwargs)
                prompt = json.loads(messages[-1]["content"])
                answer = fixture_plan(prompt, action="observe")
                result["text"] = json.dumps(answer)
                result["response"]["choices"][0]["message"]["content"] = result["text"]
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = TemporalTests()
            helper.make_windows(root)
            cfg = helper.config(root)
            client = ObserveFixture()
            with patch("gas_a.candidates.train_candidate", side_effect=AssertionError("Observe must not train")):
                summary = run_temporal(cfg, root, (3, 4), cloud=False, client=client)
            self.assertTrue(summary["success"])
            self.assertEqual(summary["costs"]["calls"], 1)
            self.assertEqual(summary["costs"]["candidate_cpu_seconds_known"], 0)
            self.assertEqual(history(Path(cfg["runtime"]) / "history.jsonl")[0]["decision_source"], "model_observe")

    def test_no_gain_pause_resets_after_observation_and_data_change(self):
        past = [{"parent_checkpoint_sha256": self.parent, "train_sha256": self.train["sha256"],
                 "validation_sha256": self.val["sha256"], "action": "retain",
                 "experiments": [{"name": "gentle", "eligible": False}]} for _ in range(3)]
        args = (self.parent, self.train["sha256"], self.val["sha256"], self.config)
        self.assertTrue(stopping_evidence(past, *args)["pause"])
        self.assertFalse(stopping_evidence(past, self.parent, "new-authorized-train", self.val["sha256"], self.config)["pause"])
        self.assertFalse(stopping_evidence([*past, {"decision_source": "budget_pause"}], *args)["pause"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = Registry(root / "deployment")
            registry.initialize(self.config["baseline"], self.metric)
            audit = Audit(root / "audit", cloud=False)
            audit.start({})
            row = {"batch": 7, "drift": self.drift, "static_baseline": self.metric, "fixed_static_baseline": self.metric}
            with patch("gas_a.candidates.train_candidate", side_effect=AssertionError("Paused must not train")):
                optimize_window(self.config, registry, self.train, self.val, self.val, row, past, audit, None, root)
            self.assertEqual(row["decision_source"], "budget_pause")
            self.assertFalse(row["experiments"])
            self.assertFalse(list((audit.root / "model_calls").glob("*.json")))

    def test_all_supported_classes_protected_against_static_baseline(self):
        baseline = copy.deepcopy(self.metric)
        active = copy.deepcopy(baseline)
        candidate = copy.deepcopy(baseline)
        for metric in (baseline, active, candidate):
            for name in CLASSES:
                metric["per_class"][name].update(support=100, recall=.8)
        active["per_class"]["Ammonia"]["recall"] = .1
        candidate["per_class"]["Ammonia"]["recall"] = .1
        decision = protect_decision({"accepted": True, "reasons": ["quality_gates_passed"]},
                                    candidate, active, baseline, self.config["gate"])
        self.assertFalse(decision["accepted"])
        self.assertIn("class_recall_regression:Ammonia", decision["reasons"])

    def test_historical_high_f1_cannot_restore_collapsed_class(self):
        def metric(f1, ammonia):
            value = copy.deepcopy(self.metric)
            value["macro_f1"] = f1
            for name in CLASSES:
                value["per_class"][name].update(support=100, recall=.8)
            value["per_class"]["Ammonia"]["recall"] = ammonia
            return value
        base_metric, active_metric, historic_metric = metric(.7, .8), metric(.65, .8), metric(.9, .1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = Registry(root / "deploy")
            registry.initialize(self.config["baseline"], base_metric)
            cp = root / "model.npz"
            self.base.save(cp)
            registry.register("historic", cp, "baseline", historic_metric, {})
            registry.activate("historic", "fixture")
            registry.register("current", cp, "historic", active_metric, {})
            registry.activate("current", "fixture")

            def measured(path, dataset):
                return {"baseline": base_metric, "current": active_metric, "historic": historic_metric}[Path(path).parent.name]

            with patch("gas_a.temporal.load_model", side_effect=lambda path: str(path)), patch("gas_a.temporal.evaluate", side_effect=measured):
                event = performance_rollback(registry, self.val, active_metric, base_metric, self.val, self.config["gate"])
            self.assertEqual(event["to"], "baseline")
            self.assertIn("historic", event["rejected_historical_restores"])

    def test_robustness_menu_is_training_only_and_preserves_full_adam(self):
        menu = ordered_specs(self.config, [], self.parent,
            {"drift_detected": True, "fit_comparison": {"macro_f1_gap": .3, "acetaldehyde_validation_recall": 0}})
        self.assertTrue(any(s["name"] in ("sensor_robust", "noise_robust") for s in menu))
        with tempfile.TemporaryDirectory() as tmp:
            for spec in self.config["experiments"]:
                if spec["name"] not in ("sensor_robust", "noise_robust"):
                    continue
                spec = {**spec, "epochs": 2}
                limits = {**self.config["optimization"], "select_best_epoch": False}
                parent = load_model(self.config["baseline"])
                output = Path(tmp) / spec["name"]
                report = train_candidate(self.config["baseline"], self.train, self.val, output, spec, self.diagnosis, limits)
                candidate = load_model(output / "checkpoint.npz")
                fit_weighted(parent, self.train, spec, self.diagnosis["weakest_class"], float("inf"))
                for group in ("params", "first", "second"):
                    for name in getattr(parent, group):
                        np.testing.assert_array_equal(getattr(candidate, group)[name], getattr(parent, group)[name])
                np.testing.assert_array_equal(candidate.mean, self.base.mean)
                np.testing.assert_array_equal(candidate.scale, self.base.scale)
                self.assertEqual(report["train_sha256"], self.train["sha256"])
                self.assertEqual(report["validation_sha256"], self.val["sha256"])
                self.assertNotIn("active_features", candidate.meta)


if __name__ == "__main__":
    unittest.main()
