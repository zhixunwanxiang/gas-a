"""冻结必须保留非零父权重及Adam状态，解冻、架构迁移和实际规划路径均可续训。"""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from gas_a import adaptive
from gas_a.architecture import migrate_parent, available_choices
from gas_a.data import load_model, load_pair
from gas_a.io import digest, read_json
from gas_a.optimize import bounded_experiment, fit_weighted, frozen_parameters, frozen_state_hashes, train_candidate
from gas_a.pipeline import ordered_specs
from fixtures import legacy_config as load_config
from gas_a.temporal import run_temporal
from fixtures import ReplayModelFixture

ROOT = Path(__file__).resolve().parents[1]


class FreezingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/a.adaptive.json")
        cls.train, cls.val = load_pair(cls.cfg["train"], cls.cfg["validation"])
        cls.baseline = Path(cls.cfg["baseline"])
        cls.parent_sha = digest(cls.baseline)

    def parent(self, migrated=False):
        model = load_model(self.baseline)
        return migrate_parent(model, {"width_multiplier": 2, "add_layers": 2}, self.parent_sha)[0] if migrated else model

    def spec(self, **updates):
        return {**self.cfg["experiments"][0], "epochs": 2, **updates}

    def fit(self, model, spec, **options):
        return fit_weighted(model, self.train, spec, "Acetaldehyde", time.monotonic() + 30, **options)

    def assert_state_equal(self, one, two, names=None):
        for group in ("params", "first", "second"):
            for name in one.params if names is None else names:
                self.assertEqual(getattr(one, group)[name].tobytes(), getattr(two, group)[name].tobytes(), (group, name))

    def test_layer_modes_preserve_weights_and_nonzero_moments_on_original_and_migrated_parent(self):
        for migrated in (False, True):
            for mode in ("features", "head", "first_hidden"):
                with self.subTest(migrated=migrated, mode=mode):
                    model = self.parent(migrated)
                    parent = copy.deepcopy(model)
                    spec = self.spec(freeze=mode, l2=.01, anchor_l2=.1, feature_lr_scale=.25)
                    frozen = frozen_parameters(model, spec)
                    self.assertTrue(any(np.any(model.first[n]) for n in frozen))
                    records, _ = self.fit(model, spec)
                    self.assert_state_equal(parent, model, frozen)
                    trainable = model.params.keys() - frozen
                    self.assertTrue(any(not np.array_equal(parent.params[n], model.params[n]) for n in trainable))
                    self.assertTrue(any(not np.array_equal(parent.first[n], model.first[n]) for n in trainable))
                    self.assertEqual(model.meta["steps"], parent.meta["steps"] + 8)
                    self.assertEqual(model.meta["epochs"], parent.meta["epochs"] + 2)
                    self.assertEqual(model.meta["seed"], parent.meta["seed"])
                    np.testing.assert_array_equal(model.mean, parent.mean)
                    np.testing.assert_array_equal(model.scale, parent.scale)
                    self.assertTrue(all(r["frozen_parameters"] == sorted(frozen) for r in records))
                    if migrated and mode == "features":
                        self.assertTrue({"w3", "b3", "w4", "b4"} <= frozen)
        self.assertEqual(digest(self.baseline), self.parent_sha)

    def test_explicit_tensor_freezing_and_layer_mode_can_be_combined(self):
        model = self.parent(True)
        parent = copy.deepcopy(model)
        spec = self.spec(freeze="head", frozen_parameters=["b1", "w3"])
        self.assertEqual(frozen_parameters(model, spec), {"w2", "b2", "b1", "w3"})
        self.fit(model, spec)
        self.assert_state_equal(parent, model, {"w2", "b2", "b1", "w3"})
        self.assertFalse(np.array_equal(model.params["w1"], parent.params["w1"]))

    def test_invalid_freezing_fails_before_any_model_mutation(self):
        invalid = [{"freeze": "all"}, {"freeze": []}, {"frozen_parameters": "w1"},
                   {"frozen_parameters": ["w1", "w1"]}, {"frozen_parameters": [{}]},
                   {"frozen_parameters": ["w6"]}, {"frozen_parameters": ["w3"]},
                   {"frozen_parameters": ["w1", "b1", "w2", "b2"]},
                   {"freeze": "features", "frozen_parameters": ["w2", "b2"]},
                   {"freeze": "features", "kept_sensors": [1, 2]},
                   {"frozen_parameters": ["w1"], "kept_sensors": [1, 2]}]
        for update in invalid:
            with self.subTest(update=update):
                model = self.parent()
                parent = copy.deepcopy(model)
                with self.assertRaises(ValueError):
                    self.fit(model, self.spec(**update))
                self.assert_state_equal(model, parent)
                self.assertEqual(model.meta, parent.meta)

    def test_head_freezing_is_compatible_with_sensor_weight_projection(self):
        model = self.parent()
        parent = copy.deepcopy(model)
        spec = self.spec(freeze="head", kept_sensors=[1, 2])
        bounded_experiment(spec, self.cfg["optimization"])
        self.fit(model, spec)
        self.assert_state_equal(model, parent, {"w2", "b2"})
        excluded = [i for i in range(128) if i // 8 + 1 not in spec["kept_sensors"]]
        np.testing.assert_array_equal(model.params["w1"][excluded], 0)
        with tempfile.TemporaryDirectory() as tmp:
            limits = {**self.cfg["optimization"], "select_best_epoch": False}
            report = train_candidate(self.baseline, self.train, self.val, Path(tmp) / "sparse", spec,
                                     {"weakest_class": "Acetaldehyde"}, limits)
            evidence = report["freezing"]
            self.assertEqual(evidence["projected_parameter_count"], len(excluded) * 32)
            self.assertEqual(evidence["partly_projected_parameters"], ["w1"])
            self.assertEqual(sum(evidence[key] for key in ("frozen_parameter_count", "trainable_parameter_count",
                                                         "projected_parameter_count")), model.meta["parameter_count"])

    def test_no_freeze_keeps_existing_continuation_bit_identical(self):
        implicit, explicit = self.parent(), self.parent()
        spec = self.spec()
        self.fit(implicit, spec)
        self.fit(explicit, {**spec, "freeze": "none", "frozen_parameters": []})
        self.assert_state_equal(implicit, explicit)
        self.assertEqual(implicit.meta, explicit.meta)

    def test_freeze_is_per_stage_and_saved_state_can_be_unfrozen_without_reset(self):
        model, parent = self.parent(), self.parent()
        self.fit(model, self.spec(freeze="features"))
        self.assert_state_equal(model, parent, {"w1", "b1"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen-parent.npz"
            model.save(path)
            continued, expected = load_model(path), copy.deepcopy(model)
            self.fit(continued, self.spec())
            self.fit(expected, self.spec(freeze="none"))
            self.assert_state_equal(continued, expected)
            self.assertFalse(np.array_equal(continued.params["w1"], parent.params["w1"]))
            self.assertFalse(np.array_equal(continued.first["w1"], parent.first["w1"]))
            self.assertEqual(continued.meta["steps"], parent.meta["steps"] + 16)
            self.assertNotIn("freeze", continued.meta)

    def test_segmented_frozen_training_replays_cosine_anchor_and_class_focus(self):
        for migrated in (False, True):
            continuous, segmented = self.parent(migrated), self.parent(migrated)
            anchor = {k: v.copy() for k, v in segmented.params.items()}
            spec = self.spec(freeze="first_hidden", epochs=3, schedule="cosine", final_lr_ratio=.2,
                             anchor_l2=.1, class_boosts={"Acetaldehyde": 3}, balance=.5)
            self.fit(continuous, spec)
            for offset in range(3):
                self.fit(segmented, spec, epoch_offset=offset, epochs_to_run=1, anchor_params=anchor)
            self.assert_state_equal(continuous, segmented)

    def test_named_class_focus_changes_only_trainable_gradients(self):
        plain, focused = self.parent(), self.parent()
        spec = self.spec(freeze="features")
        _, plain_weights = self.fit(plain, spec)
        _, focus_weights = self.fit(focused, {**spec, "class_boosts": {"Ammonia": 3}})
        self.assert_state_equal(plain, focused, {"w1", "b1"})
        self.assertNotEqual(plain_weights, focus_weights)
        self.assertFalse(np.array_equal(plain.params["w2"], focused.params["w2"]))
        np.testing.assert_allclose(np.mean(np.array(focus_weights)[self.train["y"]]), 1., rtol=0, atol=1e-15)

    def test_export_proves_frozen_state_with_migration_and_validation_isolation(self):
        for migrated in (False, True):
            spec = self.spec(freeze="features", **({"architecture": {"add_layers": 1}} if migrated else {}))
            limits = {**self.cfg["optimization"], "select_best_epoch": False}
            altered = {**self.val, "x": -3 * self.val["x"], "y": (self.val["y"] + 1) % 6}
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name, validation in (("one", self.val), ("two", altered)):
                    train_candidate(self.baseline, self.train, validation, root / name, spec,
                                    {"weakest_class": "Acetaldehyde"}, limits)
                child = load_model(root / "one/checkpoint.npz")
                self.assert_state_equal(child, load_model(root / "two/checkpoint.npz"))
                training = read_json(root / "one/training.json")
                evidence = training["freezing"]
                self.assertTrue(evidence["verified_unchanged"])
                self.assertEqual(evidence["before_sha256"], evidence["exported_sha256"])
                self.assertEqual(evidence["exported_sha256"], frozen_state_hashes(child, evidence["frozen_parameters"]))
                self.assertEqual(evidence["frozen_parameter_count"] + evidence["trainable_parameter_count"], child.meta["parameter_count"])
                self.assertEqual(evidence["trainable_parameters"], ["b2", "w2"])
                self.assertEqual(training["parent_checkpoint_sha256"], self.parent_sha)
                parent = load_model(root / "one/migrated_parent.npz") if migrated else self.parent()
                self.assert_state_equal(child, parent, evidence["frozen_parameters"])

    def test_27b_freezing_and_named_class_proposals_are_bounded(self):
        base = self.cfg["experiments"][0]
        raw = {"experiment_names": [base["name"]], "parameter_changes": {base["name"]:
               {"freeze": "features", "class_boosts": {"Acetaldehyde": 2}}}}
        original = copy.deepcopy(raw)
        result = adaptive.proposed_specs([base], raw, self.cfg["optimization"], [], self.parent_sha)[0]
        self.assertEqual(result["freeze"], "features")
        self.assertEqual(raw, original)
        adaptive.proposed_specs([base], raw, self.cfg["optimization"], [], self.parent_sha, focus_only=True)
        with self.assertRaisesRegex(ValueError, "Only freezing"):
            adaptive.proposed_specs([base], {**raw, "parameter_changes": {base["name"]: {"learning_rate": .0005}}},
                                   self.cfg["optimization"], [], self.parent_sha, focus_only=True)
        for update in ({"freeze": "all"}, {"freeze": False}, {"frozen_parameters": ["w1"]},
                       {"class_boosts": {"unknown": 2}}, {"class_boosts": {"Ammonia": True}},
                       {"class_boosts": {"Ammonia": 3.1}}, {"class_boosts": {"Ammonia": float("nan")}},
                       {"class_boosts": {"Ammonia": 2, "Acetone": 2, "Ethanol": 2}}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                adaptive.proposed_specs([base], {**raw, "parameter_changes": {base["name"]: update}},
                                       self.cfg["optimization"], [], self.parent_sha)

    def test_temporal_27b_plan_executes_freezing_and_focus_without_extra_calls(self):
        class Controller(ReplayModelFixture):
            def complete(self, messages, **options):
                result = super().complete(messages, **options)
                prompt = self.requests[-1]
                if "allowed_experiments" in prompt:
                    name = prompt["allowed_experiments"][0]["name"]
                    reply = {"action": "experiment", "experiment_names": [name],
                             "hypothesis": "冻结已有特征并调整弱类训练贡献，验证分类头能否改善。", "reason": "受控比较，效果待验证。",
                             "parameter_changes": {name: {"freeze": "features", "class_boosts": {"Acetaldehyde": 2}}}}
                    result["text"] = json.dumps(reply)
                    result["response"]["choices"][0]["message"]["content"] = result["text"]
                return result

        for use_adaptive in (False, True):
            with self.subTest(adaptive=use_adaptive):
                self.exercise_temporal_focus(Controller, use_adaptive)

    def test_temporal_default_architecture_choice_executes_migration_and_freezing(self):
        class Controller(ReplayModelFixture):
            def complete(self, messages, **options):
                result = super().complete(messages, **options)
                prompt = self.requests[-1]
                if "allowed_experiments" in prompt:
                    name = prompt["allowed_experiments"][0]["name"]
                    reply = {"action": "experiment", "experiment_names": [name],
                             "hypothesis": "保留旧层并增加容量，验证分类边界能否改善。", "reason": "迁移后实测，效果待验证。",
                             "parameter_changes": {name: {"architecture_choice": "widen_deepen", "freeze": "first_hidden"}}}
                    result["text"] = json.dumps(reply)
                    result["response"]["choices"][0]["message"]["content"] = result["text"]
                return result

        for use_adaptive in (False, True):
            with self.subTest(adaptive=use_adaptive):
                self.exercise_temporal_focus(Controller, use_adaptive, architecture_choice="widen_deepen")

    def exercise_temporal_focus(self, controller_type, use_adaptive, architecture_choice=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = copy.deepcopy(self.cfg)
            cfg["adaptive"]["enabled"] = use_adaptive
            cfg["runtime"] = str(root / "runtime")
            cfg["experiments"] = [self.spec(epochs=1)]
            for batch, offset in ((3, 0), (4, 40)):
                rows = [str(int(y) + 1) + " " + " ".join(f"{i+1}:{v:.17g}" for i, v in enumerate(x))
                        for x, y in zip(self.val["x"][offset:offset+40], self.val["y"][offset:offset+40])]
                (root / f"batch{batch}.dat").write_text("\n".join(rows) + "\n", encoding="utf-8")
            controller = controller_type()
            summary = run_temporal(cfg, root, (3, 4), cloud=False, client=controller)
            self.assertTrue(summary["success"])
            self.assertEqual(summary["controller_failed_windows"], 0)
            self.assertEqual(summary["controller_recovered_windows"], 0)
            self.assertEqual(len(controller.requests), 2)
            prompt = controller.requests[0]
            self.assertIn("features", prompt["permitted_parameter_changes"]["freeze_choices_by_experiment"]["gentle"])
            row = read_json(Path(cfg["runtime"]) / "windows/batch03/result.json")
            trial = row["experiments"][0]
            self.assertNotIn("error", trial)
            self.assertTrue(trial["report"]["freezing"]["verified_unchanged"])
            if architecture_choice:
                self.assertEqual(prompt["permitted_parameter_changes"]["architecture_choice"], cfg["architecture_choices"])
                self.assertTrue(trial["report"]["architecture_migration"]["performed"])
                self.assertEqual(trial["report"]["architecture"], {"hidden": 64, "hidden_layers": 2})
                self.assertEqual(trial["report"]["parent_checkpoint_sha256"], self.parent_sha)
            else:
                self.assertEqual(trial["configuration"]["class_boosts"], {"Acetaldehyde": 2})

    def test_architecture_choices_are_filtered_on_actual_parent_and_unlisted_requests_are_rejected(self):
        choices = self.cfg["architecture_choices"]
        self.assertEqual(set(available_choices(self.parent(), choices)), {"widen_hidden", "deepen_hidden", "widen_deepen"})
        cap = migrate_parent(self.parent(), {"hidden": 128}, self.parent_sha)[0]
        self.assertEqual(set(available_choices(cap, choices)), {"deepen_hidden"})
        deep = migrate_parent(self.parent(), {"hidden_layers": 4}, self.parent_sha)[0]
        self.assertEqual(set(available_choices(deep, choices)), {"widen_hidden"})
        cap = migrate_parent(self.parent(), {"hidden": 128, "hidden_layers": 4}, self.parent_sha)[0]
        self.assertEqual(available_choices(cap, choices), {})
        self.assertNotIn("architecture_choice", adaptive.focus_bounds([], {}, cap.meta))
        base = self.cfg["experiments"][0]
        for name, request in choices.items():
            plan = {"experiment_names": [base["name"]], "parameter_changes": {base["name"]: {"architecture_choice": name}}}
            result = adaptive.proposed_specs([base], plan, self.cfg["optimization"], [], self.parent_sha,
                                            focus_only=True, architecture_choices=available_choices(self.parent(), choices))[0]
            self.assertEqual(result["architecture"], request)
            self.assertNotIn("architecture_choice", result)
        for update in ({"architecture": {"hidden": 64}}, {"architecture_choice": "unknown"}, {"architecture_choice": True}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                adaptive.proposed_specs([base], {"experiment_names": [base["name"]],
                    "parameter_changes": {base["name"]: update}}, self.cfg["optimization"], [], self.parent_sha,
                    focus_only=True, architecture_choices=choices)
        with self.assertRaises(ValueError):
            available_choices(self.parent(), {"shrink": {"hidden": 16}})

    def test_sdk_menu_has_freezing_choices_and_sparse_projection_choices_are_compatible(self):
        specs = [s for s in self.cfg["experiments"] if s["name"] in ("head_only", "features_only")]
        cfg = {**self.cfg, "experiments": specs}
        menu = ordered_specs(cfg, [], self.parent_sha, model=self.parent())
        self.assertEqual({s["freeze"] for s in menu}, {"features", "head"})
        sparse = {"name": "explicit_projection_probe", "kept_sensors": [1, 2]}
        self.assertEqual(adaptive.parameter_bounds([sparse])["freeze_choices_by_experiment"][sparse["name"]], ["none", "head"])


if __name__ == "__main__":
    unittest.main()
