"""AI直接组合可执行训练配置；预设菜单仅兼容旧运行，不限制新的AI建议。"""
import copy
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from .architecture import shape, target_shape
from . import response
from . import robustness
from .controller_policy import bind_plan, build_planning_prompt, validate_plan, scoped_experiments
from .io import CLASSES, digest, read_json
from .optimize import bounded_experiment, validate_training_spec


DEFAULTS = {"balance": 0., "weak_boost": 1., "l2": 0., "focal_gamma": 0., "anchor_l2": 0.,
            "feature_lr_scale": 1., "final_lr_ratio": 1., "schedule": "constant", "input_noise": 0.,
            "sensor_dropout": 0., "sensor_gain": 0., "sensor_shift": 0., "response_gain": 0.,
            "response_sensor_gain": 0., "pair_weight": 0., "pair_margin": 0., "pair_other": "Ethanol",
            "adversarial_epsilon": 0., "class_boosts": {}, "freeze": "none", "frozen_parameters": [],
            "minimum_epochs": 1, "clean_fraction": 0.}
FIELDS = {"name", "learning_rate", "batch_size", "epochs", "loss", "architecture", "response_adapter", "kept_sensors", *DEFAULTS}
CONTROL_SOURCES = ('architecture.py', 'response.py', 'optimize.py', 'robustness.py',
                   'quality.py', 'controller_policy.py', 'adaptive.py')


def composed(config):
    return config.get("controller", {}).get("planning_mode", "compose") == "compose"


def mechanism_tags(spec, model):
    """Describe implemented interventions, independently of AI names and numeric nudges."""
    groups = {
        "class_loss": spec.get("balance", 0) != 0 or spec.get("weak_boost", 1) != 1
            or spec.get("focal_gamma", 0) > 0 or spec.get("loss") == "focal"
            or any(value != 1 for value in spec.get("class_boosts", {}).values()),
        "sensor_affine": any(spec.get(key, 0) > 0 for key in ("sensor_gain", "sensor_shift")),
        "sensor_dropout": spec.get("sensor_dropout", 0) > 0,
        "response_gain": any(spec.get(key, 0) > 0 for key in ("response_gain", "response_sensor_gain")),
        "response_adapter": 'response_adapter' in spec,
        "input_noise": spec.get("input_noise", 0) > 0,
        "adversarial": spec.get("adversarial_epsilon", 0) > 0,
        "confusion_pair": spec.get("pair_weight", 0) > 0,
        "feature_update": spec.get("freeze", "none") != "none" or bool(spec.get("frozen_parameters"))
            or spec.get("feature_lr_scale", 1) != 1,
        "capacity": bool(spec.get("architecture")) and target_shape(model, spec["architecture"]) != shape(model),
        "sensor_subset": "kept_sensors" in spec and sorted(spec["kept_sensors"]) != list(range(1, 17)),
        "regularization": spec.get("l2", 0) > 0 or spec.get("anchor_l2", 0) > 0,
    }
    return sorted(name for name, enabled in groups.items() if enabled) or ["optimization_only"]


def iteration_brief(config, model, evidence, past, parent_sha):
    """Measured mechanism memory; compare only the same parent AND known data hashes."""
    facts = evidence["facts"]
    target_labels = list(config["gate"].get("minimum_class_recall", {}))
    columns = evidence["class_columns"]
    metrics = {key: {label: dict(zip(columns, row)) for label, row in zip(CLASSES, facts[key]["classes"])}
               for key in ("C", "T")}
    observed = [label for label in CLASSES if metrics["C"][label]["support"] > 0]
    if not target_labels and observed:
        target_labels = [min(observed, key=lambda name: metrics["C"][name]["f1"])]
    current_train, current_val = facts["T"].get("sha256"), facts["F"].get("sha256")
    grouped, components, skipped, advice = {}, {}, 0, []
    for row in past:
        for trial in row.get("experiments", []):
            report = trial.get("report", {})
            train_sha = report.get("train_sha256", row.get("train_sha256"))
            val_sha = report.get("validation_sha256", row.get("validation_sha256"))
            parent = report.get("parent_checkpoint_sha256", row.get("parent_checkpoint_sha256"))
            if (parent != parent_sha or not current_train or not current_val
                    or train_sha != current_train or val_sha != current_val):
                skipped += 1
                continue
            spec = report.get("configuration", trial.get("configuration", {}))
            tags = tuple(mechanism_tags(spec, model))
            group = grouped.setdefault(tags, {"mechanisms": list(tags), "attempts": 0, "eligible": 0,
                "rounds": [], "best_f1_gain": None, "best_target_recall": {}, "failures": {}})
            group["attempts"] += 1
            group["eligible"] += bool(trial.get("eligible", trial.get("gate", trial.get("local_gate", {})).get("accepted", False)))
            index = row.get("round", row.get("batch"))
            if index is not None and index not in group["rounds"]:
                group["rounds"].append(index)
            for tag in tags:
                component = components.setdefault(tag, {"mechanism": tag, "attempts_including_combinations": 0,
                                                       "standalone_attempts": 0, "rounds": []})
                component["attempts_including_combinations"] += 1
                component["standalone_attempts"] += len(tags) == 1
                if index is not None and index not in component["rounds"]:
                    component["rounds"].append(index)
            before, after = report.get("validation_before", {}), report.get("validation_after", {})
            if "macro_f1" in before and "macro_f1" in after:
                gain = after["macro_f1"] - before["macro_f1"]
                group["best_f1_gain"] = round(max(group["best_f1_gain"], gain) if group["best_f1_gain"] is not None else gain, 6)
            for label in target_labels:
                value = after.get("per_class", {}).get(label, {}).get("recall")
                if value is not None:
                    group["best_target_recall"][label] = round(max(group["best_target_recall"].get(label, 0), value), 6)
            reasons = {reason for key in ("fixed_gate", "window_gate", "local_gate", "gate", "robust_gate")
                       for reason in trial.get(key, {}).get("reasons", []) if reason != "quality_gates_passed"}
            if trial.get("error"):
                reasons.add("execution_error")
            for reason in reasons:
                group["failures"][reason] = group["failures"].get(reason, 0) + 1
            suggestion = row.get("selection_feedback") or {}
            if suggestion.get("next_test") and index not in [item["round"] for item in advice]:
                advice.append({"round": index, "next_test": suggestion["next_test"], "origin": "model_hypothesis_not_measured_fact"})
    families = list(grouped.values())
    for group in families:
        group["failures"] = dict(sorted(group["failures"].items(), key=lambda item: (-item[1], item[0]))[:3])
    return {"targets": [{"gas": name, "train_recall": metrics["T"][name]["recall"],
                         "validation_recall": metrics["C"][name]["recall"],
                         "required_recall": config["gate"].get("minimum_class_recall", {}).get(name)} for name in target_labels],
            "same_parent_same_public_data": families, "excluded_other_or_unknown_scope_trials": skipped,
            "component_coverage": list(components.values()),
            "available_history_rounds": [row.get("round", row.get("batch")) for row in past
                                         if row.get("round", row.get("batch")) is not None],
            "last_decision_advice": advice[-2:],
            "reading": "component_coverage包含组合中用过的组件；没有单独对照不等于没试过，组合结果不能单独归因。正召回但未达门槛不能写成历史全为0。排除不同/未知数据摘要，不宣称最好或未试。"}


def search_space(config, model):
    space = {"required": ["name", "learning_rate", "batch_size", "epochs"],
            "learning_rate": [1e-6, .01], "batch_size": [16, 256], "epochs": [1, config["optimization"]["max_epochs"]],
            "loss": ["cross_entropy", "focal"], "focal_gamma": [0, 2],
            "schedule": ["constant", "cosine"], "final_lr_ratio": [.05, 1],
            "balance": [0, 1], "weak_boost": [.1, 5],
            "class_boosts": {"range": [.1, 5],
                             "format": "气体类名→倍率的字典，允许降权；不要用classes/multiplier包装"},
            "l2": [0, .01], "anchor_l2": [0, 1], "feature_lr_scale": [.05, 1],
            "freeze": ["none", "features", "head", "first_hidden"],
            "frozen_parameters": "可选权重/偏置列表；w1/b1第一隐藏层，w2/b2输出，新增层w3/b3起；必须存在且不能全冻结",
            "architecture": {"hidden": [model.meta["hidden"], 128], "hidden_layers": [shape(model)["hidden_layers"], 4],
                             "rule": "可省略；旧参数及Adam兼容迁移，只能保持或扩大，输入128、输出6不变"},
            "response_adapter": {"ridge": [.01, 1000], "strength": [1, 32], "guard_multiplier": [1, 4],
                "rule": "可选对象，三个数值均必填。显式零头迁移保留父权重/Adam/原scaler；全16传感器按特征类型求相对响应，附加统计仅在train首次拟合且冻结；类平衡岭回归增量训练附加头，再全网Adam续训。推理用冻结的训练覆盖检测，覆盖外仅用父网络分支。",
                "scale": "岭目标幅度strength与原MLP logits叠加，过小可能不能纠正原边界，过大会伤及其余气体；ridge越大越保守；guard_multiplier扩展训练最大马氏距离。",
                "compatibility": "不能与kept_sensors、input_noise、sensor_gain/shift/dropout、adversarial_epsilon并用；允许原响应增益、clean_fraction及普通训练字段。父模型已有附加头时，省略本对象表示仅Adam续训，不重复岭回归。"},
            "augmentation": {"input_noise": [0, 1], "sensor_dropout": [0, .5], "sensor_gain": [0, 1.5],
                             "sensor_shift": [0, 2], "response_gain": [0, 1.5], "response_sensor_gain": [0, 1],
                             "adversarial_epsilon": [0, 2], "clean_fraction": [0, 1]},
            "augmentation_semantics": {
                "sensor_gain/shift": "父scaler标准化后，每传感器8维共享随机增益/偏移；input_noise则逐维独立高斯噪声。",
                "response_gain/response_sensor_gain": "原始响应上按样本全阵列/单传感器随机缩放；归一响应的无响应点保留1，再用原父scaler。",
                "clean_fraction": "每条训练样本保留原输入的概率，可与扰动样本混合；推理预处理不变。",
                "scope": "均为训练扰动假设；强度不是已测出的物理漂移量，须公开验证检验。"},
            "pair_weight": [0, 5], "pair_margin": [0, 4], "pair_other": CLASSES,
            "kept_sensors": "默认保留全部16个传感器，由训练学习连接权重；子集仅用于有证据的显式实验，会强制屏蔽其余连接，并非自动学习传感器重要性。PSI排名不能证明传感器无用；须检查六类代价，不能同时冻结w1。",
            "minimum_epochs": "可选整数1..epochs；其余可选项省略时不启用，feature_lr_scale/final_lr_ratio/weak_boost默认为1",
            "combination": "可在同一配置组合任意已开放字段，不限制最多两项；augmentation中的字段直接写入配置顶层",
            "optimizer": "固定继承父Adam状态；不重置权重、矩、seed、scaler或累计计数；未知方法不能假称支持"}
    if response.enabled(model):
        for key in ('input_noise','sensor_gain','sensor_shift','sensor_dropout','adversarial_epsilon'):
            space['augmentation'].pop(key)
        space.pop('kept_sensors')
        space['current_parent_restrictions'] = '父模型已含相对响应头：仅允许原始响应增强response_gain/response_sensor_gain及clean_fraction；标准化后扰动、对抗增强和传感器子集不可执行。response_adapter对象可重新训练附加头，省略表示仅Adam续训。'
    return space


def build_prompt(config, model, menu, evidence, budget, max_seconds, *, mode, index, feedback=None, parameter_bounds=None, past=(), parent_sha=None):
    # Optional public developer evidence is not fabricated platform history. Its
    # data, actual parent AND algorithm source must all match before exposing it.
    evidence['facts'].pop('P', None)
    controls_path = Path(__file__).resolve().parents[2] / 'configs/evidence/response_controls.json'
    if controls_path.is_file():
        controls = read_json(controls_path)
        facts = evidence['facts']
        if (controls.get('parent_checkpoint_sha256') == parent_sha
                and controls.get('train_sha256') == facts['T'].get('sha256')
                and controls.get('validation_sha256') == facts['F'].get('sha256')
                and controls.get('quality_gate') == config['gate']
                and controls.get('robustness_policy') == robustness.settings(config['optimization'])
                and set(controls.get('source_sha256', {})) == set(CONTROL_SOURCES)
                and all(digest(Path(__file__).parent/n) == sha
                    for n, sha in controls['source_sha256'].items())):
            facts['P'] = controls
    prompt = build_planning_prompt(menu, evidence, config["gate"] if mode == "sdk" else
        {**config["gate"], **({"hard_minimum_class_recall": config.get("adaptive", {}).get("hard_minimum_class_recall", {})}
                            if config.get("adaptive", {}).get("enabled") else {})}, budget, max_seconds,
        mode=mode, index=index, feedback=feedback, parameter_bounds=parameter_bounds if not composed(config) else None)
    if not composed(config):
        return prompt
    prompt.update(prompt_version="gas-a.configuration-planning.v4-response",
        request="你负责受控实验迭代。先读取iteration_brief的实测失败与目标差距，提出可证伪的新假设和可执行配置。只返回简短JSON，不展开推导，不照抄历史配方。",
        parent={**shape(model), "learning_rate": model.meta["learning_rate"], "batch_size": model.meta["batch_size"], "optimizer": "Adam"},
        search_space=search_space(config, model),
        plan_contract={"action": "experiment或observe", "hypothesis": "100字内：待验证机制和目标类别",
                       "fallback": "可选retain_parent：最终选版回复无效时，明确授权保留本轮合格父模型",
                       "reason": "40字内：为什么本轮值得做；只做1项时说明原因",
                       "iteration": {"history_rounds": "实际用于本轮判断的历史轮次列表；无历史时[]",
                                     "lesson": "40字内：实测支持什么、仍未解决什么",
                                     "tests": "{本轮实验name:30字内的干预机制与可观察预测}；observe时{}"},
                       "experiments": f"最多{budget}个配置对象；预算允许时优先2个机制不同或关键干预可对照的配置；observe时[]；每项必须有name、learning_rate、batch_size、epochs"})
    prompt.pop("allowed_experiments", None)
    prompt.pop("experiment_guide", None)
    prompt["decision_context"]["training_allowed"] = budget > 0 and not evidence["facts"].get("S", {}).get("pause")
    prompt["decision_steps"] = [
        "从逐类训练/验证差距和主要混淆定位瓶颈；分布差异只支持待验证假设，不能认定过拟合或物理故障。",
        "训练已能识别、验证却漏检时，优先检验跨分布稳定性。可组合结构化传感器增益/偏移、原响应增益、干净样本混合、混淆对约束或特征层更新；能力字段见search_space，参数由你设计。",
        "先总结同父同数据的机制反馈。某类干预多次未达目标时，只改学习率、轮数或类别倍率不能当作新解释；须说明具体新机制、对照及预期可观察变化。",
        "若训练识别好、验证某类持续为0，可检验response_adapter：它改变可学习表示，全部16传感器参与；原模型始终保留。需以guard_multiplier或ridge的对照区分干净收益与覆盖外风险，不能只看一类召回。",
        "如有证据P，它是同父、同公开数据、同算法源码的开发CPU对照，不是平台历史。可据此提出候选及对照，仍须本轮实际训练、宿主测量；claims仅支持既有H_事实，不能伪造P为平台完成记录。",
        "核对component_coverage：组合中用过的组件只能说尚无独立对照，不能说从未测试；lesson区分实测与假设，不把局部进展写成零收益，不把相关性写成确定原因。",
        "预算允许2项时优先用A/B区分机制，避免堆叠所有扰动造成归因不清；只有一条可检验路线时可做1项并说明。",
        "新候选须同时守住F1、各气体召回与稳健门槛。F1小涨但目标仍漏检，表示局部进展而非修复成功。默认全部16传感器，PSI不是重要性。",
        "training_allowed=false时observe且experiments=[]；其余按证据试验。输出iteration简短记录历史教训及各候选预测，不能编造平台测量或未来收益。",
    ]
    prompt["iteration_brief"] = iteration_brief(config, model, evidence, past, parent_sha)
    if response.enabled(model):
        prompt['decision_steps'][1] = '当前父模型已有相对响应头。只设计search_space列出的可执行配置；可对照岭回归强度、覆盖扩展、原响应增强或普通Adam续训，不再提出标准化后sensor_gain/shift/dropout、input_noise、adversarial_epsilon或传感器子集。'
    if 'P' in evidence['facts']:
        prompt['developer_controls'] = copy.deepcopy(evidence['facts']['P'])
    n = sum(row[0] for row in evidence["facts"]["T"]["classes"])
    prompt["training_work"] = {"rows": n, "updates_per_epoch_examples": {str(b): math.ceil(n / b) for b in (32, 128)},
                               "reading": "按实际优化步数比较训练量；epochs是上限，可能早停并导出更早epoch。minimum_epochs只限制早停时机，不保证导出该轮。"}
    prompt["candidate_source"] = "仅训练本轮AI提出并通过校验的配置；程序不追加固定传感器配方，不占用AI候选预算。"
    prompt["prediction_contract"] = {
        "inputs": "预测时只有当前128维features；不提供真实标签、真实混淆矩阵或未来窗口。",
        "updates": "梯度只用授权train；scaler继承真实父模型；隐藏预测阶段不学习参数或拟合标准化。",
        "augmentation": "可检验训练扰动，clean_fraction表示每批保持原始输入的概率；参数须由授权验证实测选择。"}
    if evidence.get("facts", {}).get("R"):
        prompt["robust_validation"] = evidence["facts"]["R"]
        prompt['robust_validation']['policy'] = robustness.settings(config['optimization'])
        prompt["decision_steps"].append("结合干净验证与全部16个传感器分别缺失的实测结果检验稳健性；不根据隐藏标签选传感器。扰动是待检验假设，不等于物理原因或已实现收益。")
    prompt["evidence"].pop("history_columns", None)
    prompt["evidence"].pop("history_rows", None)
    recent = []
    for row in reversed(past):
        for trial in reversed(row.get("experiments", [])):
            report = trial.get("report", {})
            targets = {}
            for label in config["gate"].get("minimum_class_recall", {}):
                def recall(metric):
                    return metric.get("per_class", {}).get(label, {}).get("recall")
                targets[label] = {"train_before": recall(report.get("train_before", {})), "train_after": recall(report.get("train_after", {})),
                                  "fixed_before": recall(report.get("validation_before", {})), "fixed_after": recall(report.get("validation_after", {})),
                                  "current_before": recall(row.get("shadow_before", {})), "current_after": recall(trial.get("shadow_metrics", {}))}
            recent.append({"configuration": {k: v for k, v in report.get("configuration", trial.get("configuration", {})).items() if k in FIELDS},
                           "proposal_origin": trial.get("proposal_origin", "historical_configuration"),
                           "fixed_f1": report.get("validation_after", {}).get("macro_f1"),
                           "robust_worst_f1": report.get("robust_validation_after", {}).get("worst_macro_f1"),
                           "current_f1": trial.get("shadow_metrics", {}).get("macro_f1"), "error": trial.get("error"),
                           "target_recall_feedback": targets,
                           "selected_updates": report.get("final_steps", 0) - report.get("initial_steps", 0) if report else None,
                           "executed_updates": report.get("executed_steps", 0) - report.get("initial_steps", 0) if report else None,
                           "rejection_reasons": [reason for key in ("fixed_gate", "window_gate", "local_gate", "gate")
                                                 for reason in trial.get(key, {}).get("reasons", []) if reason != "quality_gates_passed"],
                           "eligible": trial.get("eligible"), "same_parent": report.get("parent_checkpoint_sha256",
                               row.get("parent_checkpoint_sha256")) == parent_sha})
            if len(recent) == 2:
                break
        if len(recent) == 2:
            break
    prompt["recent_configurations"] = recent
    return prompt


def signature(spec, model, weak_class=None):
    value = {**DEFAULTS, **{k: v for k, v in spec.items() if k in FIELDS and k != "name"}}
    value.pop("loss", None)  # loss的执行含义已规范成focal_gamma。
    if value["weak_boost"] != 1 or value["pair_weight"]:
        value["diagnostic_class"] = weak_class
    value["batch_size"] = value.get("batch_size", model.meta["batch_size"])
    value["architecture"] = target_shape(model, value["architecture"]) if value.get("architecture") else shape(model)
    value["frozen_parameters"] = sorted(value["frozen_parameters"])
    if "kept_sensors" in value:
        value["kept_sensors"] = sorted(value["kept_sensors"])
        if value["kept_sensors"] == list(range(1, 17)):
            value.pop("kept_sensors")
    return json.dumps(value, sort_keys=True, allow_nan=False)


def validate_model_configuration(spec, model):
    """Validate migration and freezing/response conflicts without allocating a model."""
    target = target_shape(model, spec['architecture']) if 'architecture' in spec else shape(model)
    if target.get('response_adapter') and not response.enabled(model) and 'response_adapter' not in spec:
        raise ValueError('New response adapter requires an explicit training specification')
    keys = {'w1', 'b1', 'w2', 'b2', *[f'{kind}{layer+1}'
            for layer in range(2, target['hidden_layers'] + 1) for kind in ('w', 'b')]}
    if target.get('response_adapter'):
        keys.update(('wa', 'ba'))
    validate_training_spec(SimpleNamespace(params=dict.fromkeys(keys), meta={**model.meta, **target}), spec)


def configurations(raw, config, model, past, parent_sha, budget, weak_class=None, *, train_sha=None, validation_sha=None):
    if not isinstance(raw, dict) or not isinstance(raw.get("experiments"), list):
        raise ValueError("Return experiments as configuration objects, not experiment_names or a preset name")
    items = raw["experiments"]
    if raw.get("action") == "observe" and items or raw.get("action") == "experiment" and not 1 <= len(items) <= budget:
        raise ValueError(f"Experiment count must fit the {budget}-candidate budget; observe requires []")
    forbidden = set(raw) - {"action", "experiments", "hypothesis", "reason", "claims", "evidence_ids", "success_condition", "stop_condition", "fallback", "iteration"}
    if forbidden:
        raise ValueError("Unsupported plan fields: " + ",".join(sorted(forbidden)))
    if "fallback" in raw and raw["fallback"] != "retain_parent":
        raise ValueError("Only explicit retain_parent fallback authorization is supported")
    if "iteration" in raw:
        note = raw["iteration"]
        known_rounds = {row.get("round", row.get("batch")) for row in past}
        if not isinstance(note, dict) or set(note) != {"history_rounds", "lesson", "tests"}:
            raise ValueError("iteration needs history_rounds, lesson and tests")
        references = note["history_rounds"]
        if (not isinstance(references, list) or any(type(i) is not int or i not in known_rounds for i in references)
                or len(references) != len(set(references))):
            raise ValueError("iteration may cite only actual available history rounds")
        if not isinstance(note["lesson"], str) or not note["lesson"].strip() or len(note["lesson"]) > 256:
            raise ValueError("iteration.lesson must be nonempty concise text")
        tests = note["tests"]
        expected_names = {item.get("name") for item in items if isinstance(item, dict) and isinstance(item.get("name"), str)}
        if (not isinstance(tests, dict) or set(tests) != expected_names
                or any(not isinstance(value, str) or not value.strip() or len(value) > 256 for value in tests.values())):
            raise ValueError("iteration.tests must describe each proposed experiment by its exact name")
    used = set()
    for row, trial in scoped_experiments(past, parent_sha, train_sha, validation_sha):
        report = trial.get('report', {})
        old = report.get('configuration', trial.get('configuration'))
        if old:
            used.add(signature(old, model, report.get('weak_class', row.get('diagnosis', {}).get('weakest_class', weak_class))))
    result, names = [], set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each experiment must be a configuration object")
        if set(item) - FIELDS:
            raise ValueError("Unsupported configuration fields: " + ",".join(sorted(set(item) - FIELDS)))
        if any(key not in item for key in ("name", "learning_rate", "batch_size", "epochs")):
            raise ValueError("Each configuration needs name, learning_rate, batch_size and epochs")
        name = item["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name) or name in names
                or name in {"state", "parent", "source", "model_calls", "baseline", "monitor", "diagnosis"}):
            raise ValueError("Names must be unique ASCII identifiers, up to 40 characters")
        spec = copy.deepcopy({"balance": 0., "weak_boost": 1., "l2": 0., **item})
        boosts = spec.get("class_boosts")
        if isinstance(boosts, dict) and set(boosts) == {"classes", "multiplier"}:
            labels = boosts["classes"]
            if (not isinstance(labels, list) or not labels or any(not isinstance(c, str) or c not in CLASSES for c in labels)
                    or len(set(labels)) != len(labels)):
                raise ValueError("class_boosts requires a valid class-name:multiplier map")
            spec["class_boosts"] = {label: boosts["multiplier"] for label in labels}
        loss = spec.get("loss", "focal" if spec.get("focal_gamma", 0) else "cross_entropy")
        if loss not in ("cross_entropy", "focal"):
            raise ValueError("Only cross_entropy and focal losses are implemented; Adam must be inherited")
        if loss == "cross_entropy" and spec.get("focal_gamma", 0) != 0:
            raise ValueError("cross_entropy conflicts with nonzero focal_gamma")
        if loss == "focal":
            spec.setdefault("focal_gamma", 1.)
        spec["loss"] = loss
        spec = bounded_experiment(spec, config["optimization"])
        validate_model_configuration(spec, model)
        if spec.get("pair_weight", 0) and spec.get("pair_other", "Ethanol") == weak_class:
            raise ValueError("Confusion pair must use two distinct classes")
        key = signature(spec, model, weak_class)
        if key in used:
            raise ValueError("This exact configuration was already attempted on the same parent and public data; change the actual parameters")
        used.add(key)
        names.add(name)
        spec["proposal_origin"] = "ai_composed_configuration"
        result.append(spec)
    return result


def resolve_plan(raw, config, model, menu, evidence, budget, past, parent_sha, *, repair=False, architecture_choices=None, focus_only=False, weak_class=None):
    from .adaptive import proposed_specs
    from .resilience import repair_plan
    train_sha = evidence['facts']['T'].get('sha256')
    validation_sha = evidence['facts']['F'].get('sha256')
    if composed(config):
        if repair:
            raise ValueError("An invalid composed proposal must be revised by AI, not rewritten by the program")
        specs = configurations(raw, config, model, past, parent_sha, budget, weak_class,
                               train_sha=train_sha, validation_sha=validation_sha)
        plan = bind_plan({key: copy.deepcopy(value) for key, value in raw.items() if key != "experiments"})
        plan["experiments"] = copy.deepcopy(raw["experiments"])
        plan["experiment_names"] = [s["name"] for s in specs]
        plan["planning_mode"] = "ai_composed_configuration"
        plan["configuration_normalizations"] = [{"experiment": s["name"], "field": "class_boosts",
            "input": raw["experiments"][i].get("class_boosts"), "executed": s.get("class_boosts"),
            "origin": "lossless_format_expansion_not_parameter_change"} for i, s in enumerate(specs)
            if raw["experiments"][i].get("class_boosts") != s.get("class_boosts")]
        # 新配置被拒绝时必须由AI修改，不能将程序填出的配置假称为AI建议。
        # 提示词仍要求简短；可读的较长理由不应仅因80字文案上限消耗一次付费纠正。
        validate_plan(plan, specs, budget, evidence, text_limits={"hypothesis": 256, "reason": 512})
        return plan, specs
    plan = bind_plan(raw)
    if repair:
        plan = repair_plan(plan, menu, budget, evidence)
    else:
        validate_plan(plan, menu, budget, evidence, text_limits={"hypothesis": 256, "reason": 512} if not focus_only else None)
    specs = proposed_specs(menu, plan, config["optimization"], past, parent_sha,
                           focus_only=focus_only, architecture_choices=architecture_choices,
                           train_sha=train_sha, validation_sha=validation_sha)
    selected = [next(s for s in specs if s['name'] == name) for name in plan['experiment_names']]
    for spec in selected:
        validate_model_configuration(spec, model)
    return plan, selected


def correction_prompt(prompt, raw, error):
    if "search_space" not in prompt:
        return {**prompt, "correction": {"error": str(error), "previous_response": raw}}
    kept = ("prompt_version", "objective", "parent", "search_space", "plan_contract", "budget", "evidence",
            "iteration_brief", "decision_context", "decision_steps", "prediction_contract", "training_work", "robust_validation", "developer_controls")
    return {key: copy.deepcopy(prompt[key]) for key in kept if key in prompt} | {
        "request": "上个配置未执行。按拒绝原因修改为可执行配置，保留有依据的调优目标，只返回修正后的完整JSON。",
        "validation_error": str(error), "previous_response": raw}
