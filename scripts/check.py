"""默认执行比赛核心检查；--full 执行全部专项回归，不调用外部模型。"""
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import time
import json
import argparse
import platform

ROOT = Path(__file__).resolve().parents[1]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tests")]

# 只保留与随包任务说明、续训指南和 SDK 提交合同直接相关的检查。
# SwanLab、本地 HTTP、时间回放、展示和算法专项通过 --full 按需执行。
CORE_TESTS = (
    'test_audit_fixes.AuditFixTests.test_default_sdk_rule_honors_configured_target_recall_drop',
    'test_audit_fixes.AuditFixTests.test_epoch_selection_uses_original_baseline_class_protection',
    'test_audit_fixes.AuditFixTests.test_epoch_selection_uses_the_same_target_drop_rule_as_release',
    'test_audit_fixes.AuditFixTests.test_duplicate_scope_requires_identical_parent_and_both_public_data_hashes',
    'test_audit_fixes.AuditFixTests.test_rule_menu_reuses_configuration_when_public_data_changes',
    'test_audit_fixes.AuditFixTests.test_response_refit_freeze_conflicts_are_rejected_before_execution',
    'test_protocol.ProtocolTests.test_sdk_revises_response_freeze_conflict_before_training',
    "test_response_adapter.ResponseAdapterTests.test_zero_head_migration_preserves_function_and_all_parent_state",
    "test_response_adapter.ResponseAdapterTests.test_export_passes_unchanged_quality_gates_and_reproduces_selected_prefix",
    "test_response_adapter.ResponseAdapterTests.test_existing_head_only_exposes_compatible_augmentation_and_can_preserve_architecture",
    "test_protocol.ProtocolTests.test_sdk_response_head_is_uploaded_measured_selected_and_predictable",
    "test_hidden_safe.HiddenSafeTests.test_validation_stress_covers_every_sensor_without_refitting_scaler",
    "test_hidden_safe.HiddenSafeTests.test_all_clean_fraction_replays_parent_adam_and_leaves_scaler_exact",
    "test_hidden_safe.HiddenSafeTests.test_robust_best_epoch_exports_actual_parent_state_and_measured_suite",
    "test_hidden_safe.HiddenSafeTests.test_stress_rejection_blocks_otherwise_accepted_clean_candidate",
    "test_hidden_safe.HiddenSafeTests.test_blind_predictions_ignore_targets_and_scoring_follows_complete_stream",
    "test_hidden_safe.HiddenSafeTests.test_nonlabel_token_does_not_block_prediction_but_fails_later_scoring",
    "test_hidden_safe.HiddenSafeTests.test_prediction_restarts_and_input_order_do_not_require_feedback",
    "test_hidden_safe.HiddenSafeTests.test_qwen_planning_and_selection_receive_only_authorized_robustness_evidence",
    "test_protocol.ProtocolTests.test_describe",
    "test_protocol.ProtocolTests.test_sdk_prediction_replies_before_reading_next_sample",
    "test_protocol.ProtocolTests.test_missing_host_measurements_do_not_become_local_scores",
    "test_sdk_compatibility.SdkCompatibilityTests.test_missing_parent_score_promotes_by_explicit_common_local_validation",
    "test_sdk_compatibility.SdkCompatibilityTests.test_complete_host_scores_keep_host_gate_and_regression_is_rejected",
    "test_sdk_compatibility.SdkCompatibilityTests.test_missing_candidate_host_measurement_cannot_be_replaced_by_local_score",
    "test_sdk_compatibility.SdkCompatibilityTests.test_parent_is_selected_by_digest_among_declared_nested_checkpoints",
    "test_sdk_compatibility.SdkCompatibilityTests.test_local_fallback_cannot_mix_high_host_score_with_local_parent",
    "test_sdk_compatibility.SdkCompatibilityTests.test_parent_digest_mismatch_is_rejected_before_any_model_call",
    "test_sdk104.Sdk104Tests.test_candidate_and_state_use_hash_checked_chunked_upload_ids",
    "test_sdk104.Sdk104Tests.test_old_host_is_rejected_before_model_calls_or_training",
    "test_sdk104.Sdk104Tests.test_host_candidate_and_state_size_limits_are_honored",
    "test_sdk104.Sdk104Tests.test_interrupted_round_does_not_retry_or_consume_late_responses",
    "test_protocol.ProtocolTests.test_sdk_round_real_candidate_receipts_history_and_state",
    "test_protocol.ProtocolTests.test_sdk_single_eligible_decision_and_long_chinese_reason",
    "test_protocol.ProtocolTests.test_sdk_corrected_selection_cites_all_actual_calls",
    "test_protocol.ProtocolTests.test_sdk_executes_freezing_and_class_focus_in_the_existing_two_calls",
    "test_protocol.ProtocolTests.test_sdk_default_27b_can_choose_each_architecture_without_architecture_config",
    "test_protocol.ProtocolTests.test_sdk_default_composed_configuration_and_ai_revision_execute",
    "test_protocol.ProtocolTests.test_sdk_parent_fallback_is_authorized_by_a_real_round_model_receipt",
    "test_optimization_policy.OptimizationPolicyTests.test_unmasked_later_parent_resumes_learning_with_exact_adam_replay",
    "test_optimization_policy.OptimizationPolicyTests.test_default_training_updates_connections_for_every_sensor",
    "test_optimization_policy.OptimizationPolicyTests.test_shipped_configs_do_not_supply_fixed_sensor_candidates",
    "test_optimization_policy.OptimizationPolicyTests.test_old_automatic_reference_config_is_rejected_before_training",
    "test_planning.PlanningTests.test_many_combined_fields_execute_real_parent_continuation_and_freezing",
    "test_planning.PlanningTests.test_temporal_rejected_configuration_gets_a_real_second_ai_proposal_before_training",
    "test_protocol.ProtocolTests.test_contest_image_sources_work_without_direct_clients",
    "test_training.LifecycleTests.test_input_leakage_overlap_and_nonfinite_rejected",
    "test_training.LifecycleTests.test_continuation_and_validation_never_changes_gradients",
    "test_training.LifecycleTests.test_candidate_evidence_records_each_epoch_and_follows_registered_version",
    "test_training.RepairTests.test_reference_checkpoint_and_original_scaler_with_preserved_adam",
    "test_freezing.FreezingTests.test_layer_modes_preserve_weights_and_nonzero_moments_on_original_and_migrated_parent",
    "test_training.OptimizationTests.test_best_epoch_restores_adam_and_counts_discarded_training",
    "test_architecture.ArchitectureTests.test_migration_preserves_parent_predictions_and_every_old_optimizer_element",
    "test_architecture.ArchitectureTests.test_export_records_migration_and_resumes_actual_selected_migrated_parent",
    "test_architecture.ArchitectureTests.test_best_migrated_epoch_restores_all_adam_tensors",
    "test_architecture.ArchitectureTests.test_robust_migrated_epoch_restores_all_adam_tensors",
    "test_protocol.ProtocolTests.test_sdk_migrated_candidates_keep_parent_and_task_a_contract",
    "test_resilience.ResilienceTests.test_unverified_historical_best_comment_is_rebound_and_raw_reply_is_preserved",
    "test_resilience.ResilienceTests.test_activation_interruption_resumes_pointer_and_log_exactly_once",
    "test_resilience.ResilienceTests.test_process_crash_releases_os_lock_without_manual_stale_file_deletion",
    "test_resilience.ResilienceTests.test_missing_or_malformed_window_is_rejected_before_model_and_audit_start",
)


def test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from test_ids(test)
        else:
            yield test.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--full", action="store_true", help="执行完整专项回归；默认仅执行比赛核心检查")
    parser.add_argument("--list", action="store_true", help="列出选定的检查，不执行")
    args = parser.parse_args()
    if args.list and not args.full:
        print("\n".join(CORE_TESTS))
        return 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="gas-a-sdk-") as tmp:
        root = Path(tmp).resolve()
        with tarfile.open(ROOT / "vendor/contest-harness-sdk-1.0.4.tar.gz") as archive:
            for member in archive.getmembers():
                dest = (root / member.name).resolve()
                if not dest.is_relative_to(root) or member.issym() or member.islnk():
                    raise ValueError("Unsafe SDK archive member")
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.extractfile(member).read())
                else:
                    raise ValueError("SDK archive contains a special file")
        sys.path.insert(0, str(root / "contest-harness-sdk-1.0.4"))
        from contest_harness.protocol import SDK_VERSION
        if SDK_VERSION != "1.0.4":
            raise RuntimeError("Offline checks require the vendored SDK 1.0.4")
        suite = (unittest.defaultTestLoader.discover(str(ROOT / "tests")) if args.full
                 else unittest.defaultTestLoader.loadTestsFromNames(CORE_TESTS))
        selected = list(test_ids(suite))
        if args.list:
            print("\n".join(selected))
            return 0
        suite_name = "full_regression" if args.full else "competition_core"
        print(f"Running {suite_name}: {len(selected)} checks", flush=True)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if args.report:
            import numpy as np
            report = {"suite": suite_name, "selected_tests": selected, "sdk_version": SDK_VERSION,
                      "python": platform.python_version(), "numpy": np.__version__,
                      "pinned_dependency_match": np.__version__ == "1.26.4", "tests_run": result.testsRun,
                      "failures": len(result.failures), "errors": len(result.errors),
                      "passed": result.wasSuccessful(), "seconds": time.monotonic() - started,
                      "external_model_calls": 0, "docker_image_verified": False,
                      "official_platform_verified": False,
                      "scope": ("Local checks selected against supplied contest task/SDK contracts; host/model replies are offline fixtures"
                                if not args.full else "Full local regression, including local HTTP/cloud/replay fixtures; no real external model calls")}
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
