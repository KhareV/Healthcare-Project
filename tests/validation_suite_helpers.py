from experiments.search_governance import canonical_sha256


TASKS = ("recovery", "icu_stay_time", "organ_support")


def candidate_manifest(task="recovery", family="gru", count=30):
    candidates = [
        {
            "candidate_id": "{}-{}-{:02d}".format(task, family, index),
            "config": {"index": index, "family": family, "task": task},
            "config_hash": canonical_sha256(
                {"index": index, "family": family, "task": task}
            ),
        }
        for index in range(count)
    ]
    return {
        "manifest_version": "scientific_candidate_manifest_v1",
        "run_type": "scientific",
        "test_accessed": False,
        "task": task,
        "family": family,
        "candidate_count": count,
        "candidates": candidates,
        "candidate_list_hash": canonical_sha256(candidates),
        "search_space_hash": "a" * 64,
        "acceptance_report_sha256": "b" * 64,
        "split_hash": "c" * 64,
        "allowed_partitions": ["train", "validation"],
        "feature_version": "feature_schema_v1",
        "label_version": task + "_labels_v1",
        "code_commit": "1" * 40,
    }


def complete_reruns(role, family, parent_prefix):
    return [
        {
            "task": task,
            "model_family": family,
            "run_role": role,
            "status": "completed",
            "run_id": role + "-" + task,
            "parent_run_id": parent_prefix + task,
            "candidate_id": "",
        }
        for task in TASKS
    ]
