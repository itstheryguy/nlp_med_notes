"""SageMaker training job entry point (runs in AWS's PyTorch GPU container).

The container has already pip-installed requirements.txt from the code tarball. Everything it needs comes
from environment variables set by med_notes.sagemaker_job.training_job_request.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # med_notes/ ships next to this file

from med_notes.remote import S3Store, run_remote  # noqa: E402


def main() -> int:
    import torch

    print(f"REMOTE device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
          f"torch={torch.__version__}", flush=True)
    status = run_remote(
        S3Store(os.environ["S3_ROOT"]),
        run_rel=os.environ["RUN_REL"],
        models_rel=os.environ["MODELS_REL"],
        volume_root=os.environ["VOLUME_ROOT"],
        work_dir="/opt/ml/work",
        log=lambda m: print(m, flush=True),
    )
    ok = sum(r["status"] == "succeeded" for r in status)
    print(f"REMOTE finished: {ok}/{len(status)} runs succeeded", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
