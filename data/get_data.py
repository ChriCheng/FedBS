import os
import time
import sys

sys.path.append(".")

import scipy.io as scio
from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2015001
from moabb.paradigms import MotorImagery


def patch_pooch_keep_progress(
    *, max_tries=25, backoff_base=1.6, timeout=(10, 300), verify_tls=True
):
    """
    Monkey-patch pooch.downloaders.HTTPDownloader.__call__:
    - 保留 pooch 原来的 tqdm 进度条（我们不接管流式写入）
    - 只在外层做网络异常重试 + 指数退避 + 更长 timeout
    """
    import requests
    from pooch.downloaders import HTTPDownloader

    if getattr(HTTPDownloader, "_keep_progress_patched", False):
        return

    orig_call = HTTPDownloader.__call__

    def wrapped_call(self, url, output_file, pooch_obj):
        last_err = None
        for attempt in range(1, max_tries + 1):
            try:
                # 注入更稳的下载参数（pooch 原实现会用到）
                kwargs = dict(getattr(self, "kwargs", {}))
                kwargs["timeout"] = timeout
                kwargs["verify"] = verify_tls
                kwargs.setdefault("allow_redirects", True)

                # 这里不碰 stream/iter_content，让 pooch 自己做 -> 进度条还在
                self.kwargs = kwargs

                return orig_call(self, url, output_file, pooch_obj)

            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                last_err = e
                sleep_s = min(60.0, backoff_base ** (attempt - 1))
                print(
                    f"[retry] attempt {attempt}/{max_tries} failed: {e} -> sleep {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
                continue

        raise RuntimeError(
            f"Download failed after {max_tries} attempts. Last error: {last_err}"
        )

    HTTPDownloader.__call__ = wrapped_call
    HTTPDownloader._keep_progress_patched = True


def download_data(MI, resample=250):
    # 关键：用“保留进度条”的 patch
    patch_pooch_keep_progress(
        max_tries=25, backoff_base=1.6, timeout=(10, 300), verify_tls=True
    )

    if MI == "MI1":
        root = "./BNCI2014001/"
        dataset = BNCI2014001()
        n_classes = 4
    elif MI == "MI2":
        root = "./BNCI2014002/"
        dataset = BNCI2014002()
        n_classes = 2
    elif MI == "MI3":
        root = "./BNCI2015001/"
        dataset = BNCI2015001()
        n_classes = 2
    else:
        raise ValueError("MI must be one of: 'MI1', 'MI2', 'MI3'")

    os.makedirs(root, exist_ok=True)

    subjects = dataset.subject_list
    print("Subject", len(subjects))

    paradigm = MotorImagery(n_classes=n_classes, fmin=8, fmax=30, resample=resample)

    for idx, subj in enumerate(subjects, start=1):
        out_file = os.path.join(root, f"{idx}.mat")
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            print(f"[skip] {MI} subject {subj} already saved: {out_file}")
            continue

        print(f"Downloading subject {subj} data")
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subj])

        print("y", y.shape)
        print("X", X.shape)

        scio.savemat(out_file, {"X": X, "y": y})
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    download_data("MI1")
    download_data("MI2")
    download_data("MI3")
