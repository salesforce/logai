#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import sys
import importlib.util
from packaging import version

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

get_pkg_version = importlib_metadata.version


def is_torch_available():
    """
    Checks if pytorch is available.
    """
    if importlib.util.find_spec("torch") is not None:
        _torch_version = importlib_metadata.version("torch")
        if version.parse(_torch_version) < version.parse("1.6"):
            raise EnvironmentError(f"Torch found but with version {_torch_version}. " f"The minimum version is 1.6")
        return True
    else:
        return False


def is_tf_available():
    """
    Checks if tensorflow 2.0 is available.
    """
    candidates = (
        "tensorflow",
        "tensorflow-cpu",
        "tensorflow-gpu",
        "tf-nightly",
        "tf-nightly-cpu",
        "tf-nightly-gpu",
        "intel-tensorflow",
        "intel-tensorflow-avx512",
        "tensorflow-rocm",
        "tensorflow-macos",
    )
    _tf_version = None
    for pkg in candidates:
        try:
            _tf_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    if _tf_version is not None:
        if version.parse(_tf_version) < version.parse("2"):
            raise EnvironmentError(f"Tensorflow found but with version {_tf_version}. " f"The minimum version is 2.0")
        return True
    else:
        return False


def is_transformers_available():
    """
    Checks if the `transformers` library is installed.
    """
    if importlib.util.find_spec("transformers") is not None:
        _version = importlib_metadata.version("transformers")
        if version.parse(_version) < version.parse("4.0"):
            raise EnvironmentError(f"Transformers found but with version {_version}. " f"The minimum version is 4.0")
        return True
    else:
        return False


def is_nltk_available():
    """
        Checks if the `nltk` library is installed.
        """
    if importlib.util.find_spec("nltk") is not None:
        return True
    else:
        return False
