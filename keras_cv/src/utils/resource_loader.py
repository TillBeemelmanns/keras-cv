# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities similar to tf.python.platform.resource_loader."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import tensorflow as tf

TF_VERSION_FOR_ABI_COMPATIBILITY = "2.17"
abi_warning_already_raised = False


def get_project_root():
    """Returns project root folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path_to_datafile(path):
    """Resolve the absolute path to a resource within the KerasCV package.

    When developing from source the compiled custom op shared objects live under
    ``bazel-bin`` instead of the Python package tree.  This helper now searches
    a small set of well-known locations so that tests and binaries can locate
    those artifacts without manual copying.

    Args:
        path: Resource path expressed relative to ``keras_cv/``.

    Returns:
        Absolute filesystem path to the requested resource.

    Raises:
        FileNotFoundError: If the resource cannot be located in any of the
        supported search locations.
    """

    normalized_path = path.replace("/", os.sep)

    root_dir = get_project_root()
    package_root = os.path.dirname(root_dir)
    workspace_root = os.path.dirname(package_root)

    search_roots = []

    # Allow overriding the search root explicitly for custom setups.
    override_root = os.environ.get("KERAS_CV_DATA_ROOT")
    if override_root:
        search_roots.append(override_root)

    search_roots.extend(
        [
            root_dir,
            package_root,
            os.path.join(workspace_root, "bazel-bin", "keras_cv", "src"),
        ]
    )

    attempted_paths = []
    for candidate_root in search_roots:
        if not candidate_root:
            continue

        candidate = os.path.normpath(
            os.path.join(candidate_root, normalized_path)
        )
        attempted_paths.append(candidate)
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Unable to locate resource '{}'. Looked in: {}".format(
            path, ", ".join(attempted_paths)
        )
    )


class LazySO:
    def __init__(self, relative_path):
        self.relative_path = relative_path
        self._ops = None

    @property
    def ops(self):
        if self._ops is None:
            self.display_warning_if_incompatible()
            self._ops = tf.load_op_library(
                get_path_to_datafile(self.relative_path)
            )
        return self._ops

    def display_warning_if_incompatible(self):
        global abi_warning_already_raised
        if abi_warning_already_raised or abi_is_compatible():
            return

        user_version = tf.__version__
        warnings.warn(
            f"You are currently using TensorFlow {user_version} and "
            f"trying to load a KerasCV custom op.\n"
            f"KerasCV has compiled its custom ops against TensorFlow "
            f"{TF_VERSION_FOR_ABI_COMPATIBILITY}, and there are no "
            f"compatibility guarantees between the two versions.\n"
            "This means that you might get segfaults when loading the custom "
            "op, or other kind of low-level errors.\n"
            "If you do, do not file an issue on Github. "
            "This is a known limitation.",
            UserWarning,
        )
        abi_warning_already_raised = True


def abi_is_compatible():
    return tf.__version__.startswith(TF_VERSION_FOR_ABI_COMPATIBILITY)
