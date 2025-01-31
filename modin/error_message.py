# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import warnings
from typing import NoReturn, Optional, Set

from modin.logging import get_logger
from modin.utils import get_current_execution


class ErrorMessage(object):
    # Only print full ``default to pandas`` warning one time.
    printed_default_to_pandas = False
    printed_warnings: Set[int] = set()  # Set of hashes of printed warnings

    @classmethod
    def not_implemented(cls, message: str = "") -> NoReturn:
        if message == "":
            message = "This functionality is not yet available in Modin."
        get_logger().info(f"Modin Error: NotImplementedError: {message}")
        raise NotImplementedError(
            f"{message}\n"
            + "To request implementation, file an issue at "
            + "https://github.com/modin-project/modin/issues or, if that's "
            + "not possible, send an email to feature_requests@modin.org."
        )

    @classmethod
    def single_warning(
        cls, message: str, category: Optional[type[Warning]] = None
    ) -> None:
        # note that there should not be identical messages with different categories since
        # only the message is used as the hash key.
        message_hash = hash(message)
        logger = get_logger()
        if message_hash in cls.printed_warnings:
            logger.debug(
                f"Modin Warning: Single Warning: {message} was raised and suppressed."
            )
            return

        logger.debug(f"Modin Warning: Single Warning: {message} was raised.")
        warnings.warn(message, category=category)
        cls.printed_warnings.add(message_hash)

    @classmethod
    def default_to_pandas(cls, message: str = "", reason: str = "") -> None:
        # TODO(https://github.com/modin-project/modin/issues/7429): Use
        # frame-level engine config.

        if message != "":
            execution_str = get_current_execution()
            message = (
                f"{message} is not currently supported by {execution_str}, "
                + "defaulting to pandas implementation."
            )
        else:
            message = "Defaulting to pandas implementation."

        if not cls.printed_default_to_pandas:
            message = (
                f"{message}\n"
                + "Please refer to "
                + "https://modin.readthedocs.io/en/stable/supported_apis/defaulting_to_pandas.html for explanation."
            )
            cls.printed_default_to_pandas = True
        if reason:
            message += f"\nReason: {reason}"
        get_logger().debug(f"Modin Warning: Default to pandas: {message}")
        warnings.warn(message)

    @classmethod
    def catch_bugs_and_request_email(
        cls, failure_condition: bool, extra_log: str = ""
    ) -> None:
        if failure_condition:
            get_logger().info(f"Modin Error: Internal Error: {extra_log}")
            raise Exception(
                "Internal Error. "
                + "Please visit https://github.com/modin-project/modin/issues "
                + "to file an issue with the traceback and the command that "
                + "caused this error. If you can't file a GitHub issue, "
                + f"please email bug_reports@modin.org.\n{extra_log}"
            )

    @classmethod
    def non_verified_udf(cls) -> None:
        get_logger().debug("Modin Warning: Non Verified UDF")
        warnings.warn(
            "User-defined function verification is still under development in Modin. "
            + "The function provided is not verified."
        )

    @classmethod
    def bad_type_for_numpy_op(cls, function_name: str, operand_type: type) -> None:
        cls.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for {function_name}, not {operand_type}. Defaulting to NumPy."
        )

    @classmethod
    def mismatch_with_pandas(cls, operation: str, message: str) -> None:
        get_logger().debug(
            f"Modin Warning: {operation} mismatch with pandas: {message}"
        )
        cls.single_warning(
            f"`{operation}` implementation has mismatches with pandas:\n{message}."
        )

    @classmethod
    def warn(cls, message: str) -> None:
        warnings.warn(message)

    @classmethod
    def not_initialized(cls, engine: str, code: str) -> None:
        get_logger().debug(f"Modin Warning: Not Initialized: {engine}")
        warnings.warn(
            f"{engine} execution environment not yet initialized. Initializing...\n"
            + "To remove this warning, run the following python code before doing dataframe operations:\n"
            + f"{code}"
        )
