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
from modin.logging import get_logger


class ErrorMessage(object):
    # Only print full ``default to pandas`` warning one time.
    printed_default_to_pandas = False
    printed_warnings = set()

    @classmethod
    def not_implemented(cls, message=""):
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
    def single_warning(cls, message):
        message_hash = hash(message)
        logger = get_logger()
        if message_hash in cls.printed_warnings:
            logger.debug(
                f"Modin Warning: Single Warning: {message} was raised and suppressed."
            )
            return

        logger.debug(f"Modin Warning: Single Warning: {message} was raised.")
        warnings.warn(message)
        cls.printed_warnings.add(message_hash)

    @classmethod
    def default_to_pandas(cls, message=""):
        if message != "":
            message = f"{message} defaulting to pandas implementation."
        else:
            message = "Defaulting to pandas implementation."

        if not cls.printed_default_to_pandas:
            message = (
                f"{message}\n"
                + "Please refer to "
                + "https://modin.readthedocs.io/en/stable/supported_apis/defaulting_to_pandas.html for explanation."
            )
            cls.printed_default_to_pandas = True
        get_logger().debug(f"Modin Warning: Default to pandas: {message}")
        warnings.warn(message)

    @classmethod
    def catch_bugs_and_request_email(cls, failure_condition, extra_log=""):
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
    def non_verified_udf(cls):
        get_logger().debug("Modin Warning: Non Verified UDF")
        warnings.warn(
            "User-defined function verification is still under development in Modin. "
            + "The function provided is not verified."
        )

    @classmethod
    def missmatch_with_pandas(cls, operation, message):
        get_logger().debug(
            f"Modin Warning: {operation} mismatch with pandas: {message}"
        )
        cls.single_warning(
            f"`{operation}` implementation has mismatches with pandas:\n{message}."
        )

    @classmethod
    def not_initialized(cls, engine, code):
        get_logger().debug(f"Modin Warning: Not Initialized: {engine}")
        warnings.warn(
            f"{engine} execution environment not yet initialized. Initializing...\n"
            + "To remove this warning, run the following python code before doing dataframe operations:\n"
            + f"{code}"
        )
