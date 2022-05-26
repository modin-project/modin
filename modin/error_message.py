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
from modin.logging import logger_decorator


class ErrorMessage(object):
    # Only print full ``default to pandas`` warning one time.
    printed_default_to_pandas = False
    printed_warnings = set()

    @classmethod
    @logger_decorator("MODIN-ERROR", "ErrorMessage.not_implemented", "debug")
    def not_implemented(cls, message=""):
        if message == "":
            message = "This functionality is not yet available in Modin."
        raise NotImplementedError(
            f"{message}\n"
            + "To request implementation, file an issue at "
            + "https://github.com/modin-project/modin/issues or, if that's "
            + "not possible, send an email to feature_requests@modin.org."
        )

    @classmethod
    @logger_decorator("MODIN-ERROR", "ErrorMessage.single_warning", "debug")
    def single_warning(cls, message):
        message_hash = hash(message)
        if message_hash in cls.printed_warnings:
            return

        warnings.warn(message)
        cls.printed_warnings.add(message_hash)

    @classmethod
    @logger_decorator("MODIN-ERROR", "ErrorMessage.default_to_pandas", "debug")
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
        warnings.warn(message)

    @classmethod
    @logger_decorator(
        "MODIN-ERROR", "ErrorMessage.catch_bugs_and_request_email", "debug"
    )
    def catch_bugs_and_request_email(cls, failure_condition, extra_log=""):
        if failure_condition:
            raise Exception(
                "Internal Error. "
                + "Please visit https://github.com/modin-project/modin/issues "
                + "to file an issue with the traceback and the command that "
                + "caused this error. If you can't file a GitHub issue, "
                + f"please email bug_reports@modin.org.\n{extra_log}"
            )

    @classmethod
    @logger_decorator("MODIN-ERROR", "ErrorMessage.non_verified_udf", "debug")
    def non_verified_udf(cls):
        warnings.warn(
            "User-defined function verification is still under development in Modin. "
            + "The function provided is not verified."
        )

    @classmethod
    @logger_decorator("MODIN-ERROR", "ErrorMessage.mismatch_with_pandas", "debug")
    def missmatch_with_pandas(cls, operation, message):
        cls.single_warning(
            f"`{operation}` implementation has mismatches with pandas:\n{message}."
        )

    @classmethod
    @logger_decorator("MODIN-ERROR", "ErrorMessage.not_initialized", "debug")
    def not_initialized(cls, engine, code):
        warnings.warn(
            f"{engine} execution environment not yet initialized. Initializing...\n"
            + "To remove this warning, run the following python code before doing dataframe operations:\n"
            + f"{code}"
        )
