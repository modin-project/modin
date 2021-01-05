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


class ErrorMessage(object):
    # Only print the request implementation one time. This only applies to Warnings.
    printed_request_implementation = False
    printed_warnings = set()

    @classmethod
    def not_implemented(cls, message=""):
        if message == "":
            message = "This functionality is not yet available in Modin."
        raise NotImplementedError(
            "{}\n"
            "To request implementation, send an email to "
            "feature_requests@modin.org".format(message)
        )

    @classmethod
    def single_warning(cls, message):
        message_hash = hash(message)
        if message_hash in cls.printed_warnings:
            return

        warnings.warn(message)
        cls.printed_warnings.add(message_hash)

    @classmethod
    def default_to_pandas(cls, message=""):
        if message != "":
            message = "{} defaulting to pandas implementation.".format(message)
        else:
            message = "Defaulting to pandas implementation."

        if not cls.printed_request_implementation:
            message = (
                "{}\n".format(message)
                + "To request implementation, send an email to "
                + "feature_requests@modin.org."
            )
            cls.printed_request_implementation = True
        warnings.warn(message)

    @classmethod
    def catch_bugs_and_request_email(cls, failure_condition, extra_log=""):
        if failure_condition:
            raise Exception(
                "Internal Error. "
                "Please email bug_reports@modin.org with the traceback and command that"
                " caused this error.\n{}".format(extra_log)
            )

    @classmethod
    def non_verified_udf(cls):
        warnings.warn(
            "User-defined function verification is still under development in Modin. "
            "The function provided is not verified."
        )

    @classmethod
    def missmatch_with_pandas(cls, operation, message):
        cls.single_warning(
            f"`{operation}` implementation has mismatches with pandas:\n{message}."
        )

    @classmethod
    def not_initialized(cls, engine, code):
        warnings.warn(
            "{} execution environment not yet initialized. Initializing...\n"
            "To remove this warning, run the following python code before doing dataframe operations:\n"
            "{}".format(engine, code)
        )
