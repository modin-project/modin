import warnings


class ErrorMessage(object):
    # Only print the request implementation one time. This only applies to Warnings.
    printed_request_implementation = False

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
    def catch_bugs_and_request_email(cls, failure_condition):
        if failure_condition:
            raise Exception(
                "Internal Error. "
                "Please email bug_reports@modin.org with the traceback and command that"
                " caused this error."
            )

    @classmethod
    def non_verified_udf(cls):
        warnings.warn(
            "User-defined function verification is still under development in Modin. "
            "The function provided is not verified."
        )
