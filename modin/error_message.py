class ErrorMessage(object):
    @classmethod
    def not_implemented(cls, message=""):
        if message == "":
            message = "This functionality is not yet available in Modin."
        raise NotImplementedError(
            "{}\n"
            "To request implementation, send an email to dev@modin.org".format(message)
        )

    @classmethod
    def default_to_pandas(cls, message=""):
        if message != "":
            message = "WARN: {} Defaulting to pandas implementation.".format(message)
        else:
            message = "WARN: Defaulting to pandas implementation."
        print(
            "{}\n"
            "WARN: To request implementation, send an email to dev@modin.org or create "
            "an issue: http://github.com/modin-project/modin/issues.".format(message)
        )

    @classmethod
    def catch_bugs_and_request_email(cls, failure_condition):
        if failure_condition:
            raise Exception(
                "Internal Error. "
                "Please email bugs@modin.org with the traceback and command that "
                "caused this error."
            )

    @classmethod
    def non_verified_udf(cls):
        print(
            "User-defined function verification is still under development in Modin. "
            "The function provided is not verified."
        )
