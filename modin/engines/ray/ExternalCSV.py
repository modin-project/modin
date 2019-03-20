import os
import re
import boto3

S3_REGEX = re.compile(u"s3:/+(.*?)/(.*)")


class BaseExternalCSV:
    def seek(self, pos, rel=True):
        pass

    def read(self, bytes=None):
        pass

    def readline(self):
        pass

    @property
    def pos(self):
        pass

    @property
    def size(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    pass


class LocalCSV(BaseExternalCSV):
    def __init__(self, path):
        self.path = path
        self._file_obj = open(self.path, "rb")
        self._size = os.stat(self.path).st_size

    def seek(self, pos, rel=True):
        self._file_obj.seek(pos, os.SEEK_CUR if rel else os.SEEK_SET)

    def read(self, bytes=-1):
        return self._file_obj.read(bytes)

    def readline(self):
        return self._file_obj.readline()

    @property
    def pos(self):
        return self._file_obj.tell()

    @property
    def size(self):
        return self._size

    def close(self):
        self._file_obj.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return self


class S3CSV(BaseExternalCSV):
    def __init__(self, path):
        path = str(path)

        pattern = S3_REGEX
        res = pattern.search(path)

        assert res is not None, "Invalid s3 path"

        self.bucket_name = res.group(1)
        self.object_name = res.group(2)

        self._s3 = boto3.resource("s3")
        self._bucket = self._s3.Bucket(self.bucket_name)
        self._object = self._bucket.Object(self.bucket_name)

        self._size = self._object.content_length

        # All these parameters are already set by self.seek and are here to stop the interpreter from complaining
        self._pos = 0
        self._result = None
        self._metadata = None
        self._li_buffer = False
        self.seek(0, rel=False)

    def seek(self, pos, rel=True):
        if rel:
            self._pos += pos
        else:
            self._pos = pos

        if self._result:
            self._result.close()
            self._result = None

        self._metadata = self._object.get(Range=("%i-" % self._pos))
        self._result = self._metadata["Body"]
        self._li_buffer = False

    def read(self, amt=None):
        if self._li_buffer:
            self.seek(0, rel=True)
        res = self._result.read(amt=amt)
        self._pos += len(res)
        return res

    def readline(self):
        if not self._li_buffer:
            self._li_buffer = self._result.iter_lines(amt=1000)
        res = next(self._li_buffer)
        self._pos += len(res)
        return res

    @property
    def pos(self):
        return self._pos

    @property
    def size(self):
        return self._size

    def close(self):
        self._result.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return self


def open_csv(file_path) -> BaseExternalCSV:
    # TODO: Devin please save me and tell me what to do here thanks
    if str(file_path).startswith("s3://"):
        pass
    else:
        return LocalCSV(file_path)
