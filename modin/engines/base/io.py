from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from collections import OrderedDict
from modin.error_message import ErrorMessage
from modin.backends.base.query_compiler import BaseQueryCompiler


class BaseIO(object):
    @classmethod
    def from_pandas(cls, df):
        return cls.query_compiler_cls.from_pandas(df, cls.frame_mgr_cls)

    @classmethod
    def read_parquet(cls, path, engine, columns, **kwargs):
        """Load a parquet object from the file path, returning a DataFrame.
           Ray DataFrame only supports pyarrow engine for now.

        Args:
            path: The filepath of the parquet file.
                  We only support local files for now.
            engine: Ray only support pyarrow reader.
                    This argument doesn't do anything for now.
            kwargs: Pass into parquet's read_pandas function.

        Notes:
            ParquetFile API is used. Please refer to the documentation here
            https://arrow.apache.org/docs/python/parquet.html
        """
        ErrorMessage.default_to_pandas("`read_parquet`")
        return cls.from_pandas(pandas.read_parquet(path, engine, columns, **kwargs))

    @classmethod
    def read_csv(
        cls,
        filepath_or_buffer,
        sep=",",
        delimiter=None,
        header="infer",
        names=None,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=None,
        mangle_dupe_cols=True,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skipinitialspace=False,
        skiprows=None,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        parse_dates=False,
        infer_datetime_format=False,
        keep_date_col=False,
        date_parser=None,
        dayfirst=False,
        iterator=False,
        chunksize=None,
        compression="infer",
        thousands=None,
        decimal=b".",
        lineterminator=None,
        quotechar='"',
        quoting=0,
        escapechar=None,
        comment=None,
        encoding=None,
        dialect=None,
        tupleize_cols=None,
        error_bad_lines=True,
        warn_bad_lines=True,
        skipfooter=0,
        doublequote=True,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
    ):
        kwargs = {
            "filepath_or_buffer": filepath_or_buffer,
            "sep": sep,
            "delimiter": delimiter,
            "header": header,
            "names": names,
            "index_col": index_col,
            "usecols": usecols,
            "squeeze": squeeze,
            "prefix": prefix,
            "mangle_dupe_cols": mangle_dupe_cols,
            "dtype": dtype,
            "engine": engine,
            "converters": converters,
            "true_values": true_values,
            "false_values": false_values,
            "skipinitialspace": skipinitialspace,
            "skiprows": skiprows,
            "nrows": nrows,
            "na_values": na_values,
            "keep_default_na": keep_default_na,
            "na_filter": na_filter,
            "verbose": verbose,
            "skip_blank_lines": skip_blank_lines,
            "parse_dates": parse_dates,
            "infer_datetime_format": infer_datetime_format,
            "keep_date_col": keep_date_col,
            "date_parser": date_parser,
            "dayfirst": dayfirst,
            "iterator": iterator,
            "chunksize": chunksize,
            "compression": compression,
            "thousands": thousands,
            "decimal": decimal,
            "lineterminator": lineterminator,
            "quotechar": quotechar,
            "quoting": quoting,
            "escapechar": escapechar,
            "comment": comment,
            "encoding": encoding,
            "dialect": dialect,
            "tupleize_cols": tupleize_cols,
            "error_bad_lines": error_bad_lines,
            "warn_bad_lines": warn_bad_lines,
            "skipfooter": skipfooter,
            "doublequote": doublequote,
            "delim_whitespace": delim_whitespace,
            "low_memory": low_memory,
            "memory_map": memory_map,
            "float_precision": float_precision,
        }
        ErrorMessage.default_to_pandas("`read_csv`")
        return cls._read(**kwargs)

    @classmethod
    def _read(cls, **kwargs):
        """Read csv file from local disk.
        Args:
            filepath_or_buffer:
                  The filepath of the csv file.
                  We only support local files for now.
            kwargs: Keyword arguments in pandas.read_csv
        """
        pd_obj = pandas.read_csv(**kwargs)
        if isinstance(pd_obj, pandas.DataFrame):
            return cls.from_pandas(pd_obj)
        if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
            # Overwriting the read method should return a ray DataFrame for calls
            # to __next__ and get_chunk
            pd_read = pd_obj.read
            pd_obj.read = lambda *args, **kwargs: cls.from_pandas(
                pd_read(*args, **kwargs)
            )
        return pd_obj

    @classmethod
    def read_json(
        cls,
        path_or_buf=None,
        orient=None,
        typ="frame",
        dtype=True,
        convert_axes=True,
        convert_dates=True,
        keep_default_dates=True,
        numpy=False,
        precise_float=False,
        date_unit=None,
        encoding=None,
        lines=False,
        chunksize=None,
        compression="infer",
    ):
        ErrorMessage.default_to_pandas("`read_json`")
        kwargs = {
            "path_or_buf": path_or_buf,
            "orient": orient,
            "typ": typ,
            "dtype": dtype,
            "convert_axes": convert_axes,
            "convert_dates": convert_dates,
            "keep_default_dates": keep_default_dates,
            "numpy": numpy,
            "precise_float": precise_float,
            "date_unit": date_unit,
            "encoding": encoding,
            "lines": lines,
            "chunksize": chunksize,
            "compression": compression,
        }
        return cls.from_pandas(pandas.read_json(**kwargs))

    @classmethod
    def read_gbq(
        cls,
        query,
        project_id=None,
        index_col=None,
        col_order=None,
        reauth=False,
        auth_local_webserver=False,
        dialect=None,
        location=None,
        configuration=None,
        credentials=None,
        private_key=None,
        verbose=None,
    ):
        ErrorMessage.default_to_pandas("`read_gbq`")
        return cls.from_pandas(
            pandas.read_gbq(
                query,
                project_id=project_id,
                index_col=index_col,
                col_order=col_order,
                reauth=reauth,
                auth_local_webserver=auth_local_webserver,
                dialect=dialect,
                location=location,
                configuration=configuration,
                credentials=credentials,
                private_key=private_key,
                verbose=verbose,
            )
        )

    @classmethod
    def read_html(
        cls,
        io,
        match=".+",
        flavor=None,
        header=None,
        index_col=None,
        skiprows=None,
        attrs=None,
        parse_dates=False,
        tupleize_cols=None,
        thousands=",",
        encoding=None,
        decimal=".",
        converters=None,
        na_values=None,
        keep_default_na=True,
        displayed_only=True,
    ):
        ErrorMessage.default_to_pandas("`read_html`")
        kwargs = {
            "io": io,
            "match": match,
            "flavor": flavor,
            "header": header,
            "index_col": index_col,
            "skiprows": skiprows,
            "attrs": attrs,
            "parse_dates": parse_dates,
            "tupleize_cols": tupleize_cols,
            "thousands": thousands,
            "encoding": encoding,
            "decimal": decimal,
            "converters": converters,
            "na_values": na_values,
            "keep_default_na": keep_default_na,
            "displayed_only": displayed_only,
        }
        return cls.from_pandas(pandas.read_html(**kwargs)[0])

    @classmethod
    def read_clipboard(cls, sep=r"\s+", **kwargs):  # pragma: no cover
        ErrorMessage.default_to_pandas("`read_clipboard`")
        return cls.from_pandas(pandas.read_clipboard(sep=sep, **kwargs))

    @classmethod
    def read_excel(
        cls,
        io,
        sheet_name=0,
        header=0,
        names=None,
        index_col=None,
        parse_cols=None,
        usecols=None,
        squeeze=False,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skiprows=None,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        verbose=False,
        parse_dates=False,
        date_parser=None,
        thousands=None,
        comment=None,
        skip_footer=0,
        skipfooter=0,
        convert_float=True,
        mangle_dupe_cols=True,
        **kwds
    ):
        if skip_footer != 0:
            skipfooter = skip_footer
        ErrorMessage.default_to_pandas("`read_excel`")
        intermediate = pandas.read_excel(
            io,
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            parse_cols=parse_cols,
            usecols=usecols,
            squeeze=squeeze,
            dtype=dtype,
            engine=engine,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            verbose=verbose,
            parse_dates=parse_dates,
            date_parser=date_parser,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            convert_float=convert_float,
            mangle_dupe_cols=mangle_dupe_cols,
            **kwds
        )
        if isinstance(intermediate, OrderedDict):
            parsed = OrderedDict()
            for key in intermediate.keys():
                parsed[key] = cls.from_pandas(intermediate.get(key))
            return parsed
        else:
            return cls.from_pandas(intermediate)

    @classmethod
    def read_hdf(cls, path_or_buf, key=None, mode="r", columns=None):
        ErrorMessage.default_to_pandas("`read_hdf`")
        return cls.from_pandas(
            pandas.read_hdf(path_or_buf, key=key, mode=mode, columns=columns)
        )

    @classmethod
    def read_feather(cls, path, columns=None, use_threads=True):
        ErrorMessage.default_to_pandas("`read_feather`")
        return cls.from_pandas(
            pandas.read_feather(path, columns=columns, use_threads=use_threads)
        )

    @classmethod
    def read_msgpack(cls, path_or_buf, encoding="utf-8", iterator=False, **kwargs):
        ErrorMessage.default_to_pandas("`read_msgpack`")
        return cls.from_pandas(
            pandas.read_msgpack(
                path_or_buf, encoding=encoding, iterator=iterator, **kwargs
            )
        )

    @classmethod
    def read_stata(
        cls,
        filepath_or_buffer,
        convert_dates=True,
        convert_categoricals=True,
        encoding=None,
        index_col=None,
        convert_missing=False,
        preserve_dtypes=True,
        columns=None,
        order_categoricals=True,
        chunksize=None,
        iterator=False,
    ):
        ErrorMessage.default_to_pandas("`read_stata`")
        kwargs = {
            "filepath_or_buffer": filepath_or_buffer,
            "convert_dates": convert_dates,
            "convert_categoricals": convert_categoricals,
            "encoding": encoding,
            "index_col": index_col,
            "convert_missing": convert_missing,
            "preserve_dtypes": preserve_dtypes,
            "columns": columns,
            "order_categoricals": order_categoricals,
            "chunksize": chunksize,
            "iterator": iterator,
        }
        return cls.from_pandas(pandas.read_stata(**kwargs))

    @classmethod
    def read_sas(
        cls,
        filepath_or_buffer,
        format=None,
        index=None,
        encoding=None,
        chunksize=None,
        iterator=False,
    ):  # pragma: no cover
        ErrorMessage.default_to_pandas("`read_sas`")
        return cls.from_pandas(
            pandas.read_sas(
                filepath_or_buffer,
                format=format,
                index=index,
                encoding=encoding,
                chunksize=chunksize,
                iterator=iterator,
            )
        )

    @classmethod
    def read_pickle(cls, path, compression="infer"):
        ErrorMessage.default_to_pandas("`read_pickle`")
        return cls.from_pandas(pandas.read_pickle(path, compression=compression))

    @classmethod
    def read_sql(
        cls,
        sql,
        con,
        index_col=None,
        coerce_float=True,
        params=None,
        parse_dates=None,
        columns=None,
        chunksize=None,
    ):
        ErrorMessage.default_to_pandas("`read_sql`")
        return cls.from_pandas(
            pandas.read_sql(
                sql,
                con,
                index_col=index_col,
                coerce_float=coerce_float,
                params=params,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
            )
        )

    @classmethod
    def read_fwf(
        cls, filepath_or_buffer, colspecs="infer", widths=None, infer_nrows=100, **kwds
    ):
        ErrorMessage.default_to_pandas("`read_fwf`")
        return cls.from_pandas(
            pandas.read_fwf(
                filepath_or_buffer,
                colspecs=colspecs,
                widths=widths,
                infer_nrows=infer_nrows,
                **kwds
            )
        )

    @classmethod
    def read_sql_table(
        cls,
        table_name,
        con,
        schema=None,
        index_col=None,
        coerce_float=True,
        parse_dates=None,
        columns=None,
        chunksize=None,
    ):
        ErrorMessage.default_to_pandas("`read_sql_table`")
        return cls.from_pandas(
            pandas.read_sql_table(
                table_name,
                con,
                schema=schema,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
            )
        )

    @classmethod
    def read_sql_query(
        cls,
        sql,
        con,
        index_col=None,
        coerce_float=True,
        params=None,
        parse_dates=None,
        chunksize=None,
    ):
        ErrorMessage.default_to_pandas("`read_sql_query`")
        return cls.from_pandas(
            pandas.read_sql_query(
                sql,
                con,
                index_col=index_col,
                coerce_float=coerce_float,
                params=params,
                parse_dates=parse_dates,
                chunksize=chunksize,
            )
        )

    @classmethod
    def to_sql(
        cls,
        qc,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
    ):
        ErrorMessage.default_to_pandas("`to_sql`")
        df = qc.to_pandas()
        df.to_sql(
            name=name,
            con=con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )

    @classmethod
    def to_pickle(cls, obj, path, compression="infer", protocol=4):
        if protocol == 4:
            # This forces pandas to use default pickling, which is different for python3
            # and python2
            protocol = -1
        ErrorMessage.default_to_pandas("`to_pickle`")
        if isinstance(obj, BaseQueryCompiler):
            return pandas.to_pickle(
                obj.to_pandas(), path, compression=compression, protocol=protocol
            )
        else:
            return pandas.to_pickle(
                obj, path, compression=compression, protocol=protocol
            )
