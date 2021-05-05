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

import pytest
import modin.utils


# Note: classes below are used for purely testing purposes - they
# simulate real-world use cases for _inherit_docstring
class BaseParent:
    def method(self):
        """ordinary method (base)"""

    def base_method(self):
        """ordinary method in base only"""

    @property
    def prop(self):
        """property"""

    @staticmethod
    def static():
        """static method"""

    @classmethod
    def clsmtd(cls):
        """class method"""


class BaseChild(BaseParent):
    """this is class docstring"""

    def method(self):
        """ordinary method (child)"""

    def own_method(self):
        """own method"""

    def no_overwrite(self):
        """another own method"""

    F = property(method)


@pytest.fixture(scope="module")
def wrapped_cls():
    @modin.utils._inherit_docstrings(BaseChild)
    class Wrapped:
        def method(self):
            pass

        def base_method(self):
            pass

        def own_method(self):
            pass

        def no_overwrite(self):
            """not overwritten doc"""

        @property
        def prop(self):
            return None

        @staticmethod
        def static():
            pass

        @classmethod
        def clsmtd(cls):
            pass

        F = property(method)

    return Wrapped


def _check_doc(wrapped, orig):
    assert wrapped.__doc__ == orig.__doc__
    if isinstance(wrapped, property):
        assert wrapped.fget.__doc_inherited__
    else:
        assert wrapped.__doc_inherited__


def test_doc_inherit_clslevel(wrapped_cls):
    _check_doc(wrapped_cls, BaseChild)



def test_doc_inherit_methods(wrapped_cls):
    _check_doc(wrapped_cls.method, BaseChild.method)
    _check_doc(wrapped_cls.base_method, BaseParent.base_method)
    _check_doc(wrapped_cls.own_method, BaseChild.own_method)
    assert wrapped_cls.no_overwrite.__doc__ != BaseChild.no_overwrite.__doc__
    assert not getattr(wrapped_cls.no_overwrite, "__doc_inherited__", False)


def test_doc_inherit_special(wrapped_cls):
    _check_doc(wrapped_cls.static, BaseChild.static)
    _check_doc(wrapped_cls.clsmtd, BaseChild.clsmtd)


def test_doc_inherit_props(wrapped_cls):
    assert type(wrapped_cls.method) == type(BaseChild.method)  # noqa: E721
    _check_doc(wrapped_cls.prop, BaseChild.prop)
    _check_doc(wrapped_cls.F, BaseChild.F)


def test_doc_inherit_prop_builder():
    def builder(name):
        return property(lambda self: name)

    class Parent:
        prop = builder("Parent")

    @modin.utils._inherit_docstrings(Parent)
    class Child(Parent):
        prop = builder("Child")

    assert Parent().prop == "Parent"
    assert Child().prop == "Child"
