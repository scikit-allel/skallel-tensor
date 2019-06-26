# Development roadmap

## Design notes

This project (scikit-allel-model) is intended to provide array
functions and classes for representing and transforming genetic
variation data. It factors out and rewrites the functionality
currently available in the scikit-allel version 1.x modules under the
[allel.model](https://github.com/cggh/scikit-allel/tree/master/allel/model)
package.

Why split this functionality out into a separate project? Primarily to
make it easier to maintain. The current scikit-allel code base is
relatively large and complex, and does not have consistent standards
regarding whether or not unit test coverage is expected or even if
unit tests are required. The data model functionality is relatively
self-contained and straightforward to test, and so separating it out
means that development and maintenance processes and infrastructure
can be simplified.

Why rewrite it? Several reasons:

* We've learned the hard way that [writing dumb
  code](https://matthewrocklin.com/blog/work/2018/01/27/write-dumb-code)
  and [avoiding
  indirection](https://matthewrocklin.com/blog/work/2019/06/23/avoid-indirection)
  are excellent pieces of advice which make maintaining a code base
  much easier, as well as making it easier for others to review and
  contribute code. On reflection, there is way too much cleverness and
  indirection in the current code base.
  
* Some of the existing code is rarely used and just too complex to be
  worth maintaining. If dropped, it would allow us to focus our
  limited energies on core functionalities.
  
* There are a number of specific opportunities to simplify the
  underlying code. In particular, [numba](https://numba.pydata.org/)
  has come a long way, and now provides a great way to write fast code
  without having to use cython or complicated numpy tricks. Cython is
  a wonderful tool but adds complexity both in the implementation and
  in the packaging and distribution. Numpy is also wonderful but
  things can get obscure when algorithms are using several vectorized
  operations with lots of broadcasting and indexing tricks. Numba
  allows straightforward algorithm implementations using plain Python
  code.
  
* There are specific opportunities to simplify the API for users. For
  example, the v1.x scikit-allel API exposes three classes for working
  with genotype data: `GenotypeArray`, `GenotypeDaskArray` and
  `GenotypeChunkedArray`. Now that dask is mature, we can drop support
  for the "chunked" classes, e.g., `GenotypeChunkedArray`. It would
  also be simpler if we had just a single `GenotypeArray` class that
  could be used with either numpy or dask arrays as the underlying
  data container.
  
* There are some technical issue with the v1.x code that arise when
  using dask with a distributed cluster. These issues only surface in
  a distributed setting because functions are being pickled and sent
  to workers where they are unpickled. We need to anticipate and
  accommodate that type of usage.

Below are some further details on the approach being taken.

### Use numba

The functions in the v1.x `allel.model` sub-modules are currently
written either using numpy's built-in vectorized functions or using a
cython module
([allel.opt.model](https://github.com/cggh/scikit-allel/blob/master/allel/opt/model.pyx)).

Any function previously written in cython will be rewritten using
numba's
[@jit](https://numba.pydata.org/numba-doc/latest/user/jit.html)
decorator in nopython (and nogil) mode. There are two
advantages. First, the code is written as pure Python, so no need to
understand any of the quirks of cython syntax. Second, the code can be
unit tested with coverage by disabling JIT compilation during the test
run. 100% test coverage should be a minimum requirement for code to
get into scikit-allel-model master branch, and so being able to get
coverage reports for all functions is vital. Numba code is generally
as fast as cython code, so there shouldn't be any performance
implications.

Wherever possible, any function previously written using numpy
vectorized functions will also be rewritten using numba's
[@jit](https://numba.pydata.org/numba-doc/latest/user/jit.html)
decorator in nopython (and nogil) mode. There are two main
advantages. First, code using vectorized functions can be hard to
understand for anyone who doesn't regularly use numpy. This is a
barrier to contribution, particularly for developers coming from other
languages. Writing the algorithm using good-old-fashioned for loops is
often clearer and simpler. Second, numba code is often faster than
equivalent numpy vectorized code, and avoids memory allocation for
intermediate results.

### Pure functions layer

We have run into problems with the current `allel.model` architecture
when used in a distributed computing environment, primarily because it
makes extensive use of wrapper classes like `GenotypeArray`, but these
don't have proper pickle/unpickle support. That could be fixed, but it
would be simpler if there was a base layer of functionality that used
only pure functions and did not involve any wrapper classes. This base
layer would not be exposed to the regular user, but would be used
internally to avoid any internal use of wrapper classes like
`GenotypeArray`.

### Unify and simplify wrapper classes

Currently there are three sets of wrapper classes, one for numpy
arrays (e.g., `GenotypeArray`), one for the legacy "chunked" computing
approach (e.g., `GenotypeChunkedArray`), and one for dask (e.g.,
`GenotypeDaskArray`).

Firstly, the "chunked" classes will get dropped completely, as these
are no longer necessary given the maturity of dask.

Secondly, the numpy and dask wrappers will be merged, so the user will
just use e.g. the `GenotypeArray` class, which can wrap either a numpy
or dask array as the underlying data container.

### Avoid cryptic errors

An issue with the current `allel.model` code base is that it tries to
be too helpful. For example, when a function requires a numpy array,
and the provided argument is not a numpy array, then it will be
converted to one internally. This is a common pattern in several
libraries, but then if something unexpected happened during the
conversion to numpy array, this can lead to cryptic errors
downstream. Second-guessing the user also adds a lot of complexity to
the code.

In the interests of both code maintainability and avoiding cryptic
errors, it would be better to place harder constraints on the types of
objects received by functions. If the user provides an incorrect type,
raise a `TypeError` with an informative message. I.e., fail fast, so
the user has some clue how to fix the problem themselves (e.g., by
converting the object to the correct type in their own code).

### User and developer namespaces

For technical reasons, there will be two root namespaces.

The `allel` namespace will be maintained for users, with all
user-facing functions and classes imported to the top level. So, e.g.,
the user can call `allel.GenotypeArray`.

However, the functions and classes that go into the `allel` namespace
will be provisioned from different sub-projects, one of which is this
project (scikit-allel-model), and there will be other sub-projects
(e.g., probably scikit-allel-io and scikit-allel-stats
subprojects). 

To make this possible, each sub-project has to use a [namespace
package](https://www.python.org/dev/peps/pep-0420/) without an
`__init__.py` file. The `skallel` namespace package will be used as
the root package in all sub-projects. The `skallel` package can be
thought of as the "developer namespace".

There will then be a scikit-allel meta-package, which declares the
`allel` user namespace and imports all user-facing functions and
classes from sub-projects.

So in summary, there will be the following projects, each with their
own github repo and pypi distributions:

* scikit-allel (meta-package, provides the `allel` package)
* scikit-allel-model (provides `skallel.model`)
* scikit-allel-io (provides `skallel.io`)
* scikit-allel-stats (provides `skallel.stats`)

From a user point of view this will be all hidden, so the user can
still install with, e.g.:

```
pip install scikit-allel==2.0.0
```

...and work with an API very similar to the current v1.x API, e.g.:


```
import allel
data = ...  # a numpy or dask array
gt = allel.GenotypeArray(data)
```
