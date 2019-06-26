# Development roadmap

## Design notes

This project (scikit-allel-model) is intended to provide array
functions and classes for representing and transforming genetic
variation data. It factors out and rewrites the functionality
currently available in the scikit-allel version 1.x modules under the
`allel.model` package.

Why split this functionality out into a separate project? Primarily to
make it easier to maintain. The current scikit-allel code base is
relatively large and complex, and does not have consistent standards
regarding whether or not unit test coverage is expected or even if
unit tests are required. The data model functionality is relatively
self-contained and straightforward to test, and so separating it out
means that development and maintenance processes and infrastructure
can be simplified.

Why rewrite it? Several reasons:

* We've learned the hard way that **writing dumb code** and **avoiding
  indirection** are excellent pieces of advice which make maintaining
  a code base much easier, as well as making it easier for others to
  review and contribute code. On reflection, there is way too much
  cleverness and indirection in the current `allel.model` code.
  
* Some of the existing `allel.model` code is rarely used and just too
  complex to be worth maintaining. If dropped, it would allow us to
  focus our limited energies on core functionalities.
  
* There are a number of specific opportunities to simplify the
  underlying code. In particular, **numba** has come a long way, and
  now provides a great way to write fast code without having to use
  cython or complicated numpy tricks. Cython is a wonderful tool but
  adds complexity both in the implementation and in the packaging and
  distribution. Numpy is also wonderful but things can get very
  obscure when algorithms are using several vectorized operations with
  lots of broadcasting and indexing tricks. Numba allows
  straightforward algorithm implementations using plain Python code.
  
* There are specific opportunities to simplify the API for users. For
  example, the current `allel.model` API exposes three classes for
  working with genotype data: `GenotypeArray`, `GenotypeDaskArray` and
  `GenotypeChunkedArray`. Now that dask is mature, we can drop support
  for the "chunked" classes, e.g., `GenotypeChunkedArray`. It would
  also be simpler if we had just a single `GenotypeArray` class that
  could be used with either numpy or dask arrays as the underlying
  data container.
  
* There are some technical issue with the current `allel.model` code
  that arise when using dask with a distributed cluster. These issues
  only surface in a distributed setting because functions are being
  pickled and sent to workers where they are unpickled. We need to
  anticipate and accommodate that type of usage.

Below are some further details on the approach being taken.

### Use numba

@@TODO

### Pure functions layer

@@TODO i.e., no wrapper classes

### Unify and simplify wrapper classes

@@TODO

### Avoid cryptic errors

@@TODO

### User and developer namespaces

@@TODO explain how `allel` and `skallel` are being used.
