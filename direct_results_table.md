| K_B | n | p^(n) unreduced | p^(n) reduced (results.json) | difference | formulation | LP size (vars / rows / nnz) | solve time |
|---|---|---|---|---|---|---|---|
| classical bit | 1 | 1.0000000000 | 1.0000000000 | +0.0e+00 | explicit Kronecker LP | 36 / 77 / 3.4e+02 | 0 s |
| classical bit | 2 | 0.7500000000 | 0.7500000000 | +0.0e+00 | explicit Kronecker LP | 216 / 675 / 6.1e+03 | 0 s |
| classical bit | 3 | 0.7500000000 | 0.7500000000 | +0.0e+00 | explicit Kronecker LP | 1,296 / 5,609 / 1.1e+05 | 0 s |
| classical bit | 4 | 0.7500000000 | 0.7500000000 | -4.4e-16 | explicit Kronecker LP | 7,776 / 45,081 / 1.9e+06 | 4 s |
| classical bit | 5 | 0.7500000000 | 0.7500000000 | +3.3e-16 | explicit Kronecker LP | 46,656 / 355,457 / 3.4e+07 | 92 s |
| square gbit | 1 | 1.5000000000 | 1.5000000000 | -6.7e-16 | explicit Kronecker LP | 81 / 275 / 3.5e+03 | 0 s |
| square gbit | 2 | 1.0000000000 | 1.0000000000 | +1.1e-15 | explicit Kronecker LP | 729 / 4,583 / 2.5e+05 | 0 s |
| square gbit | 3 | 1.0000000000 | 1.0000000000 | +1.7e-13 | explicit Kronecker LP | 6,561 / 72,827 / 1.8e+07 | 42 s |
| pentagon | 1 | 1.3416407865 | 1.3416407865 | +2.0e-15 | explicit Kronecker LP | 81 / 419 / 7.1e+03 | 0 s |
| pentagon | 2 | 0.9208203932 | 0.9208203932 | -1.4e-14 | explicit Kronecker LP | 729 / 8,487 / 5.9e+05 | 3 s |
| pentagon | 3 | 0.9042054869 | 0.9042054869 | -3.0e-14 | explicit Kronecker LP | 6,561 / 167,291 / 5.0e+07 | 1012 s |
| pentagon | 4 | 0.8948818927 | 0.8948818927 | +6.0e-14 | tree LP (complete facet family) | 2,569,869 / 5,802,675 / 2.7e+07 | 9485 s |
| pentagon+ineq | 1 | 1.1733126292 | 1.1733126292 | -1.4e-14 | explicit Kronecker LP | 81 / 439 / 7.2e+03 | 0 s |
| pentagon+ineq | 2 | 0.9181365515 | 0.9181365515 | -3.0e-14 | explicit Kronecker LP | 729 / 8,887 / 6.1e+05 | 3 s |
| pentagon+ineq | 3 | 0.8952138949 | 0.8952138949 | -9.1e-13 | explicit Kronecker LP | 6,561 / 175,291 / 5.1e+07 | 1529 s |
