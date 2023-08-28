# LLVM SingleSource Test Suite Comparison Table

So far, we have tested ClangIR **only against the SingleSource** tests from this suite.

The table below presents the Compile Time (CT) and Execution Time (ET) of each test, alongside the relative difference between ClangIR and the baseline for each metric. These metrics, however, are not a reliable performance comparison as many of these tests are too small, but rather a means to identify possible issues in ClangIR.

| Program | status | Base CT | ClangIR CT | Diff CT (%)  | Base ET | ClangIR ET | Diff ET (%) |
|---------|--------|--------------|--------------|--------------|-----------|-----------|-----------|
{% for row in site.data.clangir-singlesource-test-suite %}| {{ row.Program }} | {{ row.status }} | {{ row.compile_time_base | round: 3 }} | {{ row.compile_time_clangir | round: 3 }} | {{ row.compile_time_diff | times: 100 | round: 1 }} | {{ row.exec_time_base | round: 3 }} | {{ row.exec_time_clangir | round: 3 }} | {{row.exec_time_diff | times: 100 | round: 1 }} |
{% endfor %}
