import qumo_py


def pytest_report_header(config):
    return [f"qumo: {qumo_py.version()}"]


def pytest_benchmark_update_json(config, benchmarks, output_json):
    output_json["qumo_info"] = {"qumo": qumo_py.version()}
