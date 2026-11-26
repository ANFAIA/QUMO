"""GENERATED from agents/spec/guard_rules.json by agents/spec/gen_mojo_rules.py. Do not edit:
change the JSON and run the script. The lists of the security rules both agent harnesses
share (agents/spec/README.md)."""


comptime GUARD_RULES_SCHEMA = "qumo.guard_rules.v1"
comptime MAX_REDIRECTS = 3
comptime RESEARCH_SCHEME = "https"
comptime RESEARCH_PORT = 443
comptime ARG_OTHER_CHARS = "_./:=,@+-"
comptime GIT_FLAGS_TEXT = "-- --cached --name-only --name-status --staged --stat"
comptime QUMO_SUBCOMMANDS_TEXT = "convert describe draw emulate expect help info run statevector transpile validate version vqe"
comptime UNTRUSTED_MARKER = "UNTRUSTED-DATA"
comptime MAX_ANSWER_CHARS = 6000

def skip_dirs() -> List[String]:
    """Directories no agent tool reads (compared without regard to ASCII case)."""
    var out = List[String]()
    out.append(".git")
    out.append(".pixi")
    out.append(".venv")
    out.append(".worktrees")
    out.append("__pycache__")
    out.append("build")
    out.append("models")
    out.append("node_modules")
    return out^

def secret_name_prefixes() -> List[String]:
    """A file name starting with one of these is secret-looking."""
    var out = List[String]()
    out.append(".env")
    return out^

def secret_name_contains() -> List[String]:
    """A file name holding one of these is secret-looking."""
    var out = List[String]()
    out.append("secret")
    out.append("credential")
    out.append("id_rsa")
    return out^

def secret_name_suffixes() -> List[String]:
    """A file name ending in one of these is secret-looking."""
    var out = List[String]()
    out.append(".pem")
    out.append(".key")
    return out^

def arg_ranges() -> List[String]:
    """Character ranges of a run_check argument, each as `a-z`."""
    var out = List[String]()
    out.append("A-Z")
    out.append("a-z")
    out.append("0-9")
    return out^

def git_flags() -> List[String]:
    """Options run_check accepts after `git diff` / `git status --short`."""
    var out = List[String]()
    out.append("--stat")
    out.append("--name-only")
    out.append("--name-status")
    out.append("--cached")
    out.append("--staged")
    out.append("--")
    return out^

def qumo_subcommands() -> List[String]:
    """`build/qumo` subcommands run_check accepts."""
    var out = List[String]()
    out.append("run")
    out.append("statevector")
    out.append("expect")
    out.append("transpile")
    out.append("emulate")
    out.append("validate")
    out.append("info")
    out.append("draw")
    out.append("convert")
    out.append("vqe")
    out.append("describe")
    out.append("version")
    out.append("help")
    return out^

def qumo_output_options() -> List[String]:
    """`build/qumo` options that write a file."""
    var out = List[String]()
    out.append("-o")
    out.append("--output")
    return out^

def qumo_options() -> List[String]:
    """Other `build/qumo` options run_check accepts."""
    var out = List[String]()
    out.append("--shots")
    out.append("--seed")
    out.append("--params")
    out.append("--max-qubits")
    out.append("--max-iterations")
    out.append("--device")
    out.append("--precision")
    out.append("--json")
    out.append("--pretty")
    out.append("--max-amplitudes")
    out.append("--observable")
    out.append("--target")
    out.append("--level")
    out.append("--basis")
    out.append("--pipeline")
    out.append("--qasm")
    out.append("--compile")
    out.append("--no-compile")
    out.append("--width")
    out.append("--unicode")
    out.append("--to")
    out.append("--instructions")
    out.append("--layers")
    out.append("--entanglement")
    out.append("--optimizer")
    out.append("--gradient")
    out.append("--maxiter")
    out.append("--qubits")
    out.append("-h")
    out.append("--help")
    return out^

def secret_env_substrings() -> List[String]:
    """An environment name holding one of these (any case) is a secret."""
    var out = List[String]()
    out.append("TOKEN")
    out.append("SECRET")
    out.append("KEY")
    out.append("PASSWORD")
    out.append("CREDENTIAL")
    out.append("COOKIE")
    return out^

def check_env_extra_substrings() -> List[String]:
    """Further substrings the Python check environment drops."""
    var out = List[String]()
    out.append("AUTH")
    out.append("SESSION")
    return out^

def git_injected_exact() -> List[String]:
    """Git configuration passed through the environment: whole names."""
    var out = List[String]()
    out.append("GIT_CONFIG_COUNT")
    out.append("GIT_CONFIG_PARAMETERS")
    return out^

def git_injected_numbered() -> List[String]:
    """Git configuration passed through the environment: prefixes of numbered names."""
    var out = List[String]()
    out.append("GIT_CONFIG_KEY_")
    out.append("GIT_CONFIG_VALUE_")
    return out^

def minimal_child_env_names() -> List[String]:
    """The only variables a child of the Mojo harness inherits."""
    var out = List[String]()
    out.append("HOME")
    out.append("LOGNAME")
    out.append("PATH")
    out.append("SHELL")
    out.append("TERM")
    out.append("USER")
    out.append("LANG")
    out.append("LC_ALL")
    out.append("LC_CTYPE")
    out.append("TMPDIR")
    return out^

def api_hosts() -> List[String]:
    """Hosts the research API tools may call."""
    var out = List[String]()
    out.append("export.arxiv.org")
    out.append("pypi.org")
    out.append("api.github.com")
    return out^

def web_groups() -> List[String]:
    """Tool groups that fetch from the web."""
    var out = List[String]()
    out.append("research")
    return out^

def exec_groups() -> List[String]:
    """Tool groups that run code."""
    var out = List[String]()
    out.append("check")
    return out^

def run_check_allowed() -> List[List[String]]:
    """The argv prefixes run_check accepts, in order."""
    var out = List[List[String]]()
    out.append([".venv/bin/mojo", "run", "-I", "."])
    out.append(["bench/.venv/bin/python", "examples/run_examples.py", "--only"])
    out.append(["bench/.venv/bin/python", "-m", "pytest", "-q", "plugins/hardware/tests"])
    out.append(["build/qumo"])
    out.append(["git", "diff"])
    out.append(["git", "status", "--short"])
    return out^
